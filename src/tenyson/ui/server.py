from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import parse_qs, urlparse
import webbrowser

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tenyson.core import wandb_store


_DEFAULT_HISTORY_LIMIT = 240
_DEFAULT_CACHE_TTL_SECONDS = 5.0
_INTERNAL_HISTORY_KEYS = {"_step", "_runtime", "_timestamp"}
_PHASE_ORDER = {"sft": 0, "eval": 1, "rl": 2}
_PREFERRED_HISTORY_KEYS = {
    "sft": [
        "train/loss",
        "eval/loss",
        "train/grad_norm",
        "train/learning_rate",
        "train/epoch",
    ],
    "rl": [
        "reward",
        "kl",
        "loss",
        "train/loss",
        "train/reward",
        "profiling/Time taken: UnslothGRPOTrainer.transformers.generate",
        "profiling/Time taken: UnslothGRPOTrainer.reward_wordle_strict",
    ],
    "eval": [
        "constraint_accuracy",
        "dict_accuracy",
        "format_accuracy",
        "total_samples",
    ],
}


def _normalize_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    normalized = _normalize_timestamp(value)
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_backend_ref(
    *,
    db_url: Optional[str],
    entity: Optional[str],
    project: Optional[str],
) -> str:
    candidate = str(db_url or os.getenv("TENYSON_DB_URL") or "").strip()
    if candidate:
        wandb_store.parse_backend_ref(candidate)
        return candidate

    resolved_entity = str(
        entity or os.getenv("TENYSON_WANDB_ENTITY") or os.getenv("WANDB_ENTITY") or ""
    ).strip()
    resolved_project = str(
        project or os.getenv("TENYSON_WANDB_PROJECT") or os.getenv("WANDB_PROJECT") or ""
    ).strip()
    if not resolved_entity or not resolved_project:
        raise ValueError(
            "Missing W&B target. Pass --db-url or set TENYSON_WANDB_ENTITY and "
            "TENYSON_WANDB_PROJECT."
        )
    backend_ref = f"wandb://{resolved_entity}/{resolved_project}"
    wandb_store.parse_backend_ref(backend_ref)
    return backend_ref


def _safe_summary_get(run: Any, key: str) -> Any:
    summary = getattr(run, "summary", None)
    if summary is None:
        return None
    try:
        return summary.get(key)
    except Exception:
        try:
            return summary[key]
        except Exception:
            return None


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Mapping):
        return dict(value.items())
    return {}


def _maybe_json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class _DashboardTokenizerFallback:
    def encode(self, text: Any, add_special_tokens: bool = False) -> List[int]:
        _unused = add_special_tokens
        return list(range(len(str(text or ""))))


def _is_wordle_like_eval(
    config_preview: Mapping[str, Any], results_payload: Mapping[str, Any]
) -> bool:
    task_cfg = _coerce_mapping(config_preview.get("task"))
    detailed_results = results_payload.get("detailed_results")
    if not isinstance(detailed_results, list) or not detailed_results:
        return False
    first_row = _coerce_mapping(detailed_results[0])
    if not first_row:
        return False
    return bool(task_cfg.get("wordlists")) and {
        "prompt",
        "completion",
    }.issubset(first_row.keys())


def _wordle_failure_reasons(scored: Mapping[str, Any]) -> List[str]:
    strict_ok = bool(scored.get("strict_ok"))
    dict_ok = bool(scored.get("is_wordle_valid"))
    totals = _coerce_mapping(scored.get("totals"))
    is_consistent = (
        strict_ok
        and dict_ok
        and int(scored.get("sat_count") or 0)
        == sum(int(value or 0) for value in totals.values())
    )
    reasons: List[str] = []
    if not strict_ok:
        reasons.append("format")
        return reasons
    if not dict_ok:
        reasons.append("dictionary")
    if not is_consistent:
        reasons.append("constraints")
    return reasons


def _maybe_enrich_wordle_eval_results(
    results_payload: Mapping[str, Any], config_preview: Mapping[str, Any]
) -> Dict[str, Any]:
    payload = _coerce_mapping(results_payload)
    if not _is_wordle_like_eval(config_preview, payload):
        return payload

    try:
        from examples.wordle.wordle_task import get_wordlists, score_completion
    except Exception:
        return payload

    task_cfg = _coerce_mapping(config_preview.get("task"))
    try:
        solutions, allowed = get_wordlists({"task": task_cfg})
    except Exception:
        return payload
    valid_set = set(solutions) | set(allowed)
    tokenizer = _DashboardTokenizerFallback()

    enriched_rows: List[Dict[str, Any]] = []
    for raw_row in payload.get("detailed_results", []):
        row = _coerce_mapping(raw_row)
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if not prompt and not completion:
            enriched_rows.append(row)
            continue
        try:
            scored = score_completion(
                prompt_text=prompt,
                completion_text=completion,
                valid_set=valid_set,
                task_cfg=task_cfg,
                tokenizer=tokenizer,
            )
        except Exception:
            enriched_rows.append(row)
            continue

        failure_reasons = _wordle_failure_reasons(scored)
        totals = _coerce_mapping(scored.get("totals"))
        merged = dict(row)
        merged.update(scored)
        merged["parsed_guess"] = row.get("parsed_guess", scored.get("parsed_guess"))
        merged["format_ok"] = bool(row.get("format_ok", scored.get("strict_ok")))
        merged["dict_ok"] = bool(row.get("dict_ok", scored.get("is_wordle_valid")))
        merged["consistent"] = (
            merged["format_ok"]
            and merged["dict_ok"]
            and int(scored.get("sat_count") or 0)
            == sum(int(value or 0) for value in totals.values())
        )
        merged["passed"] = len(failure_reasons) == 0
        merged["failure_reasons"] = failure_reasons
        enriched_rows.append(merged)

    payload["detailed_results"] = enriched_rows
    return payload


def _downsample_rows(rows: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    if max_points == 1:
        return [rows[-1]]
    selected: List[Dict[str, Any]] = []
    for index in range(max_points):
        source_index = round(index * (len(rows) - 1) / (max_points - 1))
        selected.append(rows[source_index])
    return selected


def _sort_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(run: Dict[str, Any]) -> tuple[Any, ...]:
        created_at = _parse_iso_timestamp(run.get("created_at"))
        created_ts = (
            -created_at.timestamp() if created_at is not None else float("inf")
        )
        phase = str(run.get("phase") or "")
        return (
            0 if run.get("is_active") else 1,
            created_ts,
            _PHASE_ORDER.get(phase, 99),
            str(run.get("run_name") or ""),
        )

    return sorted(runs, key=_sort_key)


def _count_statuses(runs: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for run in runs:
        status = str(run.get("status") or "unknown").strip().lower() or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


@dataclass
class DashboardServerConfig:
    backend_ref: str
    experiment_id: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8787
    refresh_seconds: float = 10.0
    open_browser: bool = False
    history_limit: int = _DEFAULT_HISTORY_LIMIT
    cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS


class DashboardDataService:
    def __init__(
        self,
        *,
        backend_ref: str,
        default_experiment_id: Optional[str] = None,
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
        history_limit: int = _DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.backend_ref = str(backend_ref)
        self.target = wandb_store.parse_backend_ref(self.backend_ref)
        self.default_experiment_id = (
            str(default_experiment_id or "").strip() or None
        )
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.history_limit = max(10, int(history_limit))
        self._cache_lock = threading.Lock()
        self._project_runs_cache: tuple[float, List[Any]] | None = None
        self._run_detail_cache: Dict[tuple[str, str, str], tuple[float, Dict[str, Any]]] = {}

    def project_url(self) -> str:
        return self.target.project_url

    def _project_runs(self) -> List[Any]:
        now = time.monotonic()
        with self._cache_lock:
            if self._project_runs_cache is not None:
                cached_at, cached_runs = self._project_runs_cache
                if (now - cached_at) <= self.cache_ttl_seconds:
                    return cached_runs

        import wandb

        api = wandb.Api()
        runs = list(api.runs(path=f"{self.target.entity}/{self.target.project}"))
        with self._cache_lock:
            self._project_runs_cache = (now, runs)
        return runs

    def _match_experiment_id(self, run: Any) -> Optional[str]:
        summary_experiment_id = str(
            _safe_summary_get(run, wandb_store.SUMMARY_EXPERIMENT_ID)
            or getattr(run, "group", "")
            or ""
        ).strip()
        return summary_experiment_id or None

    def _match_run(self, run: Any, *, experiment_id: str, phase: str, run_name: str) -> bool:
        summary = self._normalize_run_summary(run)
        return (
            summary.get("experiment_id") == experiment_id
            and summary.get("phase") == phase
            and summary.get("run_name") == run_name
        )

    def _normalize_run_summary(self, run: Any) -> Dict[str, Any]:
        metrics = _maybe_json_dict(_safe_summary_get(run, wandb_store.SUMMARY_METRICS_JSON))
        job_result = _maybe_json_dict(
            _safe_summary_get(run, wandb_store.SUMMARY_JOB_RESULT_JSON)
        )
        run_name = str(
            _safe_summary_get(run, wandb_store.SUMMARY_RUN_NAME)
            or getattr(run, "name", "")
            or ""
        ).strip()
        phase = str(
            _safe_summary_get(run, wandb_store.SUMMARY_PHASE)
            or getattr(run, "job_type", "")
            or ""
        ).strip()
        status = str(
            _safe_summary_get(run, wandb_store.SUMMARY_STATUS)
            or job_result.get("status")
            or "unknown"
        ).strip()
        total_time_seconds = _safe_summary_get(run, wandb_store.SUMMARY_TOTAL_TIME)
        if total_time_seconds is None:
            total_time_seconds = job_result.get("total_time_seconds")
        return {
            "experiment_id": self._match_experiment_id(run),
            "phase": phase,
            "run_name": run_name,
            "display_name": run_name,
            "status": status,
            "is_active": bool(_safe_summary_get(run, wandb_store.SUMMARY_IS_ACTIVE)),
            "provider": _safe_summary_get(run, wandb_store.SUMMARY_PROVIDER),
            "heartbeat_at": _normalize_timestamp(
                _safe_summary_get(run, wandb_store.SUMMARY_HEARTBEAT_AT)
            ),
            "created_at": _normalize_timestamp(getattr(run, "created_at", None)),
            "updated_at": _normalize_timestamp(getattr(run, "updated_at", None)),
            "wandb_url": str(
                _safe_summary_get(run, wandb_store.SUMMARY_WANDB_URL)
                or getattr(run, "url", "")
                or ""
            ).strip()
            or None,
            "metrics": metrics,
            "total_time_seconds": total_time_seconds,
            "failure_reason": _safe_summary_get(run, wandb_store.SUMMARY_FAILURE_REASON)
            or job_result.get("failure_reason"),
            "hf_repo_id": _safe_summary_get(run, wandb_store.SUMMARY_HF_REPO_ID)
            or job_result.get("hf_repo_id"),
            "hf_revision": _safe_summary_get(run, wandb_store.SUMMARY_HF_REVISION)
            or job_result.get("hf_revision"),
            "stopped_early": bool(job_result.get("stopped_early")),
            "processed_samples": job_result.get("processed_samples"),
            "expected_samples": job_result.get("expected_samples"),
            "stop_requested": bool(
                _safe_summary_get(run, wandb_store.SUMMARY_STOP_REQUESTED)
            ),
            "stop_requested_at": _normalize_timestamp(
                _safe_summary_get(run, wandb_store.SUMMARY_STOP_REQUESTED_AT)
            ),
            "attempt_token": _safe_summary_get(run, wandb_store.SUMMARY_ATTEMPT_TOKEN),
            "project_url": _safe_summary_get(run, wandb_store.SUMMARY_PROJECT_URL)
            or self.project_url(),
            "job_result_present": bool(job_result),
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for run in self._project_runs():
            experiment_id = self._match_experiment_id(run)
            if not experiment_id:
                continue
            summary = self._normalize_run_summary(run)
            bucket = grouped.setdefault(
                experiment_id,
                {
                    "experiment_id": experiment_id,
                    "run_count": 0,
                    "active_run_count": 0,
                    "phase_counts": {},
                    "status_counts": {},
                    "latest_activity_at": None,
                    "project_url": summary.get("project_url") or self.project_url(),
                },
            )
            bucket["run_count"] += 1
            if summary.get("is_active"):
                bucket["active_run_count"] += 1
            phase = str(summary.get("phase") or "unknown")
            bucket["phase_counts"][phase] = bucket["phase_counts"].get(phase, 0) + 1
            status = str(summary.get("status") or "unknown")
            bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
            latest = _parse_iso_timestamp(summary.get("heartbeat_at")) or _parse_iso_timestamp(
                summary.get("created_at")
            )
            previous = _parse_iso_timestamp(bucket.get("latest_activity_at"))
            if latest is not None and (previous is None or latest > previous):
                bucket["latest_activity_at"] = latest.isoformat()

        return sorted(
            grouped.values(),
            key=lambda item: _parse_iso_timestamp(item.get("latest_activity_at"))
            or datetime.fromtimestamp(0, tz=timezone.utc),
            reverse=True,
        )

    def get_experiment_snapshot(self, experiment_id: str) -> Dict[str, Any]:
        experiment_id = str(experiment_id or "").strip()
        if not experiment_id:
            raise ValueError("experiment_id is required.")

        runs = [
            self._normalize_run_summary(run)
            for run in self._project_runs()
            if self._match_experiment_id(run) == experiment_id
        ]
        runs = _sort_runs(runs)
        counts = _count_statuses(runs)
        active_runs = [run for run in runs if run.get("is_active")]
        latest_heartbeat_at = None
        for run in runs:
            heartbeat = _parse_iso_timestamp(run.get("heartbeat_at"))
            if heartbeat is None:
                continue
            if latest_heartbeat_at is None or heartbeat > latest_heartbeat_at:
                latest_heartbeat_at = heartbeat

        return {
            "experiment": {
                "experiment_id": experiment_id,
                "backend_ref": self.backend_ref,
                "project_url": self.project_url(),
                "run_count": len(runs),
                "active_run_count": len(active_runs),
                "status_counts": counts,
                "latest_heartbeat_at": latest_heartbeat_at.isoformat()
                if latest_heartbeat_at is not None
                else None,
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
            },
            "runs": runs,
        }

    def _build_history_payload(self, run: Any, *, phase: str) -> Dict[str, Any]:
        history_rows: List[Dict[str, Any]] = []
        numeric_keys: set[str] = set()
        try:
            iterator = run.scan_history(page_size=min(500, self.history_limit))
            for index, raw_row in enumerate(iterator):
                if index >= self.history_limit:
                    break
                row = dict(raw_row)
                normalized: Dict[str, Any] = {
                    "step": int(row.get("_step", index) or index),
                    "timestamp": row.get("_timestamp"),
                    "runtime": row.get("_runtime"),
                }
                for key, value in row.items():
                    if key in _INTERNAL_HISTORY_KEYS:
                        continue
                    if value is None:
                        continue
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        normalized[key] = float(value)
                        numeric_keys.add(key)
                history_rows.append(normalized)
        except Exception:
            return {"keys": [], "rows": [], "default_keys": []}

        history_rows = _downsample_rows(history_rows, self.history_limit)
        keys = sorted(numeric_keys)
        return {
            "keys": keys,
            "rows": history_rows,
            "default_keys": self._pick_default_history_keys(keys, phase=phase),
        }

    def _pick_default_history_keys(self, keys: List[str], *, phase: str) -> List[str]:
        preferred = _PREFERRED_HISTORY_KEYS.get(str(phase or ""), [])
        selected = [key for key in preferred if key in keys]
        if len(selected) >= 3:
            return selected[:3]
        selected_set = set(selected)
        for key in keys:
            if key in selected_set:
                continue
            if key.endswith("global_step"):
                continue
            selected.append(key)
            selected_set.add(key)
            if len(selected) >= 3:
                break
        if not selected and keys:
            return keys[: min(3, len(keys))]
        return selected

    def _find_run(self, *, experiment_id: str, phase: str, run_name: str) -> Any:
        matches = [
            run
            for run in self._project_runs()
            if self._match_run(
                run,
                experiment_id=experiment_id,
                phase=phase,
                run_name=run_name,
            )
        ]
        if matches:
            matches.sort(
                key=lambda run: (
                    _parse_iso_timestamp(
                        _safe_summary_get(run, wandb_store.SUMMARY_HEARTBEAT_AT)
                    )
                    or _parse_iso_timestamp(getattr(run, "updated_at", None))
                    or _parse_iso_timestamp(getattr(run, "created_at", None))
                    or datetime.fromtimestamp(0, tz=timezone.utc)
                ),
                reverse=True,
            )
            return matches[0]
        return wandb_store.fetch_run(
            self.backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
        )

    def get_run_detail(self, *, experiment_id: str, phase: str, run_name: str) -> Dict[str, Any]:
        cache_key = (str(experiment_id), str(phase), str(run_name))
        now = time.monotonic()
        with self._cache_lock:
            cached = self._run_detail_cache.get(cache_key)
            if cached is not None and (now - cached[0]) <= self.cache_ttl_seconds:
                return cached[1]

        run = self._find_run(experiment_id=experiment_id, phase=phase, run_name=run_name)
        summary = self._normalize_run_summary(run)
        result_pair = wandb_store.fetch_run_result(
            self.backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            attempt_token=summary.get("attempt_token"),
        )
        results_payload: Dict[str, Any] = {}
        job_result_payload: Dict[str, Any] = {}
        if result_pair is not None:
            results_payload, job_result_payload = result_pair

        config_preview = _coerce_mapping(getattr(run, "config", {}))
        results_payload = _maybe_enrich_wordle_eval_results(
            results_payload, config_preview
        )
        detail = {
            "summary": summary,
            "result_payload": _json_ready(results_payload),
            "job_result_payload": _json_ready(job_result_payload),
            "history": self._build_history_payload(run, phase=phase),
            "config": _json_ready(config_preview),
            "detail_kind": str(phase),
        }
        with self._cache_lock:
            self._run_detail_cache[cache_key] = (now, detail)
        return detail


def _parse_query_params(path: str) -> Dict[str, str]:
    parsed = urlparse(path)
    params = parse_qs(parsed.query)
    return {key: values[0] for key, values in params.items() if values}


def _static_asset_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "static" / name


def _build_request_handler(
    *,
    service: DashboardDataService,
    config: DashboardServerConfig,
):
    class DashboardRequestHandler(BaseHTTPRequestHandler):
        server_version = "TenysonDashboard/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _write_json(self, payload: Dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _write_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self._write_json({"error": "Not found."}, status=404)
                return
            mime_type, _encoding = mimetypes.guess_type(str(path))
            body = path.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type", mime_type or "application/octet-stream"
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = _parse_query_params(self.path)
            try:
                if parsed.path == "/":
                    self._write_file(_static_asset_path("index.html"))
                    return
                if parsed.path == "/assets/app.js":
                    self._write_file(_static_asset_path("app.js"))
                    return
                if parsed.path == "/assets/styles.css":
                    self._write_file(_static_asset_path("styles.css"))
                    return
                if parsed.path == "/api/config":
                    self._write_json(
                        {
                            "backend_ref": service.backend_ref,
                            "default_experiment_id": service.default_experiment_id,
                            "project_url": service.project_url(),
                            "refresh_seconds": config.refresh_seconds,
                        }
                    )
                    return
                if parsed.path == "/api/experiments":
                    self._write_json(
                        {
                            "experiments": service.list_experiments(),
                            "default_experiment_id": service.default_experiment_id,
                        }
                    )
                    return
                if parsed.path == "/api/experiment":
                    experiment_id = (
                        str(params.get("experiment_id") or "").strip()
                        or service.default_experiment_id
                    )
                    if not experiment_id:
                        self._write_json(
                            {"error": "experiment_id is required."},
                            status=400,
                        )
                        return
                    self._write_json(service.get_experiment_snapshot(experiment_id))
                    return
                if parsed.path == "/api/run":
                    experiment_id = (
                        str(params.get("experiment_id") or "").strip()
                        or service.default_experiment_id
                    )
                    phase = str(params.get("phase") or "").strip()
                    run_name = str(params.get("run_name") or "").strip()
                    if not experiment_id or not phase or not run_name:
                        self._write_json(
                            {
                                "error": "experiment_id, phase, and run_name are required."
                            },
                            status=400,
                        )
                        return
                    self._write_json(
                        service.get_run_detail(
                            experiment_id=experiment_id,
                            phase=phase,
                            run_name=run_name,
                        )
                    )
                    return
                self._write_json({"error": "Not found."}, status=404)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"error": str(exc)}, status=500)

    return DashboardRequestHandler


def serve_dashboard(config: DashboardServerConfig) -> None:
    service = DashboardDataService(
        backend_ref=config.backend_ref,
        default_experiment_id=config.experiment_id,
        cache_ttl_seconds=config.cache_ttl_seconds,
        history_limit=config.history_limit,
    )
    handler = _build_request_handler(service=service, config=config)
    server = ThreadingHTTPServer((config.host, int(config.port)), handler)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"[tenyson.ui] Serving telemetry dashboard at {url}", flush=True)
    if config.open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parse_args() -> DashboardServerConfig:
    parser = argparse.ArgumentParser(
        prog="python -m tenyson.ui",
        description="Serve a local telemetry dashboard for one W&B-backed Tenyson project.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Telemetry backend ref (wandb://<entity>/<project>).",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity fallback when --db-url is omitted.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project fallback when --db-url is omitted.",
    )
    parser.add_argument(
        "--experiment-id",
        default=os.getenv("TENYSON_EXPERIMENT_ID")
        or os.getenv("TENYSON_RECOVER_EXPERIMENT_ID"),
        help="Experiment id to open by default.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=10.0,
        help="Client polling interval for live updates.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=_DEFAULT_HISTORY_LIMIT,
        help="Maximum number of history rows to ship to the browser for one run.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the dashboard in the default browser after startup.",
    )
    args = parser.parse_args()
    backend_ref = _format_backend_ref(
        db_url=args.db_url,
        entity=args.entity,
        project=args.project,
    )
    return DashboardServerConfig(
        backend_ref=backend_ref,
        experiment_id=str(args.experiment_id or "").strip() or None,
        host=str(args.host),
        port=int(args.port),
        refresh_seconds=max(2.0, float(args.refresh_seconds)),
        open_browser=bool(args.open_browser),
        history_limit=max(10, int(args.history_limit)),
    )


def main() -> None:
    serve_dashboard(_parse_args())
