from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
from typing import Any, Callable, Optional, Sequence

from tenyson.bootstrap import (
    ensure_local_controller_environment,
    load_env_file,
    resolve_project_root,
)
from tenyson.cloud.modal import ModalManager
from tenyson.core.run_config import shared_overrides_from_env
from tenyson.experiment import ConfigTemplates, ExperimentSession
from tenyson.loader import load_task
from tenyson.reporting.fixed import ExperimentReport

DEFAULT_REPORT_ENV_VAR = "TENYSON_REPORT_PATH"
DEFAULT_FUNCTIONAL_FILENAME = "functional.py"
DEFAULT_RECOVERY_RESTART_STAGE_ENV_VAR = "TENYSON_RECOVER_RESTART_FROM_STAGE"


@dataclass(frozen=True)
class LocalExperimentContext:
    anchor_file: Path
    project_root: Path
    base_dir: Path
    env_path: Path
    loaded_env: dict[str, str]

    def file(self, relative_path: str | Path) -> Path:
        return self.base_dir / relative_path


def bootstrap_local_experiment(
    anchor_file: str | Path,
    *,
    env_filename: str = ".env",
) -> LocalExperimentContext:
    anchor = Path(anchor_file).resolve()
    ensure_local_controller_environment(anchor_file=anchor)
    project_root = resolve_project_root(anchor)
    base_dir = anchor.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    project_env_path = project_root / env_filename
    local_env_path = base_dir / env_filename
    env_path = project_env_path if project_env_path.is_file() else local_env_path
    loaded_env = load_env_file(env_path)
    return LocalExperimentContext(
        anchor_file=anchor,
        project_root=project_root,
        base_dir=base_dir,
        env_path=env_path,
        loaded_env=loaded_env,
    )


def install_sigterm_handler(
    *,
    label: str,
    on_signal: Optional[Callable[[int], None]] = None,
) -> None:
    def _handler(signum: int, _frame: object) -> None:
        if on_signal is not None:
            on_signal(signum)
        signal_name = signal.Signals(signum).name
        print(f"[{label}] Received {signal_name}; shutting down cleanly.", flush=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handler)


def env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    return int(raw or str(default))


def env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    return float(raw or str(default))


def resolve_recovery_experiment_id(
    env_var: str = "TENYSON_RECOVER_EXPERIMENT_ID",
) -> str | None:
    return str(os.getenv(env_var, "")).strip() or None


def resolve_on_failure_policy(
    *,
    env_var: str = "TENYSON_ON_FAILURE",
    default: str = "wait",
) -> str:
    return str(os.getenv(env_var, default)).strip() or default


def default_report_filename(anchor_file: str | Path) -> str:
    stem = _slug(Path(anchor_file).resolve().stem)
    if stem == "experiment":
        return "final_report.md"
    return f"final_report_{stem}.md"


def default_experiment_prefix(
    *,
    environment_name: str,
    anchor_file: str | Path,
) -> str:
    stem = _slug(Path(anchor_file).resolve().stem)
    env_name = _slug(environment_name) or "experiment"
    if stem == "experiment":
        return f"{env_name}_"
    return f"{env_name}_{stem}_"


def _slug(value: str | Path) -> str:
    text = str(value or "").strip().replace("-", "_")
    compact = "_".join(part for part in text.split() if part)
    return compact.lower() or "experiment"


def configure_experiment_identity(
    *,
    context: LocalExperimentContext,
    default_experiment_prefix: str | None = None,
    report_env_var: str,
    default_report_filename: str,
    experiment_id_env_var: str = "TENYSON_EXPERIMENT_ID",
    recovery_experiment_id_env_var: str = "TENYSON_RECOVER_EXPERIMENT_ID",
    regenerate_when_loaded_experiment_id: bool = False,
) -> str:
    recovery_experiment_id = str(
        os.getenv(recovery_experiment_id_env_var, "")
    ).strip()
    experiment_id = str(os.getenv(experiment_id_env_var, "")).strip()
    if recovery_experiment_id and (
        not experiment_id or experiment_id_env_var in context.loaded_env
    ):
        experiment_id = recovery_experiment_id
        os.environ[experiment_id_env_var] = experiment_id
    elif default_experiment_prefix and (
        not experiment_id
        or (
            regenerate_when_loaded_experiment_id
            and experiment_id_env_var in context.loaded_env
        )
    ):
        experiment_id = recovery_experiment_id or (
            default_experiment_prefix
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        os.environ[experiment_id_env_var] = experiment_id

    if not str(os.getenv(report_env_var, "")).strip():
        os.environ[report_env_var] = str(context.file(default_report_filename))

    return experiment_id


def resolve_report_output_path(
    *,
    context: LocalExperimentContext,
    report_env_var: str,
    default_filename: str,
) -> Path:
    configured = str(os.getenv(report_env_var, "")).strip()
    if configured:
        return Path(configured)
    return context.file(default_filename)


def build_experiment_report(
    *,
    context: LocalExperimentContext,
    report_env_var: str = DEFAULT_REPORT_ENV_VAR,
    default_filename: str | None = None,
) -> ExperimentReport:
    resolved_default_filename = str(
        default_filename or default_report_filename(context.anchor_file)
    ).strip()
    return ExperimentReport(
        output_path=resolve_report_output_path(
            context=context,
            report_env_var=report_env_var,
            default_filename=resolved_default_filename,
        )
    )


def load_local_task(
    *,
    context: LocalExperimentContext,
    task_filename: str = DEFAULT_FUNCTIONAL_FILENAME,
) -> Any:
    return load_task(str(context.file(task_filename)))


def resolve_recovery_restart_stage_name(
    *,
    env_var: str = DEFAULT_RECOVERY_RESTART_STAGE_ENV_VAR,
    fallback_env_vars: Sequence[str] = (),
) -> str | None:
    candidate_env_vars = [
        str(env_var).strip(),
        *[str(name).strip() for name in fallback_env_vars],
    ]
    for name in candidate_env_vars:
        if not name:
            continue
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return None


def resolve_recovery_restart_stages(
    *,
    ordered_stage_ids: Sequence[str],
    env_var: str = DEFAULT_RECOVERY_RESTART_STAGE_ENV_VAR,
    fallback_env_vars: Sequence[str] = (),
) -> list[str]:
    restart_from = resolve_recovery_restart_stage_name(
        env_var=env_var,
        fallback_env_vars=fallback_env_vars,
    )
    if not restart_from:
        return []

    normalized_stage_ids = [
        str(stage_id).strip()
        for stage_id in ordered_stage_ids
        if str(stage_id).strip()
    ]
    if restart_from not in normalized_stage_ids:
        raise ValueError(
            f"{env_var} must be one of {normalized_stage_ids}, got {restart_from!r}."
        )
    return normalized_stage_ids[normalized_stage_ids.index(restart_from) :]


def create_modal_experiment_session(
    *,
    context: LocalExperimentContext,
    task: Any,
    report: Optional[Any] = None,
    on_failure: Optional[str] = None,
    parallel: bool = True,
    recovery_experiment_id: Optional[str] = None,
    recovery_restart_stages: Optional[Sequence[str]] = None,
    report_metric_precision: Optional[int] = 4,
    report_wandb_text: str = "run",
    cloud_factory: Optional[Callable[[], Any]] = None,
) -> ExperimentSession:
    return ExperimentSession(
        task=task,
        templates=ConfigTemplates.from_directory(context.project_root / "config_templates"),
        cloud_factory=cloud_factory
        or ModalManager.factory_from_env(
            auto_terminate=True,
            gpu=str(os.getenv("TENYSON_MODAL_GPU", "A100")).strip() or "A100",
            timeout=env_int("TENYSON_MODAL_TIMEOUT", 86400),
        ),
        on_failure=on_failure or resolve_on_failure_policy(),
        shared_overrides=shared_overrides_from_env(),
        parallel=parallel,
        report=report,
        report_metric_precision=report_metric_precision,
        report_wandb_text=report_wandb_text,
        recovery_experiment_id=recovery_experiment_id,
        recovery_restart_stages=recovery_restart_stages,
    )
