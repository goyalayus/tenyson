import copy
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tenyson.cloud.aws import AWSManager
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.result import JobResult
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_task
from tenyson.pipeline import run_pipeline
from tenyson.reporting.builder import ReportBuilder


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cloud_kwargs() -> Dict[str, Any]:
    return {
        "instance_type": os.getenv("TENYSON_AWS_INSTANCE_TYPE", "g5.2xlarge"),
        "region": os.getenv("TENYSON_AWS_REGION", "us-east-1"),
        "key_name": os.getenv("TENYSON_AWS_KEY_NAME"),
        "key_path": os.getenv("TENYSON_AWS_KEY_PATH"),
        "security_group": os.getenv("TENYSON_AWS_SECURITY_GROUP"),
        "subnet": os.getenv("TENYSON_AWS_SUBNET") or None,
        "profile": os.getenv("AWS_PROFILE") or None,
        "auto_terminate": True,
        "use_spot": True,
        "spot_max_price": os.getenv("TENYSON_AWS_SPOT_MAX_PRICE") or None,
    }


def _build_cloud(kwargs: Dict[str, Any]) -> AWSManager:
    return AWSManager(**kwargs)


def _run_single_step(step: Tuple[str, dict, type, Any], cloud: AWSManager) -> JobResult:
    label, config, _job_class, _task = step
    results = run_pipeline([step], cloud, on_failure="wait")
    run_id = config.get("training", {}).get("run_name") or config.get(
        "evaluation", {}
    ).get("run_name")
    if run_id:
        for result in reversed(results):
            if result.run_id == run_id:
                return result
    if not results:
        raise RuntimeError(f"Step '{label}' did not return a result.")
    return results[-1]


def _run_parallel_eval_steps(
    stage_label: str,
    eval_steps: List[Tuple[str, dict, type, Any]],
    cloud: AWSManager,
) -> Dict[str, JobResult]:
    results = run_pipeline(
        [{"label": stage_label, "parallel": eval_steps}],
        cloud,
        on_failure="wait",
    )
    output: Dict[str, JobResult] = {}
    for label, config, _job_class, _task in eval_steps:
        run_id = config.get("evaluation", {}).get("run_name")
        matched = None
        for result in reversed(results):
            if run_id and result.run_id == run_id:
                matched = result
                break
        if matched is None and results:
            matched = results[-1]
        if matched is None:
            raise RuntimeError(f"Parallel eval step '{label}' did not return a result.")
        output[label] = matched
    return output


def _require_adapter(result: JobResult, stage_name: str) -> Tuple[str, str]:
    repo = getattr(result, "hf_repo_id", None)
    rev = getattr(result, "hf_revision", None) or "main"
    if not repo:
        raise RuntimeError(
            f"{stage_name} finished without hf_repo_id. "
            "Set training.hf_repo_base to a valid Hugging Face namespace/repo prefix."
        )
    return repo, rev


def _set_adapter(config: Dict[str, Any], adapter_repo: str, adapter_revision: str) -> None:
    model_cfg = config.setdefault("model", {})
    model_cfg["init_adapter_repo"] = adapter_repo
    model_cfg["init_adapter_revision"] = adapter_revision


def _prepare_rl_cfg(
    base_cfg: Dict[str, Any],
    run_name: str,
    output_dir: str,
    min_turn: int,
    max_turn: int,
    adapter_repo: str,
    adapter_revision: str,
    hf_repo_base_override: Optional[str],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    train_cfg = cfg.setdefault("training", {})
    train_cfg["run_name"] = run_name
    train_cfg["output_dir"] = output_dir
    if hf_repo_base_override:
        train_cfg["hf_repo_base"] = hf_repo_base_override
    task_cfg = cfg.setdefault("task", {})
    task_cfg["min_history_turns"] = min_turn
    task_cfg["max_history_turns"] = max_turn
    _set_adapter(cfg, adapter_repo, adapter_revision)
    return cfg


def _prepare_eval_cfg(
    base_cfg: Dict[str, Any],
    run_name: str,
    output_dir: str,
    adapter_repo: str,
    adapter_revision: str,
    min_turn: int,
    max_turn: int,
    exact_turns: Optional[List[int]] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg["run_name"] = run_name
    eval_cfg["output_dir"] = output_dir
    task_cfg = cfg.setdefault("task", {})
    task_cfg["min_history_turns"] = min_turn
    task_cfg["max_history_turns"] = max_turn
    if exact_turns is None:
        task_cfg.pop("eval_exact_turns", None)
    else:
        task_cfg["eval_exact_turns"] = exact_turns
    _set_adapter(cfg, adapter_repo, adapter_revision)
    return cfg


def _format_metric(result: Optional[JobResult], key: str) -> str:
    if result is None:
        return "n/a"
    value = result.metrics.get(key)
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _wandb_link(result: Optional[JobResult]) -> str:
    if result is None or not result.wandb_url:
        return "n/a"
    return f"[run]({result.wandb_url})"


def _run_mixed_branch(
    task: Any,
    cloud_kwargs: Dict[str, Any],
    rl_base_cfg: Dict[str, Any],
    eval_base_cfg: Dict[str, Any],
    adapter_repo: str,
    adapter_revision: str,
    hf_repo_base_override: Optional[str],
) -> Dict[str, JobResult]:
    cloud = _build_cloud(cloud_kwargs)
    results: Dict[str, JobResult] = {}

    rl_cfg = _prepare_rl_cfg(
        base_cfg=rl_base_cfg,
        run_name="wordle_rl_mixed",
        output_dir="./outputs/wordle_research/mixed/rl",
        min_turn=1,
        max_turn=5,
        adapter_repo=adapter_repo,
        adapter_revision=adapter_revision,
        hf_repo_base_override=hf_repo_base_override,
    )
    rl_step = ("mixed_rl", rl_cfg, RLJob, task)
    results["mixed_rl"] = _run_single_step(rl_step, cloud)
    mixed_repo, mixed_rev = _require_adapter(results["mixed_rl"], "mixed_rl")

    final_eval_cfg = _prepare_eval_cfg(
        base_cfg=eval_base_cfg,
        run_name="wordle_mixed_final_eval",
        output_dir="./outputs/wordle_research/mixed/eval_final_mixed",
        adapter_repo=mixed_repo,
        adapter_revision=mixed_rev,
        min_turn=1,
        max_turn=5,
        exact_turns=None,
    )
    eval_step = ("mixed_final_eval", final_eval_cfg, EvalJob, task)
    results["mixed_final_eval"] = _run_single_step(eval_step, cloud)
    return results


def _curriculum_eval_turns(stage_turn: int) -> List[int]:
    if stage_turn == 2:
        return [2]
    if stage_turn == 3:
        return [2, 3]
    if stage_turn == 4:
        return [3, 4]
    if stage_turn == 5:
        return [4, 5]
    raise ValueError(f"Unsupported curriculum stage turn: {stage_turn}")


def _run_curriculum_branch(
    task: Any,
    cloud_kwargs: Dict[str, Any],
    rl_base_cfg: Dict[str, Any],
    eval_base_cfg: Dict[str, Any],
    adapter_repo: str,
    adapter_revision: str,
    hf_repo_base_override: Optional[str],
) -> Dict[str, JobResult]:
    cloud = _build_cloud(cloud_kwargs)
    results: Dict[str, JobResult] = {}
    current_repo = adapter_repo
    current_rev = adapter_revision

    for stage_turn in [2, 3, 4, 5]:
        rl_key = f"curr_rl_t{stage_turn}"
        rl_cfg = _prepare_rl_cfg(
            base_cfg=rl_base_cfg,
            run_name=f"wordle_curriculum_rl_t{stage_turn}",
            output_dir=f"./outputs/wordle_research/curriculum/rl_t{stage_turn}",
            min_turn=stage_turn,
            max_turn=stage_turn,
            adapter_repo=current_repo,
            adapter_revision=current_rev,
            hf_repo_base_override=hf_repo_base_override,
        )
        rl_step = (rl_key, rl_cfg, RLJob, task)
        results[rl_key] = _run_single_step(rl_step, cloud)
        current_repo, current_rev = _require_adapter(results[rl_key], rl_key)

        eval_turns = _curriculum_eval_turns(stage_turn)
        eval_steps: List[Tuple[str, dict, type, Any]] = []
        for turn in eval_turns:
            eval_key = f"curr_eval_after_t{stage_turn}_turn{turn}"
            eval_cfg = _prepare_eval_cfg(
                base_cfg=eval_base_cfg,
                run_name=f"wordle_curr_eval_after_t{stage_turn}_turn{turn}",
                output_dir=(
                    f"./outputs/wordle_research/curriculum/eval_after_t{stage_turn}_turn{turn}"
                ),
                adapter_repo=current_repo,
                adapter_revision=current_rev,
                min_turn=turn,
                max_turn=turn,
                exact_turns=[turn],
            )
            eval_steps.append((eval_key, eval_cfg, EvalJob, task))

        if len(eval_steps) == 1:
            label, cfg, job_class, step_task = eval_steps[0]
            results[label] = _run_single_step((label, cfg, job_class, step_task), cloud)
        else:
            parallel_results = _run_parallel_eval_steps(
                stage_label=f"curr_eval_after_t{stage_turn}",
                eval_steps=eval_steps,
                cloud=cloud,
            )
            results.update(parallel_results)

    final_eval_cfg = _prepare_eval_cfg(
        base_cfg=eval_base_cfg,
        run_name="wordle_curriculum_final_eval",
        output_dir="./outputs/wordle_research/curriculum/eval_final_mixed",
        adapter_repo=current_repo,
        adapter_revision=current_rev,
        min_turn=1,
        max_turn=5,
        exact_turns=None,
    )
    final_eval_step = ("curr_final_eval", final_eval_cfg, EvalJob, task)
    results["curr_final_eval"] = _run_single_step(final_eval_step, cloud)
    return results


def _compute_delta(a: Optional[JobResult], b: Optional[JobResult], key: str) -> str:
    if a is None or b is None:
        return "n/a"
    va = a.metrics.get(key)
    vb = b.metrics.get(key)
    if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
        return "n/a"
    return f"{float(va) - float(vb):.4f}"


def _build_report_data(
    sft_result: JobResult,
    baseline_eval_result: JobResult,
    mixed_results: Dict[str, JobResult],
    curriculum_results: Dict[str, JobResult],
) -> Dict[str, str]:
    data: Dict[str, str] = {
        "sft_status": sft_result.status,
        "sft_wandb_link": _wandb_link(sft_result),
        "baseline_eval_status": baseline_eval_result.status,
        "baseline_eval_constraint_accuracy": _format_metric(
            baseline_eval_result, "constraint_accuracy"
        ),
        "baseline_eval_dict_accuracy": _format_metric(
            baseline_eval_result, "dict_accuracy"
        ),
        "baseline_eval_format_accuracy": _format_metric(
            baseline_eval_result, "format_accuracy"
        ),
    }

    mixed_rl = mixed_results.get("mixed_rl")
    mixed_final_eval = mixed_results.get("mixed_final_eval")
    data.update(
        {
            "mixed_rl_status": mixed_rl.status if mixed_rl else "n/a",
            "mixed_rl_wandb_link": _wandb_link(mixed_rl),
            "mixed_final_eval_status": mixed_final_eval.status if mixed_final_eval else "n/a",
            "mixed_final_constraint_accuracy": _format_metric(
                mixed_final_eval, "constraint_accuracy"
            ),
            "mixed_final_dict_accuracy": _format_metric(mixed_final_eval, "dict_accuracy"),
            "mixed_final_format_accuracy": _format_metric(
                mixed_final_eval, "format_accuracy"
            ),
        }
    )

    for turn in [2, 3, 4, 5]:
        rl_key = f"curr_rl_t{turn}"
        rl_result = curriculum_results.get(rl_key)
        data[f"{rl_key}_status"] = rl_result.status if rl_result else "n/a"
        data[f"{rl_key}_wandb_link"] = _wandb_link(rl_result)

    curriculum_eval_keys = [
        "curr_eval_after_t2_turn2",
        "curr_eval_after_t3_turn2",
        "curr_eval_after_t3_turn3",
        "curr_eval_after_t4_turn3",
        "curr_eval_after_t4_turn4",
        "curr_eval_after_t5_turn4",
        "curr_eval_after_t5_turn5",
    ]
    for key in curriculum_eval_keys:
        eval_result = curriculum_results.get(key)
        data[f"{key}_status"] = eval_result.status if eval_result else "n/a"
        data[f"{key}_constraint_accuracy"] = _format_metric(
            eval_result, "constraint_accuracy"
        )
        data[f"{key}_dict_accuracy"] = _format_metric(eval_result, "dict_accuracy")
        data[f"{key}_format_accuracy"] = _format_metric(eval_result, "format_accuracy")

    curr_final_eval = curriculum_results.get("curr_final_eval")
    data.update(
        {
            "curr_final_eval_status": curr_final_eval.status if curr_final_eval else "n/a",
            "curr_final_constraint_accuracy": _format_metric(
                curr_final_eval, "constraint_accuracy"
            ),
            "curr_final_dict_accuracy": _format_metric(curr_final_eval, "dict_accuracy"),
            "curr_final_format_accuracy": _format_metric(
                curr_final_eval, "format_accuracy"
            ),
            "delta_final_constraint_accuracy": _compute_delta(
                mixed_final_eval, curr_final_eval, "constraint_accuracy"
            ),
            "delta_final_dict_accuracy": _compute_delta(
                mixed_final_eval, curr_final_eval, "dict_accuracy"
            ),
            "delta_final_format_accuracy": _compute_delta(
                mixed_final_eval, curr_final_eval, "format_accuracy"
            ),
        }
    )

    return data


def main() -> None:
    base_dir = Path(__file__).parent
    task = load_task(str(Path(__file__).with_name("wordle_task.py")))

    cloud_kwargs = _cloud_kwargs()
    required_aws = {
        "TENYSON_AWS_KEY_NAME": cloud_kwargs["key_name"],
        "TENYSON_AWS_KEY_PATH": cloud_kwargs["key_path"],
        "TENYSON_AWS_SECURITY_GROUP": cloud_kwargs["security_group"],
    }
    missing = [k for k, v in required_aws.items() if not v]
    if missing:
        raise ValueError(
            "Missing required AWS environment variables for this experiment: "
            + ", ".join(missing)
        )
    primary_cloud = _build_cloud(cloud_kwargs)

    sft_base_cfg = load_yaml(str(base_dir / "configs" / "sft_config.yaml"))
    rl_base_cfg = load_yaml(str(base_dir / "configs" / "rl_config.yaml"))
    eval_base_cfg = load_yaml(str(base_dir / "configs" / "eval_config.yaml"))

    hf_repo_base_override = os.getenv("TENYSON_HF_REPO_BASE")

    sft_cfg = copy.deepcopy(sft_base_cfg)
    sft_cfg.setdefault("training", {})["run_name"] = "wordle_sft_main"
    sft_cfg.setdefault("training", {})["output_dir"] = "./outputs/wordle_research/sft"
    if hf_repo_base_override:
        sft_cfg.setdefault("training", {})["hf_repo_base"] = hf_repo_base_override

    sft_step = ("sft_main", sft_cfg, SFTJob, task)
    sft_result = _run_single_step(sft_step, primary_cloud)
    sft_adapter_repo, sft_adapter_rev = _require_adapter(sft_result, "sft_main")

    baseline_eval_cfg = _prepare_eval_cfg(
        base_cfg=eval_base_cfg,
        run_name="wordle_eval_baseline_mixed",
        output_dir="./outputs/wordle_research/eval_baseline_mixed",
        adapter_repo=sft_adapter_repo,
        adapter_revision=sft_adapter_rev,
        min_turn=1,
        max_turn=5,
        exact_turns=None,
    )
    baseline_eval_step = ("eval_baseline_mixed", baseline_eval_cfg, EvalJob, task)
    baseline_eval_result = _run_single_step(baseline_eval_step, primary_cloud)

    with ThreadPoolExecutor(max_workers=2) as executor:
        mixed_future = executor.submit(
            _run_mixed_branch,
            task,
            cloud_kwargs,
            rl_base_cfg,
            eval_base_cfg,
            sft_adapter_repo,
            sft_adapter_rev,
            hf_repo_base_override,
        )
        curriculum_future = executor.submit(
            _run_curriculum_branch,
            task,
            cloud_kwargs,
            rl_base_cfg,
            eval_base_cfg,
            sft_adapter_repo,
            sft_adapter_rev,
            hf_repo_base_override,
        )
        mixed_results = mixed_future.result()
        curriculum_results = curriculum_future.result()

    report_data = _build_report_data(
        sft_result=sft_result,
        baseline_eval_result=baseline_eval_result,
        mixed_results=mixed_results,
        curriculum_results=curriculum_results,
    )
    report = ReportBuilder(
        template_path=str(base_dir / "report_template.md"),
        output_path=str(base_dir / "final_report.md"),
    )
    report.fill(report_data)
    report.generate()


if __name__ == "__main__":
    main()
