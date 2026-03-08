import time
from typing import Any, Dict, List, Sequence
from uuid import uuid4

from tenyson.core.hf_adapter import (
    download_hf_lora_adapter,
    resolve_hf_lora_runtime_kwargs,
    strict_load_hf_lora_adapter_weights,
)
from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.result import JobResult


class EvalJob:
    """
    Evaluation job using Unsloth + vLLM.

    This mirrors `src/runners/evals.py` but is driven by a `TaskPlugin`.
    """

    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config = config
        self.task = task
        self.run_id = resolve_required_run_name(self.config, "eval")

    def _build_model_and_tokenizer(self):
        from unsloth import FastLanguageModel

        model_cfg = self.config.get("model", {})
        vllm_cfg = self.config.get("vllm", {})

        init_repo = str(model_cfg.get("init_adapter_repo") or "").strip()
        init_adapter = None
        lora_runtime_kwargs: Dict[str, Any] | None = None
        if init_repo:
            init_rev = model_cfg.get("init_adapter_revision", "main")
            init_adapter = download_hf_lora_adapter(
                repo_id=init_repo, revision=init_rev
            )
            lora_runtime_kwargs = resolve_hf_lora_runtime_kwargs(
                init_adapter,
                expected_r=model_cfg.get("lora_r") if "lora_r" in model_cfg else None,
                expected_alpha=model_cfg.get("lora_alpha")
                if "lora_alpha" in model_cfg
                else None,
                expected_target_modules=model_cfg.get("lora_target_modules")
                if "lora_target_modules" in model_cfg
                else None,
            )
            print(
                "[EvalJob] Using init adapter "
                f"{init_repo}@{init_adapter.resolved_revision} "
                f"({init_adapter.weights_in_repo}).",
                flush=True,
            )

        model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
        seq_len = model_cfg.get("max_seq_length", 4096)
        max_lora_rank = (
            int(lora_runtime_kwargs["r"])
            if lora_runtime_kwargs is not None
            else int(model_cfg.get("lora_r", 16))
        )

        print(f"[EvalJob] Loading model {model_name}...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            fast_inference=vllm_cfg.get("enabled", True),
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if init_adapter is not None and lora_runtime_kwargs is not None:
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(lora_runtime_kwargs["r"]),
                target_modules=lora_runtime_kwargs["target_modules"],
                lora_alpha=lora_runtime_kwargs["lora_alpha"],
                lora_dropout=lora_runtime_kwargs["lora_dropout"],
                bias=lora_runtime_kwargs["bias"],
                use_gradient_checkpointing="unsloth",
            )
            loaded_tensors = strict_load_hf_lora_adapter_weights(model, init_adapter)
            print(
                "[EvalJob] Successfully loaded init adapter "
                f"from {init_repo}@{init_adapter.resolved_revision} "
                f"({loaded_tensors} tensors).",
                flush=True,
            )

        print("[EvalJob] Switching model to inference mode...", flush=True)
        model.train(False)
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _build_sampling_params(self, tokenizer: Any):
        from vllm import SamplingParams

        vllm_cfg = self.config.get("vllm", {})
        return SamplingParams(
            temperature=float(vllm_cfg.get("temperature", 0.0)),
            top_p=float(vllm_cfg.get("top_p", 1.0)),
            max_tokens=int(vllm_cfg.get("max_tokens", 1024)),
            stop=[tokenizer.eos_token]
            if getattr(tokenizer, "eos_token", None) is not None
            else None,
        )

    def _extract_prompts(self, dataset: Any) -> List[str]:
        return [row["prompt"] for row in dataset]

    def run(self) -> JobResult:
        require_gpu_provider_runtime()
        from tenyson.core.telemetry import (
            begin_run_attempt,
            Generation,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            RunControl,
            TelemetryClient,
        )

        start = time.time()
        eval_cfg = self.config.get("evaluation", {})
        run_name = self.run_id

        db_url, experiment_id = resolve_required_telemetry_context(self.config)
        client = TelemetryClient(db_url=db_url)
        if begin_run_attempt(client, experiment_id, run_name, phase="eval"):
            print(
                "[EvalJob] Cleared stale manual stop request from a previous attempt.",
                flush=True,
            )

        model, tokenizer = self._build_model_and_tokenizer()

        print("[EvalJob] Loading eval dataset via TaskPlugin...", flush=True)
        dataset = self.task.get_eval_dataset(self.config)
        print(f"[EvalJob] Loaded {len(dataset)} examples for evaluation.", flush=True)

        sampling_params = self._build_sampling_params(tokenizer)
        all_prompts: Sequence[str] = self._extract_prompts(dataset)

        batch_size = int(self.config.get("evaluation", {}).get("batch_size", 32))
        batch_size = max(1, batch_size)

        processed_prompts: List[str] = []
        processed_completions: List[str] = []

        def _should_stop() -> bool:
            session = client.Session()
            try:
                control_row = (
                    session.query(RunControl)
                    .filter(RunControl.run_id == run_name)
                    .filter(RunControl.experiment_id == experiment_id)
                    .order_by(RunControl.updated_at.desc())
                    .first()
                )
                return bool(control_row and control_row.stop_requested)
            finally:
                session.close()

        print(
            f"[EvalJob] Starting batched generation with vLLM (temp={sampling_params.temperature})...",
            flush=True,
        )
        for start_idx in range(0, len(all_prompts), batch_size):
            end_idx = min(start_idx + batch_size, len(all_prompts))
            batch_prompts = list(all_prompts[start_idx:end_idx])

            outputs = model.fast_generate(
                batch_prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
            )

            for out in outputs:
                processed_completions.append(out.outputs[0].text)
            processed_prompts.extend(batch_prompts)

            if _should_stop():
                print(
                    f"[EvalJob] Manual stop requested after processing {len(processed_prompts)} prompts; stopping early.",
                    flush=True,
                )
                break

        print("[EvalJob] Generation complete. Computing metrics...", flush=True)

        # Telemetry: log eval prompts and completions into shared Generation table.
        session = client.Session()
        try:
            for idx, (prompt, completion) in enumerate(
                zip(processed_prompts, processed_completions)
            ):
                generation = Generation(
                    id=str(uuid4()),
                    experiment_id=experiment_id,
                    run_id=run_name,
                    global_step=idx,
                    phase="eval",
                    prompt_text=str(prompt),
                    completion_text=str(completion),
                    reward=None,
                )
                session.add(generation)
            session.commit()
        finally:
            session.close()

        # Only pass the processed subset through to metrics.
        processed_samples = len(processed_prompts)
        expected_samples = len(all_prompts)
        processed_dataset = dataset.select(range(processed_samples))
        results = self.task.compute_metrics(
            processed_prompts,
            processed_completions,
            processed_dataset,
            self.config,
            tokenizer,
        )
        stopped_early = processed_samples < expected_samples
        results.setdefault("metadata", {})
        results["metadata"]["processed_samples"] = processed_samples
        results["metadata"]["expected_samples"] = expected_samples
        if stopped_early:
            # Flag partial evaluations so downstream tooling can recognise them.
            results["metadata"]["stopped_early"] = True

        metrics = results.get("metrics", {})
        for key, value in metrics.items():
            print(f"[EvalJob] {key}: {value}", flush=True)

        total_time = time.time() - start

        result = JobResult(
            run_id=run_name,
            status="partial" if stopped_early else "success",
            total_time_seconds=total_time,
            metrics=metrics,
            stopped_early=stopped_early,
            processed_samples=processed_samples,
            expected_samples=expected_samples,
        )
        record_run_summary(
            client=client,
            experiment_id=experiment_id,
            phase="eval",
            result=result,
        )
        record_run_result(
            client=client,
            experiment_id=experiment_id,
            run_id=run_name,
            phase="eval",
            results_payload=results,
            job_result_payload=result,
        )
        return result
