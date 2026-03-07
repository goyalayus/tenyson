import time
from typing import Any, Dict, List, Sequence
from uuid import uuid4

from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.jobs.result import JobResult


class EvalJob:
    """
    Evaluation job using Unsloth + vLLM.

    This mirrors `src/runners/evals.py` but is driven by a `TaskPlugin`.
    """

    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config = config
        self.task = task
        self.run_id = self.config.get("evaluation", {}).get("run_name", "eval_job")

    def _build_model_and_tokenizer(self):
        from unsloth import FastLanguageModel

        model_cfg = self.config.get("model", {})
        vllm_cfg = self.config.get("vllm", {})

        model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
        seq_len = model_cfg.get("max_seq_length", 4096)

        print(f"[EvalJob] Loading model {model_name}...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            fast_inference=vllm_cfg.get("enabled", True),
            max_lora_rank=model_cfg.get("lora_r", 16),
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        init_repo = model_cfg.get("init_adapter_repo")
        if init_repo:
            print(f"[EvalJob] Loading LoRA adapter from {init_repo}...", flush=True)
            init_rev = model_cfg.get("init_adapter_revision", "main")
            model = FastLanguageModel.get_peft_model(
                model,
                r=model_cfg.get("lora_r", 16),
                target_modules=model_cfg.get(
                    "lora_target_modules", ["up_proj", "gate_proj", "down_proj"]
                ),
                lora_alpha=model_cfg.get("lora_alpha", 32),
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )

            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download

            adapter_path = hf_hub_download(
                repo_id=init_repo,
                filename="adapter_model.safetensors",
                revision=init_rev,
            )
            adapter_state = load_file(adapter_path)

            def _maybe_default(k: str) -> str:
                if ".lora_A.weight" in k:
                    return k.replace(".lora_A.weight", ".lora_A.default.weight")
                if ".lora_B.weight" in k:
                    return k.replace(".lora_B.weight", ".lora_B.default.weight")
                return k

            model_state = model.state_dict()
            mapped = {(_maybe_default(k)): v for k, v in adapter_state.items()}
            mapped_state = {k: v for k, v in mapped.items() if k in model_state}
            model.load_state_dict(mapped_state, strict=False)

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
            Generation,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            RunControl,
            TelemetryClient,
        )

        start = time.time()
        eval_cfg = self.config.get("evaluation", {})
        run_name = eval_cfg.get("run_name", self.run_id)

        db_url, experiment_id = resolve_required_telemetry_context(self.config)
        client = TelemetryClient(db_url=db_url)

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
        processed_dataset = dataset.select(range(len(processed_prompts)))
        results = self.task.compute_metrics(
            processed_prompts,
            processed_completions,
            processed_dataset,
            self.config,
            tokenizer,
        )

        if len(processed_prompts) < len(all_prompts):
            # Flag partial evaluations so downstream tooling can recognise them.
            results.setdefault("metadata", {})
            results["metadata"]["stopped_early"] = True

        metrics = results.get("metrics", {})
        for key, value in metrics.items():
            print(f"[EvalJob] {key}: {value}", flush=True)

        total_time = time.time() - start

        result = JobResult(
            run_id=run_name,
            status="success",
            total_time_seconds=total_time,
            metrics=metrics,
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
