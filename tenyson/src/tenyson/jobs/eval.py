import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from tenyson.core.plugin import TaskPlugin
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
        from tenyson.core.telemetry import Generation, TelemetryClient

        start = time.time()
        eval_cfg = self.config.get("evaluation", {})

        output_dir = Path(eval_cfg.get("output_dir", "./outputs/evals"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility.
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        model, tokenizer = self._build_model_and_tokenizer()

        print("[EvalJob] Loading eval dataset via TaskPlugin...", flush=True)
        dataset = self.task.get_eval_dataset(self.config)
        print(f"[EvalJob] Loaded {len(dataset)} examples for evaluation.", flush=True)

        sampling_params = self._build_sampling_params(tokenizer)
        prompts = self._extract_prompts(dataset)

        print(
            f"[EvalJob] Starting batched generation with vLLM (temp={sampling_params.temperature})...",
            flush=True,
        )
        outputs = model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        completions: List[str] = []
        for out in outputs:
            # Assuming single generation per prompt for eval.
            completions.append(out.outputs[0].text)

        print("[EvalJob] Generation complete. Computing metrics...", flush=True)

        # Optional telemetry: log eval prompts and completions into the
        # shared Generation table when a telemetry DB is configured.
        telemetry_cfg = self.config.get("telemetry", {})
        db_url = telemetry_cfg.get("db_url")
        if db_url:
            client = TelemetryClient(db_url=db_url)
            session = client.Session()
            try:
                for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
                    generation = Generation(
                        id=str(uuid4()),
                        run_id=self.run_id,
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
        results = self.task.compute_metrics(
            prompts, completions, dataset, self.config, tokenizer
        )

        out_file = output_dir / "results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        metrics = results.get("metrics", {})
        for key, value in metrics.items():
            print(f"[EvalJob] {key}: {value}", flush=True)

        total_time = time.time() - start

        result = JobResult(
            run_id=self.run_id,
            status="success",
            total_time_seconds=total_time,
            metrics=metrics,
            local_output_dir=str(output_dir),
        )
        # Also persist JobResult alongside detailed results.
        result.save(str(output_dir / "job_result.json"))
        return result
