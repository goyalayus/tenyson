import json
import os
import sys
from pathlib import Path

import modal

APP_NAME = "wordle-eval-turn345-mixed"
REPO_ROOT = str(Path(__file__).resolve().parents[1])
WORKDIR = "/workspace/wordle-prime-rl-reproduction/qwen4b-lora-wordle"
SECRETS_NAME = "wordle-train-secrets"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/workspace/wordle-prime-rl-reproduction",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(SECRETS_NAME)],
)
def run_eval(
    adapter_repo: str = "goyalayus/wordle-qwen3-4b-rl-mixed-from-rl980",
    adapter_revision: str = "main",
    num_samples: int = 1,
    max_new_tokens: int = 180,
    prompts_json: str = "inference/wordle_eval_prompts_10.json",
    max_cases_per_turn: int = 0,
):
    os.chdir(WORKDIR)
    if WORKDIR not in sys.path:
        sys.path.insert(0, WORKDIR)
    inference_dir = f"{WORKDIR}/inference"
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    from inference import eval_passk_4b_turn2 as ep

    all_cases = ep.load_prompt_cases(Path(prompts_json))
    model, tokenizer, _ = ep.build_model_and_tokenizer(
        base_model="Qwen/Qwen3-4B",
        adapter_repo=adapter_repo,
        revision=adapter_revision,
    )

    results = []
    for turn in (3, 4, 5):
        cases = ep.filter_cases(all_cases, require_single_history=False, require_turn=turn)
        if max_cases_per_turn and max_cases_per_turn > 0:
            cases = cases[:max_cases_per_turn]
        print(f"[turn {turn}] evaluating {len(cases)} cases", flush=True)
        if not cases:
            results.append(
                {
                    "turn": turn,
                    "evaluated_cases": 0,
                    "binary_hit_rate_any_success": 0.0,
                    "mean_pass_at_1_estimator": 0.0,
                    "adapter_repo": adapter_repo,
                    "adapter_revision_used": adapter_revision,
                    "prompts_json": prompts_json,
                }
            )
            continue

        case_hits = []
        case_pass = []
        for case_idx, case in enumerate(cases):
            print(f"[turn {turn}] case {case_idx + 1}/{len(cases)}", flush=True)
            user = ep.get_user_message(case)
            history = ep.parse_history(user)
            constraints = ep.build_constraints(history)
            success_count = 0
            for sample_i in range(num_samples):
                _ = 3407 + turn * 100_000 + case_idx * 10_000 + sample_i
                prompt = tokenizer.apply_chat_template(
                    case["messages"], tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated = outputs[0][inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(generated, skip_special_tokens=False).strip()
                guess = ep.parse_guess(response)
                check = ep.check_guess_constraints(guess, constraints)
                if check.is_success:
                    success_count += 1

            case_hits.append(1 if success_count > 0 else 0)
            case_pass.append(ep.pass_at_k(num_samples, success_count, 1))

        results.append(
            {
                "turn": turn,
                "evaluated_cases": len(cases),
                "binary_hit_rate_any_success": sum(case_hits) / len(case_hits),
                "mean_pass_at_1_estimator": sum(case_pass) / len(case_pass),
                "adapter_repo": adapter_repo,
                "adapter_revision_used": adapter_revision,
                "prompts_json": prompts_json,
            }
        )

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    adapter_repo: str = "goyalayus/wordle-qwen3-4b-rl-mixed-from-rl980",
    adapter_revision: str = "main",
    num_samples: int = 1,
    max_new_tokens: int = 180,
    prompts_json: str = "inference/wordle_eval_prompts_10.json",
    max_cases_per_turn: int = 0,
):
    result_json = run_eval.remote(
        adapter_repo=adapter_repo,
        adapter_revision=adapter_revision,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        prompts_json=prompts_json,
        max_cases_per_turn=max_cases_per_turn,
    )
    print(result_json)
