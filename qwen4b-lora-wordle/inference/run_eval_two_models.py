#!/usr/bin/env python3
"""
Run fixed-prompt evaluation for two Wordle LoRA adapters and build an HTML report.

Outputs:
  - inference/results_qwen3_4b.json
  - inference/results_qwen3_06b.json
  - inference/eval_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_SPECS = [
    {
        "key": "qwen3_4b",
        "label": "Qwen3-4B LoRA",
        "base_model": "Qwen/Qwen3-4B",
        "adapter_repo": "goyalayus/wordle-lora-qwen3-4b",
    },
    {
        "key": "qwen3_06b",
        "label": "Qwen3-0.6B LoRA",
        "base_model": "Qwen/Qwen3-0.6B",
        "adapter_repo": "goyalayus/wordle-full-qwen06b",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate two LoRA models on fixed prompts.")
    p.add_argument(
        "--prompts-json",
        default="inference/wordle_eval_prompts_10.json",
        help="Path to prompt cases JSON",
    )
    p.add_argument(
        "--output-dir",
        default="inference",
        help="Directory to write JSON outputs and report HTML",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Max generated tokens per prompt",
    )
    p.add_argument(
        "--revision-strategy",
        choices=["latest_step", "final", "main"],
        default="latest_step",
        help="How to pick adapter revision from HF repo refs",
    )
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prompt_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No prompt cases found in {path}")
    for idx, case in enumerate(cases):
        if "case_id" not in case or "messages" not in case:
            raise ValueError(f"Case at index {idx} is missing case_id/messages")
    return cases


def _extract_step_num(name: str) -> int:
    m = re.fullmatch(r"step-(\d+)", name.strip())
    return int(m.group(1)) if m else -1


def resolve_revision(repo_id: str, strategy: str) -> Tuple[str, str]:
    api = HfApi()
    refs = api.list_repo_refs(repo_id=repo_id)
    all_names: List[str] = []
    for branch in getattr(refs, "branches", []):
        all_names.append(branch.name)
    for tag in getattr(refs, "tags", []):
        all_names.append(tag.name)

    if strategy == "main":
        return "main", "forced_main"

    if strategy == "final":
        if "final" in all_names:
            return "final", "forced_final"
        return "main", "fallback_main_no_final"

    # latest_step strategy
    step_candidates = [(name, _extract_step_num(name)) for name in all_names]
    step_candidates = [x for x in step_candidates if x[1] >= 0]
    if step_candidates:
        best = max(step_candidates, key=lambda x: x[1])[0]
        return best, "latest_step"
    if "final" in all_names:
        return "final", "fallback_final_no_step"
    return "main", "fallback_main_no_step_final"


def build_model_and_tokenizer(
    base_model: str, adapter_repo: str, revision: str
) -> Tuple[Any, Any, str]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_repo, revision=revision)
    model.eval()
    device = str(next(model.parameters()).device)
    return model, tokenizer, device


def parse_guess(text: str) -> str | None:
    m = re.search(r"<guess>\[([a-zA-Z]{5})\]</guess>", text)
    return m.group(1).lower() if m else None


@torch.inference_mode()
def generate_one(
    model: Any, tokenizer: Any, messages: List[Dict[str, str]], max_new_tokens: int
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=False).strip()


def run_eval_for_model(
    spec: Dict[str, str],
    cases: List[Dict[str, Any]],
    max_new_tokens: int,
    revision_strategy: str,
) -> Dict[str, Any]:
    revision, revision_source = resolve_revision(spec["adapter_repo"], revision_strategy)
    model, tokenizer, device = build_model_and_tokenizer(
        spec["base_model"], spec["adapter_repo"], revision
    )

    results = []
    for case in cases:
        output = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=case["messages"],
            max_new_tokens=max_new_tokens,
        )
        results.append(
            {
                "case_id": case["case_id"],
                "messages": case["messages"],
                "response_text": output,
                "parsed_guess": parse_guess(output),
            }
        )

    return {
        "model_key": spec["key"],
        "model_label": spec["label"],
        "base_model": spec["base_model"],
        "adapter_repo": spec["adapter_repo"],
        "adapter_revision_used": revision,
        "adapter_revision_source": revision_source,
        "device": device,
        "generated_at_utc": utc_now(),
        "num_cases": len(results),
        "results": results,
    }


def render_html_report(
    out_path: Path, model_a: Dict[str, Any], model_b: Dict[str, Any], cases: List[Dict[str, Any]]
) -> None:
    a_map = {r["case_id"]: r for r in model_a["results"]}
    b_map = {r["case_id"]: r for r in model_b["results"]}

    rows = []
    for case in cases:
        case_id = case["case_id"]
        user_msg = next((m["content"] for m in case["messages"] if m["role"] == "user"), "")
        a_resp = a_map.get(case_id, {}).get("response_text", "")
        b_resp = b_map.get(case_id, {}).get("response_text", "")
        a_guess = a_map.get(case_id, {}).get("parsed_guess", "")
        b_guess = b_map.get(case_id, {}).get("parsed_guess", "")
        rows.append(
            f"""
            <tr>
              <td><code>{html.escape(case_id)}</code></td>
              <td><pre>{html.escape(user_msg)}</pre></td>
              <td>
                <div><b>guess:</b> <code>{html.escape(str(a_guess))}</code></div>
                <pre>{html.escape(a_resp)}</pre>
              </td>
              <td>
                <div><b>guess:</b> <code>{html.escape(str(b_guess))}</code></div>
                <pre>{html.escape(b_resp)}</pre>
              </td>
            </tr>
            """
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wordle model eval report</title>
  <style>
    body {{ font-family: Inter, system-ui, Arial, sans-serif; margin: 20px; background: #0b1020; color: #e8ecf1; }}
    h1 {{ margin: 0 0 12px 0; }}
    .meta {{ margin-bottom: 16px; color: #a8b3c2; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #24314a; padding: 10px; vertical-align: top; }}
    th {{ background: #121a2e; position: sticky; top: 0; }}
    td {{ background: #0f172a; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 8px 0 0 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.35; }}
    code {{ background: #17223b; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Wordle eval comparison</h1>
  <div class="meta">
    <div><b>Model A:</b> {html.escape(model_a["model_label"])} | repo={html.escape(model_a["adapter_repo"])} | rev={html.escape(model_a["adapter_revision_used"])}</div>
    <div><b>Model B:</b> {html.escape(model_b["model_label"])} | repo={html.escape(model_b["adapter_repo"])} | rev={html.escape(model_b["adapter_revision_used"])}</div>
    <div><b>Generated:</b> {html.escape(utc_now())}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th style="width: 11%;">Case</th>
        <th style="width: 33%;">User prompt</th>
        <th style="width: 28%;">{html.escape(model_a["model_label"])}</th>
        <th style="width: 28%;">{html.escape(model_b["model_label"])}</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""
    out_path.write_text(doc)


def main() -> None:
    args = parse_args()
    prompts_path = Path(args.prompts_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_prompt_cases(prompts_path)
    print(f"Loaded {len(cases)} prompt cases from {prompts_path}")

    all_results = []
    for spec in MODEL_SPECS:
        print(f"\n=== Evaluating {spec['label']} ({spec['adapter_repo']}) ===")
        model_result = run_eval_for_model(
            spec=spec,
            cases=cases,
            max_new_tokens=args.max_new_tokens,
            revision_strategy=args.revision_strategy,
        )
        out_json = output_dir / f"results_{spec['key']}.json"
        out_json.write_text(json.dumps(model_result, indent=2))
        print(f"Saved {out_json}")
        all_results.append(model_result)

    report_path = output_dir / "eval_report.html"
    render_html_report(
        out_path=report_path,
        model_a=all_results[0],
        model_b=all_results[1],
        cases=cases,
    )
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
