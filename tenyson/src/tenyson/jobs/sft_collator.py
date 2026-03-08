"""
Data collator for SFT that masks non-assistant tokens so loss is computed only on
the model response. Used when loss_on_assistant_only is enabled and no task-provided
collator is used. Compatible with TRL SFTTrainer when used with a formatting function.
"""

from typing import Any, Dict, List, Optional

import torch


class CompletionOnlyDataCollator:
    """
    Sets labels to -100 for all tokens before the assistant response, so loss is
    computed only on the completion. Requires response_template to mark where
    the assistant reply starts in the tokenized sequence.
    """

    def __init__(
        self,
        tokenizer: Any,
        response_template: str,
        instruction_template: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.instruction_template = instruction_template
        self.max_length = max_length
        self._response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        if not self._response_template_ids:
            raise ValueError(
                "response_template tokenizes to empty; use a non-empty string that "
                "appears in the formatted sequence where the assistant reply starts."
            )
        self._instruction_template_ids = None
        if instruction_template:
            instruction_ids = tokenizer.encode(
                instruction_template, add_special_tokens=False
            )
            if not instruction_ids:
                raise ValueError(
                    "instruction_template tokenizes to empty; use a non-empty string "
                    "that appears where a non-assistant turn starts."
                )
            self._instruction_template_ids = instruction_ids

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Examples may have "text" (from formatting_func) or "input_ids" (pre-tokenized).
        if "input_ids" in examples[0]:
            input_ids = [ex["input_ids"] for ex in examples]
            if isinstance(input_ids[0], torch.Tensor):
                input_ids = [ids.tolist() for ids in input_ids]
            attention_mask = [
                ex.get("attention_mask", [1] * len(ids))
                for ex, ids in zip(examples, input_ids)
            ]
        else:
            texts = []
            for ex in examples:
                t = ex["text"]
                if isinstance(t, list):
                    t = t[0] if t else ""
                texts.append(t)
            tokenized = self.tokenizer(
                texts,
                padding=False,
                truncation=self.max_length is not None,
                max_length=self.max_length,
                return_tensors=None,
                add_special_tokens=False,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get(
                "attention_mask", [[1] * len(ids) for ids in input_ids]
            )

        # Find assistant reply spans in each sequence and build labels.
        labels_list = []
        for ids, mask in zip(input_ids, attention_mask):
            ids = ids if isinstance(ids, list) else ids.tolist()
            mask = mask if isinstance(mask, list) else mask.tolist()
            assistant_spans = self._find_response_spans(ids)
            label = [-100] * len(ids)
            for start, end in assistant_spans:
                for i in range(start, min(end, len(ids))):
                    label[i] = ids[i]
            for i, m in enumerate(mask):
                if i < len(label) and m == 0:
                    label[i] = -100
            labels_list.append(label)

        # Pad to max length in batch
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for ids, mask, label in zip(input_ids, attention_mask, labels_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(label + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def _find_template_occurrences(
        self, input_ids: List[int], template: Optional[List[int]]
    ) -> List[int]:
        if not template:
            return []
        occurrences = []
        for i in range(len(input_ids) - len(template) + 1):
            if input_ids[i : i + len(template)] == template:
                occurrences.append(i)
        return occurrences

    def _find_response_spans(self, input_ids: List[int]) -> List[tuple[int, int]]:
        """
        Return assistant-content spans as `(start, end)` token offsets.

        With a single assistant turn, the span runs from the response template to the
        end of the sequence. For multi-turn chat examples, `instruction_template` is
        required so we can stop masking before the next non-assistant turn instead of
        silently training on later user text.
        """
        response_starts = self._find_template_occurrences(
            input_ids, self._response_template_ids
        )
        if not response_starts:
            raise ValueError(
                "response_template was not found in a formatted SFT example. "
                "Check the chat template or disable loss_on_assistant_only for this dataset."
            )

        if len(response_starts) > 1 and self._instruction_template_ids is None:
            raise ValueError(
                "Multiple assistant turns were found in one SFT example, but "
                "instruction_template is not set. Provide training.instruction_template "
                "or use a task-specific collator so assistant-only masking can exclude "
                "intervening user turns safely."
            )

        instruction_starts = self._find_template_occurrences(
            input_ids, self._instruction_template_ids
        )

        spans: List[tuple[int, int]] = []
        for index, response_start in enumerate(response_starts):
            start = response_start + len(self._response_template_ids)
            end_candidates: List[int] = []

            next_response_start = (
                response_starts[index + 1] if index + 1 < len(response_starts) else None
            )
            if next_response_start is not None:
                end_candidates.append(next_response_start)

            if instruction_starts:
                next_instruction_start = next(
                    (
                        instruction_start
                        for instruction_start in instruction_starts
                        if instruction_start > start
                    ),
                    None,
                )
                if next_instruction_start is not None:
                    end_candidates.append(next_instruction_start)

            end = min(end_candidates) if end_candidates else len(input_ids)
            spans.append((start, end))

        return spans
