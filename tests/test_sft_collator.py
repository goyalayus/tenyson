import unittest

from tenyson.jobs.sft_collator import CompletionOnlyDataCollator


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "<U>": 1,
            "<A>": 2,
            "hello": 3,
            "world": 4,
            "again": 5,
        }
        self.pad_token_id = 0
        self.eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[token] for token in text.split()]

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=False,
    ):
        input_ids = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }


class CompletionOnlyCollatorTests(unittest.TestCase):
    def test_masks_prompt_tokens_before_assistant_response(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(tokenizer, response_template="<A>")

        batch = collator([{"text": "<U> hello <A> world"}])

        self.assertEqual(batch["input_ids"].tolist(), [[1, 3, 2, 4]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 4]])

    def test_multiple_assistant_turns_require_instruction_template(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(tokenizer, response_template="<A>")

        with self.assertRaisesRegex(ValueError, "Multiple assistant turns"):
            collator([
                {"text": "<U> hello <A> world <U> again <A> world"},
            ])

    def test_instruction_template_limits_assistant_spans_between_user_turns(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(
            tokenizer,
            response_template="<A>",
            instruction_template="<U>",
        )

        batch = collator([
            {"text": "<U> hello <A> world <U> again <A> world"},
        ])

        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, 4, -100, -100, -100, 4]],
        )

    def test_packed_style_multi_example_sequence_has_no_boundary_attention_mask(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(
            tokenizer,
            response_template="<A>",
            instruction_template="<U>",
        )

        # This simulates two short prompt/response examples concatenated into one
        # packed sequence. The collator can still mask labels correctly, but it
        # only emits a flat token-vs-padding attention mask, not per-example
        # sequence boundaries.
        batch = collator([
            {"text": "<U> hello <A> world <U> again <A> world"},
        ])

        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, 4, -100, -100, -100, 4]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[1, 1, 1, 1, 1, 1, 1, 1]],
        )


if __name__ == "__main__":
    unittest.main()
