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
            "<NL>": 98,
            "<E>": 99,
        }
        self.inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token = "<E>"
        self.eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        normalized = text.replace("\n", " <NL> ")
        return [self.vocab[token] for token in normalized.split()]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        parts = []
        for token_id in token_ids:
            token = self.inverse_vocab[token_id]
            if token == "<NL>":
                parts.append("\n")
                continue
            if not parts or parts[-1].endswith("\n"):
                parts.append(token)
            else:
                parts.append(f" {token}")
        return "".join(parts)

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

    def test_excludes_trailing_eos_token_from_assistant_labels(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(tokenizer, response_template="<A>")

        batch = collator([{"text": "<U> hello <A> world <E>"}])

        self.assertEqual(batch["input_ids"].tolist(), [[1, 3, 2, 4, 99]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 4, -100]])

    def test_excludes_eos_followed_by_trailing_newline(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(tokenizer, response_template="<A>")

        batch = collator([{"text": "<U> hello <A> world <E>\n"}])

        self.assertEqual(batch["input_ids"].tolist(), [[1, 3, 2, 4, 99, 98]])
        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, 4, -100, -100]],
        )

    def test_preserves_content_newline_before_response_terminator(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(tokenizer, response_template="<A>")

        batch = collator([{"text": "<U> hello <A> world\n<E>\n"}])

        self.assertEqual(batch["input_ids"].tolist(), [[1, 3, 2, 4, 98, 99, 98]])
        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, 4, 98, -100, -100]],
        )

    def test_instruction_template_limits_assistant_spans_between_user_turns(self) -> None:
        tokenizer = SimpleTokenizer()
        collator = CompletionOnlyDataCollator(
            tokenizer,
            response_template="<A>",
            instruction_template="<U>",
        )

        batch = collator([
            {"text": "<U> hello <A> world <E> <U> again <A> world <E>"},
        ])

        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, 4, -100, -100, -100, -100, 4, -100]],
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
            {"text": "<U> hello <A> world <E> <U> again <A> world <E>"},
        ])

        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, -100, 4, -100, -100, -100, -100, 4, -100]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        )


if __name__ == "__main__":
    unittest.main()
