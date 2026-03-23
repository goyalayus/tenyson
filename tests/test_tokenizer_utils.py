import unittest

from tenyson.jobs.tokenizer_utils import (
    ensure_assistant_mask_chat_template,
    normalize_tokenizer_special_tokens,
)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<eos>": 1,
            "<pad>": 2,
            "<|im_end|>": 3,
        }
        self.unk_token_id = 999
        self.unk_token = "<unk>"
        self.eos_token = "<broken>"
        self.eos_token_id = 1
        self.pad_token = None
        self.padding_side = "right"
        self.chat_template = None

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        reverse_vocab = {value: key for key, value in self.vocab.items()}
        return reverse_vocab.get(token_id)


class BrokenTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.eos_token = None
        self.eos_token_id = None

    def convert_tokens_to_ids(self, token):
        return None

    def convert_ids_to_tokens(self, token_id):
        return None


class TokenizerUtilsTests(unittest.TestCase):
    def test_normalize_tokenizer_special_tokens_repairs_eos_and_pad(self) -> None:
        tokenizer = FakeTokenizer()

        normalized = normalize_tokenizer_special_tokens(
            tokenizer,
            padding_side="left",
        )

        self.assertIs(normalized, tokenizer)
        self.assertEqual(tokenizer.eos_token, "<eos>")
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.padding_side, "left")

    def test_normalize_tokenizer_special_tokens_raises_when_no_valid_eos_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not resolve a valid EOS token"):
            normalize_tokenizer_special_tokens(BrokenTokenizer())

    def test_ensure_assistant_mask_chat_template_installs_fallback_when_missing(self) -> None:
        tokenizer = FakeTokenizer()

        ensure_assistant_mask_chat_template(tokenizer)

        self.assertIn("{% generation %}", tokenizer.chat_template)
        self.assertIn("{% endgeneration %}", tokenizer.chat_template)
        self.assertIn("<|im_start|>assistant", tokenizer.chat_template)

    def test_ensure_assistant_mask_chat_template_preserves_existing_generation_blocks(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = (
            "{% for message in messages %}{% generation %}x{% endgeneration %}{% endfor %}"
        )

        ensure_assistant_mask_chat_template(tokenizer)

        self.assertEqual(
            tokenizer.chat_template,
            "{% for message in messages %}{% generation %}x{% endgeneration %}{% endfor %}",
        )


if __name__ == "__main__":
    unittest.main()
