import unittest

from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens


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


if __name__ == "__main__":
    unittest.main()
