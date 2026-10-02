from __future__ import annotations

from transformers.tokenization_python import PreTrainedTokenizer


class Pflm1Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return 256

    def get_vocab(self) -> dict[str, int]:
        return {chr(i): i for i in range(256)}

    def _tokenize(self, text: str) -> list[str]:
        return [chr(b) for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str) -> int:
        return ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        return chr(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return bytes(ord(t) for t in tokens).decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple:
        return ()


__all__ = ["Pflm1Tokenizer"]
