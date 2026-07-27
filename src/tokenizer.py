from __future__ import annotations

from pathlib import Path
from typing import Iterable



class BPETokenizer:
    """Runtime wrapper for the trained byte level BPE tokenizer."""

    PADDING_TOKEN = "<|padding|>"
    UNKNOWN_TOKEN = "<|unknown|>"
    DOCUMENT_END_TOKEN = "<|document_end|>"

    def __init__(
        self,
        tokenizer_file: str | Path = "tokenizer/tokenizer.json",
    ) -> None:
        self.tokenizer_file = Path(tokenizer_file)

        if not self.tokenizer_file.exists():
            raise FileNotFoundError(
                f"tokenizer file does not exist: {self.tokenizer_file}"
            )

        if not self.tokenizer_file.is_file():
            raise ValueError("tokenizer_file must reference a file")

        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "the tokenizers package is required; run pip install -r requirements.txt"
            ) from error

        self._tokenizer = Tokenizer.from_file(
            str(self.tokenizer_file)
        )
        self._validate_special_tokens()

    def encode(
        self,
        text: str,
        *,
        add_document_end: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        token_ids = list(self._tokenizer.encode(text).ids)

        if add_document_end:
            token_ids.append(self.document_end_token_id)

        return token_ids

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        add_document_end: bool = False,
    ) -> list[list[int]]:
        text_list = list(texts)

        if any(not isinstance(text, str) for text in text_list):
            raise TypeError("texts must contain only strings")

        encoded = [
            list(result.ids)
            for result in self._tokenizer.encode_batch(text_list)
        ]

        if add_document_end:
            document_end_id = self.document_end_token_id
            for token_ids in encoded:
                token_ids.append(document_end_id)

        return encoded

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        resolved_ids = [int(token_id) for token_id in token_ids]
        return self._tokenizer.decode(
            resolved_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def token_to_id(
        self,
        token: str,
    ) -> int:
        if not isinstance(token, str):
            raise TypeError("token must be a string")

        token_id = self._tokenizer.token_to_id(token)

        if token_id is None:
            raise KeyError(f"token is not in the vocabulary: {token}")

        return int(token_id)

    @property
    def vocabulary_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    @property
    def padding_token_id(self) -> int:
        return self.token_to_id(self.PADDING_TOKEN)

    @property
    def unknown_token_id(self) -> int:
        return self.token_to_id(self.UNKNOWN_TOKEN)

    @property
    def document_end_token_id(self) -> int:
        return self.token_to_id(self.DOCUMENT_END_TOKEN)

    def _validate_special_tokens(self) -> None:
        for token in (
            self.PADDING_TOKEN,
            self.UNKNOWN_TOKEN,
            self.DOCUMENT_END_TOKEN,
        ):
            if self._tokenizer.token_to_id(token) is None:
                raise ValueError(
                    f"trained tokenizer is missing required token: {token}"
                )


Tokenizer = BPETokenizer
