from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from src.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def tokenizer_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    vocabulary = {
        BPETokenizer.PADDING_TOKEN: 0,
        BPETokenizer.UNKNOWN_TOKEN: 1,
        BPETokenizer.DOCUMENT_END_TOKEN: 2,
    }
    vocabulary.update(
        {
            f"token_{token_id}": token_id
            for token_id in range(3, 32_768)
        }
    )

    tokenizer = Tokenizer(
        WordLevel(
            vocab=vocabulary,
            unk_token=BPETokenizer.UNKNOWN_TOKEN,
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def test_runtime_tokenizer_loads(
    tokenizer_path: Path,
) -> None:
    tokenizer = BPETokenizer(tokenizer_path)
    assert tokenizer is not None


def test_runtime_tokenizer_matches_expected_vocabulary(
    tokenizer_path: Path,
) -> None:
    tokenizer = BPETokenizer(tokenizer_path)
    assert tokenizer.vocabulary_size == 32_768


def test_runtime_tokenizer_can_encode(
    tokenizer_path: Path,
) -> None:
    tokenizer = BPETokenizer(tokenizer_path)
    tokens = tokenizer.encode("Artificial intelligence is transforming software engineering.")
    assert len(tokens) > 0


def test_runtime_tokenizer_can_decode(
    tokenizer_path: Path,
) -> None:
    tokenizer = BPETokenizer(tokenizer_path)
    text = "BuildLLM is training successfully."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
