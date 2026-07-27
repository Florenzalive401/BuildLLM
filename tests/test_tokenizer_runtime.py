from pathlib import Path

from src.tokenizer import BPETokenizer

TOKENIZER_PATH = Path("tokenizer") / "tokenizer.json"


def test_runtime_tokenizer_loads() -> None:
    tokenizer = BPETokenizer(TOKENIZER_PATH)
    assert tokenizer is not None


def test_runtime_tokenizer_matches_expected_vocabulary() -> None:
    tokenizer = BPETokenizer(TOKENIZER_PATH)
    assert tokenizer.vocabulary_size == 32_768


def test_runtime_tokenizer_can_encode() -> None:
    tokenizer = BPETokenizer(TOKENIZER_PATH)
    tokens = tokenizer.encode("Artificial intelligence is transforming software engineering.")
    assert len(tokens) > 0


def test_runtime_tokenizer_can_decode() -> None:
    tokenizer = BPETokenizer(TOKENIZER_PATH)
    text = "BuildLLM is training successfully."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0