from argparse import Namespace
from pathlib import Path

import pytest

import prepare_800m_corpus


def test_800m_preparation_defaults_use_purpose_based_paths() -> None:
    assert prepare_800m_corpus.DEFAULT_CONFIG == Path("configs/corpus_800m.yaml")
    assert prepare_800m_corpus.DEFAULT_CORPUS == Path(
        "data/processed/training_corpus_800m.jsonl"
    )
    assert prepare_800m_corpus.DEFAULT_TOKENIZER == Path(
        "tokenizer/800m_tokenizer.json"
    )
    assert prepare_800m_corpus.DEFAULT_TOKEN_DIRECTORY == Path(
        "data/tokens/800m"
    )
    assert prepare_800m_corpus.DEFAULT_REPORT == Path(
        "data/reports/training_corpus_800m_report.json"
    )


def test_800m_preparation_orders_build_tokenizer_and_encoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "corpus_800m.yaml"
    corpus = tmp_path / "training_corpus_800m.jsonl"
    tokenizer = tmp_path / "800m_tokenizer.json"
    token_directory = tmp_path / "tokens"
    config.write_text("version: 1\n", encoding="utf-8")

    args = Namespace(
        config=config,
        corpus=corpus,
        report=tmp_path / "training_corpus_800m_report.json",
        tokenizer=tokenizer,
        token_directory=token_directory,
        vocabulary_size=32_768,
        validation_fraction=0.05,
        skip_build=False,
        skip_tokenizer=False,
        skip_encode=False,
    )
    commands: list[list[str]] = []

    def record(command: list[str]) -> None:
        commands.append(command)
        if command[1] == "build_training_corpus.py":
            corpus.write_text('{"text": "example"}\n', encoding="utf-8")
        elif command[1] == "train_tokenizer.py":
            tokenizer.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(prepare_800m_corpus, "parse_args", lambda: args)
    monkeypatch.setattr(prepare_800m_corpus, "run", record)
    monkeypatch.setattr(prepare_800m_corpus, "print_summary", lambda *args: None)

    prepare_800m_corpus.main()

    assert [command[1] for command in commands] == [
        "build_training_corpus.py",
        "train_tokenizer.py",
        "encode_corpus.py",
    ]


def test_800m_summary_prints_run_lab_command(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "training_corpus_800m.jsonl"
    report = tmp_path / "training_corpus_800m_report.json"
    tokenizer = tmp_path / "800m_tokenizer.json"
    token_directory = tmp_path / "tokens"
    corpus.write_text('{"text": "example"}\n', encoding="utf-8")
    tokenizer.write_text("{}", encoding="utf-8")

    prepare_800m_corpus.print_summary(
        corpus,
        report,
        tokenizer,
        token_directory,
    )

    output = capsys.readouterr().out
    assert "python run_lab.py" in output
    assert "--iteration 3" in output
    assert "--epochs 18" in output
    assert f"--tokenizer {tokenizer}" in output
    assert "checkpoints/iteration_3_800m" in output
    assert "train_small_gpt.py" not in output


def test_800m_preparation_can_reuse_an_existing_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "training_corpus_800m.jsonl"
    tokenizer = tmp_path / "800m_tokenizer.json"
    corpus.write_text('{"text": "example"}\n', encoding="utf-8")
    tokenizer.write_text("{}", encoding="utf-8")

    args = Namespace(
        config=tmp_path / "corpus_800m.yaml",
        corpus=corpus,
        report=tmp_path / "training_corpus_800m_report.json",
        tokenizer=tokenizer,
        token_directory=tmp_path / "tokens",
        vocabulary_size=32_768,
        validation_fraction=0.05,
        skip_build=True,
        skip_tokenizer=True,
        skip_encode=False,
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(prepare_800m_corpus, "parse_args", lambda: args)
    monkeypatch.setattr(
        prepare_800m_corpus,
        "run",
        lambda command: commands.append(command),
    )
    monkeypatch.setattr(prepare_800m_corpus, "print_summary", lambda *args: None)

    prepare_800m_corpus.main()

    assert [command[1] for command in commands] == ["encode_corpus.py"]
