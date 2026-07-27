from __future__ import annotations

import re
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COURSE_DIRECTORY = REPOSITORY_ROOT / "course"
LESSON_NAMES = [
    "00_GETTING_STARTED.md",
    "01_AI_AND_LLM_FOUNDATIONS.md",
    "02_DOWNLOAD_TRAINING_DATA.md",
    "03_BUILD_THE_CORPUS.md",
    "04_TOKENIZER_AND_DATASET.md",
    "05_TRANSFORMER_WALKTHROUGH.md",
    "06_TRAINING_AND_VALIDATION.md",
    "07_ITERATION_ONE.md",
    "08_GENERATION_AND_EVALUATION.md",
    "09_CPU_AND_GPU.md",
    "10_ITERATION_TWO.md",
    "11_CHECKPOINTS_AND_RECOVERY.md",
    "12_ITERATION_THREE.md",
    "13_COMPARE_THE_MODELS.md",
]
RUNNABLE_LESSONS = [
    "00_GETTING_STARTED.md",
    "02_DOWNLOAD_TRAINING_DATA.md",
    "03_BUILD_THE_CORPUS.md",
    "04_TOKENIZER_AND_DATASET.md",
    "05_TRANSFORMER_WALKTHROUGH.md",
    "06_TRAINING_AND_VALIDATION.md",
    "07_ITERATION_ONE.md",
    "08_GENERATION_AND_EVALUATION.md",
    "09_CPU_AND_GPU.md",
    "10_ITERATION_TWO.md",
    "11_CHECKPOINTS_AND_RECOVERY.md",
    "12_ITERATION_THREE.md",
]


def lesson_text(name: str) -> str:
    return (COURSE_DIRECTORY / name).read_text(encoding="utf-8")


def test_readme_workflow_matches_learning_configuration() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "corpus_learning_50m.yaml").read_text(
            encoding="utf-8"
        )
    )

    required_in_order = [
        "git clone",
        "py -m venv .venv",
        "build_wikipedia_corpus.py",
        "build_training_corpus.py",
        "train_tokenizer.py",
        "encode_corpus.py",
        "inspect_model.py",
        "run_lab.py",
        "generate_text.py",
    ]
    positions = [readme.index(text) for text in required_in_order]
    assert positions == sorted(positions)
    assert config["output"]["corpus"] in readme
    assert config["output"]["report"] in readme
    assert "tokenizer/learning_50m_tokenizer.json" in readme
    assert "data/tokens/learning_50m/train_tokens.pt" in readme
    assert "data/tokens/learning_50m/validation_tokens.pt" in readme
    assert "--checkpoint-directory checkpoints/iteration_1_learning_50m" in readme


def test_course_artifact_chain_is_taught_in_dependency_order() -> None:
    download = lesson_text("02_DOWNLOAD_TRAINING_DATA.md")
    corpus = lesson_text("03_BUILD_THE_CORPUS.md")
    tokenizer = lesson_text("04_TOKENIZER_AND_DATASET.md")
    iteration_one = lesson_text("07_ITERATION_ONE.md")
    generation = lesson_text("08_GENERATION_AND_EVALUATION.md")

    artifacts = {
        "wikipedia": "data/processed/wikipedia_simple.jsonl",
        "corpus": "data/processed/training_corpus_learning_50m.jsonl",
        "tokenizer": "tokenizer/learning_50m_tokenizer.json",
        "training": "data/tokens/learning_50m/train_tokens.pt",
        "validation": "data/tokens/learning_50m/validation_tokens.pt",
        "checkpoint": "checkpoints/iteration_1_learning_50m/best_checkpoint.pt",
    }

    assert artifacts["wikipedia"] in download and artifacts["wikipedia"] in corpus
    assert artifacts["corpus"] in corpus and artifacts["corpus"] in tokenizer
    assert artifacts["tokenizer"] in tokenizer and artifacts["tokenizer"] in iteration_one
    assert artifacts["training"] in tokenizer and artifacts["training"] in iteration_one
    assert artifacts["validation"] in tokenizer and artifacts["validation"] in iteration_one
    assert artifacts["checkpoint"] in generation


def test_corpus_lesson_documents_every_available_configuration() -> None:
    corpus_lesson = lesson_text("03_BUILD_THE_CORPUS.md")
    configurations = [
        "configs/corpus_pipeline_verification.yaml",
        "configs/corpus_learning_50m.yaml",
        "configs/corpus_balanced.yaml",
        "configs/corpus_800m.yaml",
    ]
    for configuration in configurations:
        assert configuration in corpus_lesson

    assert "recommended first complete build" in corpus_lesson
    assert "enabled source targets total 750 million characters" in corpus_lesson
    assert "data/processed/training_corpus_balanced.jsonl" in corpus_lesson
    assert "data/processed/training_corpus_800m.jsonl" in corpus_lesson


def test_runnable_lessons_have_visible_learning_and_success_gates() -> None:
    required_sections = [
        "## What you will learn",
        "## Where you are in the build",
        "## Before you begin",
        "## Files this lesson will create",
        "## What success looks like",
        "## Stop and check",
        "## Common problems and exact responses",
        "## Under the hood",
        "## Check your understanding",
        "## Next lesson",
    ]

    for lesson_name in RUNNABLE_LESSONS:
        text = lesson_text(lesson_name)
        for section in required_sections:
            assert section in text, f"{lesson_name} is missing {section}"
        positions = [text.index(section) for section in required_sections]
        assert positions == sorted(positions), f"{lesson_name} section order changed"


def test_controlled_and_larger_data_paths_are_kept_separate() -> None:
    home = lesson_text("README.md")
    iteration_two = lesson_text("10_ITERATION_TWO.md")
    iteration_three = lesson_text("12_ITERATION_THREE.md")
    comparison = lesson_text("13_COMPARE_THE_MODELS.md")

    for text in (home, iteration_two, iteration_three, comparison):
        assert "controlled comparison" in text.lower()
        assert "larger-data" in text.lower()

    assert "20,000 training examples" in iteration_two
    assert "1,000 validation examples" in iteration_two
    assert "--training-examples 20000" in iteration_two
    assert "--validation-examples 1000" in iteration_two
    assert "--training-examples 20000" in iteration_three
    assert "--validation-examples 1000" in iteration_three
    assert "Do not rank validation loss or perplexity" in comparison


def test_larger_data_artifact_chains_are_complete() -> None:
    iteration_two = lesson_text("10_ITERATION_TWO.md")
    iteration_three = lesson_text("12_ITERATION_THREE.md")

    balanced_artifacts = [
        "configs/corpus_balanced.yaml",
        "data/processed/training_corpus_balanced.jsonl",
        "data/reports/training_corpus_balanced_report.json",
        "tokenizer/balanced_tokenizer.json",
        "data/tokens/balanced/train_tokens.pt",
        "data/tokens/balanced/validation_tokens.pt",
        "data/tokens/balanced/encoding_report.json",
        "checkpoints/iteration_2_balanced",
        "--iteration 2",
        "--epochs 3",
        "generate_text.py",
    ]
    corpus_800m_artifacts = [
        "configs/corpus_800m.yaml",
        "data/processed/training_corpus_800m.jsonl",
        "data/reports/training_corpus_800m_report.json",
        "tokenizer/800m_tokenizer.json",
        "data/tokens/800m/train_tokens.pt",
        "data/tokens/800m/validation_tokens.pt",
        "data/tokens/800m/encoding_report.json",
        "checkpoints/iteration_3_800m",
        "--iteration 3",
        "--epochs 18",
        "generate_text.py",
        "prepare_800m_corpus.py",
    ]

    for artifact in balanced_artifacts:
        assert artifact in iteration_two
    for artifact in corpus_800m_artifacts:
        assert artifact in iteration_three


def test_documented_run_lab_commands_include_experiment_identity() -> None:
    for lesson_name in ("07_ITERATION_ONE.md", "10_ITERATION_TWO.md", "12_ITERATION_THREE.md"):
        text = lesson_text(lesson_name)
        for flag in (
            "--iteration",
            "--device",
            "--epochs",
            "--training-examples",
            "--validation-examples",
            "--train-tokens",
            "--validation-tokens",
            "--tokenizer",
            "--checkpoint-directory",
            "--dry-run",
        ):
            assert flag in text, f"{lesson_name} is missing {flag}"


def test_application_output_fields_are_explained() -> None:
    training = lesson_text("06_TRAINING_AND_VALIDATION.md")
    iteration_one = lesson_text("07_ITERATION_ONE.md")
    recovery = lesson_text("11_CHECKPOINTS_AND_RECOVERY.md")
    generation = lesson_text("08_GENERATION_AND_EVALUATION.md")

    for field in (
        "maximum_training_steps",
        "tensor_cores_enabled",
        "tf32_enabled",
        "fused_adamw",
        "verified_checkpoint",
        "restored_checkpoint_type",
    ):
        assert field in training
    assert "Epoch <n>: training loss <value>, validation loss <value>, steps <value>" in training
    assert "Checkpoint saved: <path>" in iteration_one
    assert "restored_global_step" in recovery
    assert "Finish reason:" in generation
    assert "Generated tokens:" in generation


def test_support_pages_are_linked_and_cover_required_fields() -> None:
    home = lesson_text("README.md")
    glossary = lesson_text("GLOSSARY.md")
    worksheet = lesson_text("RUN_RECORD_WORKSHEET.md")
    troubleshooting = lesson_text("TROUBLESHOOTING.md")

    for name in ("GLOSSARY.md", "RUN_RECORD_WORKSHEET.md", "TROUBLESHOOTING.md"):
        assert name in home

    for term in ("Byte-level BPE", "Optimizer step", "BF16", "Atomic save", "Finish reason"):
        assert term in glossary
    for field in ("Corpus YAML", "Training tokens", "Global step", "Fixed generation protocol", "Conclusion"):
        assert field in worksheet
    for stage in ("Wikipedia download", "Corpus construction", "CUDA training", "Checkpoints", "Generation"):
        assert f"## {stage}" in troubleshooting


def test_course_uses_purpose_based_names_and_current_language() -> None:
    course_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in COURSE_DIRECTORY.glob("*.md")
    ).lower()

    assert "smo" "ke" not in course_text
    assert "week" "end" not in course_text
    assert "capability-training" not in course_text
    assert "capability training" not in course_text
    assert "training_corpus_sm" "oke" not in course_text
    assert "training_corpus_week" "end" not in course_text


def test_course_markdown_is_ready_for_wordpress_paste() -> None:
    structural_line = re.compile(
        r"^\s*(?:"
        r"#{1,6}\s+|"
        r"[-+*]\s+|"
        r"\d+[.)]\s+|"
        r"\||>|"
        r"\[[^\]]+\]:|"
        r"(?:[-*_]\s*){3,}"
        r")"
    )
    fence = re.compile(r"^\s*(`{3,}|~{3,})")

    for lesson_path in COURSE_DIRECTORY.glob("*.md"):
        lines = lesson_path.read_text(encoding="utf-8").splitlines()
        in_fence = False

        def is_plain_prose(line: str) -> bool:
            return bool(line.strip()) and not (
                structural_line.match(line) or line.startswith(("    ", "\t"))
            )

        for line_number, line in enumerate(lines, start=1):
            if fence.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            assert not (
                re.match(r"^ {2,}\S", line)
                and not re.match(r"^\s+(?:[-+*]|\d+[.)])\s+", line)
            ), f"{lesson_path.name}:{line_number} has a wrapped continuation"

            if line_number < len(lines):
                following_line = lines[line_number]
                assert not (
                    is_plain_prose(line) and is_plain_prose(following_line)
                ), (
                    f"{lesson_path.name}:{line_number} has a hard-wrapped "
                    "prose paragraph"
                )
