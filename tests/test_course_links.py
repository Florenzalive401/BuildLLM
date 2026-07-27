from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXPECTED_COURSE_ORDER = [
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


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPOSITORY_ROOT.rglob("*.md")
        if ".pytest_cache" not in path.parts
    )


def local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if (
        not target
        or target.startswith(("#", "http://", "https://", "mailto:"))
    ):
        return None

    path_text = target.split("#", maxsplit=1)[0]
    if not path_text:
        return None

    return (source.parent / path_text).resolve()


def test_markdown_local_links_resolve() -> None:
    broken_links: list[str] = []

    for source in markdown_files():
        contents = source.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(contents):
            target = local_link_target(source, raw_target)
            if target is not None and not target.exists():
                source_name = source.relative_to(REPOSITORY_ROOT)
                broken_links.append(f"{source_name} -> {raw_target}")

    assert not broken_links, "Broken local Markdown links:\n" + "\n".join(
        broken_links
    )


def test_course_map_links_every_lesson() -> None:
    course_map = (REPOSITORY_ROOT / "course" / "README.md").read_text(
        encoding="utf-8"
    )

    unlisted = [
        path.name
        for path in (REPOSITORY_ROOT / "course").glob("*.md")
        if path.name not in {"README.md", "REFERENCE_IMPLEMENTATION.md"}
        and path.name not in course_map
    ]

    assert not unlisted, "Course lessons missing from course/README.md: " + ", ".join(
        sorted(unlisted)
    )


def test_course_files_and_map_follow_the_required_order() -> None:
    course_directory = REPOSITORY_ROOT / "course"
    lesson_names = [
        path.name
        for path in sorted(course_directory.glob("[0-9][0-9]_*.md"))
    ]
    assert lesson_names == EXPECTED_COURSE_ORDER

    course_map = (course_directory / "README.md").read_text(encoding="utf-8")
    positions = [course_map.index(name) for name in EXPECTED_COURSE_ORDER]
    assert positions == sorted(positions)
