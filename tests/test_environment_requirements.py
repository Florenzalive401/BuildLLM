from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def active_requirements() -> list[Requirement]:
    requirements: list[Requirement] = []
    for line in (REPOSITORY_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirement = Requirement(stripped)
        if requirement.marker is None or requirement.marker.evaluate():
            requirements.append(requirement)
    return requirements


def test_active_requirements_match_the_current_python_environment() -> None:
    python_version = Version(
        ".".join(str(part) for part in sys.version_info[:3])
    )
    problems: list[str] = []

    for requirement in active_requirements():
        try:
            distribution = importlib.metadata.distribution(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{requirement.name} is not installed")
            continue

        installed_version = Version(distribution.version)
        if (
            requirement.specifier
            and installed_version not in requirement.specifier
        ):
            problems.append(
                f"{requirement.name} {installed_version} does not satisfy "
                f"{requirement.specifier}"
            )

        requires_python = distribution.metadata.get("Requires-Python")
        if requires_python:
            from packaging.specifiers import SpecifierSet

            if python_version not in SpecifierSet(requires_python):
                problems.append(
                    f"{requirement.name} requires Python {requires_python}, "
                    f"not {python_version}"
                )

    assert not problems, "\n".join(problems)
