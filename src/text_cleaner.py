"""
Training text cleaning.

Raw text contains inconsistent spacing, control characters, repeated blank
lines, and documents that provide little value for model training.

This module cleans the source without rewriting the original content.
"""

import re
import unicodedata


MINIMUM_DOCUMENT_CHARACTERS = 500
MINIMUM_ALPHABETIC_RATIO = 0.50


def normalize_unicode(text: str) -> str:
    """Normalize equivalent Unicode representations."""

    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    """Remove control characters while preserving normal spacing."""

    return "".join(
        character
        for character in text
        if character in {"\n", "\t"}
        or unicodedata.category(character)[0] != "C"
    )


def normalize_whitespace(text: str) -> str:
    """Normalize spacing while preserving document structure."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def has_usable_content(text: str) -> bool:
    """Reject documents with too little usable language."""

    if len(text) < MINIMUM_DOCUMENT_CHARACTERS:
        return False

    alphabetic_characters = sum(
        character.isalpha()
        for character in text
    )

    alphabetic_ratio = alphabetic_characters / len(text)

    return alphabetic_ratio >= MINIMUM_ALPHABETIC_RATIO


def clean_document(text: str) -> str | None:
    """Clean one document and reject unusable content."""

    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)

    if not has_usable_content(text):
        return None

    return text