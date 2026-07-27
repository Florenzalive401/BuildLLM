"""Deterministic document quality scoring for pretraining corpora."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")
_BOILERPLATE = (
    "cookie policy",
    "accept cookies",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "enable javascript",
    "subscribe to our newsletter",
    "sign up for our newsletter",
    "click here",
)


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    score: float
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def assess_quality(
    text: str,
    minimum_score: float = 55.0,
    minimum_characters: int = 500,
    maximum_characters: int = 2_000_000,
) -> QualityAssessment:
    """Score a document from zero to one hundred without model inference."""

    length = len(text)
    words = _WORD_RE.findall(text)
    word_count = len(words)
    alphabetic = sum(character.isalpha() for character in text)
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    alpha_ratio = alphabetic / max(length, 1)
    printable_ratio = printable / max(length, 1)
    unique_ratio = len({word.lower() for word in words}) / max(word_count, 1)
    sentence_count = len(_SENTENCE_RE.findall(text))
    url_count = len(_URL_RE.findall(text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_line_ratio = sum(len(line) < 35 for line in lines) / max(len(lines), 1)
    boilerplate_hits = sum(text.lower().count(phrase) for phrase in _BOILERPLATE)
    repeated_character_runs = len(re.findall(r"(.)\1{7,}", text))

    reasons: list[str] = []
    hard_reject = False
    if length < minimum_characters:
        reasons.append("too_short")
        hard_reject = True
    if length > maximum_characters:
        reasons.append("too_long")
        hard_reject = True
    if alpha_ratio < 0.45:
        reasons.append("low_alphabetic_ratio")
        hard_reject = True
    if printable_ratio < 0.98:
        reasons.append("control_or_binary_content")
        hard_reject = True
    if word_count < 80:
        reasons.append("too_few_words")
        hard_reject = True

    length_score = _clamp(math.log10(max(length, 1) / minimum_characters + 1) / 2.2)
    alpha_score = _clamp((alpha_ratio - 0.45) / 0.30)
    vocabulary_score = _clamp((unique_ratio - 0.12) / 0.43)
    sentence_score = _clamp(sentence_count / max(word_count / 18, 1))
    structure_score = 1.0 - _clamp((short_line_ratio - 0.45) / 0.45)
    url_score = 1.0 - _clamp(url_count / max(word_count / 150, 1))
    boilerplate_score = 1.0 - _clamp(boilerplate_hits / 4)
    repeat_score = 1.0 - _clamp(repeated_character_runs / 3)

    score = 100.0 * (
        0.18 * length_score
        + 0.18 * alpha_score
        + 0.17 * vocabulary_score
        + 0.16 * sentence_score
        + 0.12 * structure_score
        + 0.07 * url_score
        + 0.07 * boilerplate_score
        + 0.05 * repeat_score
    )
    score = round(score, 2)

    if boilerplate_hits >= 4:
        reasons.append("excessive_boilerplate")
    if repeated_character_runs >= 3:
        reasons.append("repeated_character_noise")
    if short_line_ratio > 0.85 and len(lines) > 20:
        reasons.append("fragmented_layout")

    accepted = not hard_reject and score >= minimum_score
    if not accepted and not reasons:
        reasons.append("quality_score_below_threshold")

    return QualityAssessment(
        score=score,
        accepted=accepted,
        reasons=tuple(reasons),
        metrics={
            "characters": float(length),
            "words": float(word_count),
            "alphabetic_ratio": round(alpha_ratio, 4),
            "printable_ratio": round(printable_ratio, 4),
            "unique_word_ratio": round(unique_ratio, 4),
            "sentence_count": float(sentence_count),
            "short_line_ratio": round(short_line_ratio, 4),
            "url_count": float(url_count),
            "boilerplate_hits": float(boilerplate_hits),
        },
    )
