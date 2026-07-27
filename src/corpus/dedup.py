"""Exact and near duplicate detection with bounded memory overhead."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict


_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def normalized_fingerprint(text: str) -> str:
    normalized = " ".join(_TOKEN_RE.findall(text.lower()))
    return hashlib.sha256(normalized.encode("utf8")).hexdigest()


def simhash64(text: str, maximum_tokens: int = 20_000) -> int:
    tokens = _TOKEN_RE.findall(text.lower())[:maximum_tokens]
    if not tokens:
        return 0
    weights = [0] * 64
    shingles = tokens if len(tokens) < 3 else (" ".join(tokens[index:index + 3]) for index in range(len(tokens) - 2))
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class DuplicateIndex:
    """Detect exact duplicates and high overlap documents using SimHash LSH."""

    def __init__(self, maximum_hamming_distance: int = 3) -> None:
        self.maximum_hamming_distance = maximum_hamming_distance
        self._exact: set[str] = set()
        self._bands: dict[tuple[int, int], list[int]] = defaultdict(list)

    @staticmethod
    def _band_keys(value: int) -> tuple[tuple[int, int], ...]:
        return tuple((band, (value >> (band * 16)) & 0xFFFF) for band in range(4))

    def classify(self, text: str) -> str | None:
        exact = normalized_fingerprint(text)
        if exact in self._exact:
            return "exact_duplicate"

        signature = simhash64(text)
        candidates: set[int] = set()
        for key in self._band_keys(signature):
            candidates.update(self._bands.get(key, ()))
        if any(hamming_distance(signature, candidate) <= self.maximum_hamming_distance for candidate in candidates):
            return "near_duplicate"

        self._exact.add(exact)
        for key in self._band_keys(signature):
            self._bands[key].append(signature)
        return None

    def __len__(self) -> int:
        return len(self._exact)
