"""Lightweight topic labeling used for corpus reporting and balancing."""

from __future__ import annotations

import re
from collections import Counter


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "technology": ("software", "computer", "internet", "database", "algorithm", "programming", "network", "cloud"),
    "cybersecurity": ("cybersecurity", "vulnerability", "malware", "encryption", "firewall", "authentication", "threat", "security control"),
    "science": ("scientific", "physics", "chemistry", "biology", "experiment", "molecule", "species", "research"),
    "mathematics": ("mathematics", "theorem", "equation", "geometry", "calculus", "probability", "algebra", "integer"),
    "history": ("history", "century", "war", "empire", "dynasty", "historical", "revolution", "ancient"),
    "government": ("government", "law", "policy", "regulation", "congress", "court", "agency", "constitution"),
    "business": ("business", "company", "market", "finance", "economic", "management", "revenue", "industry"),
    "health": ("health", "medical", "disease", "patient", "medicine", "clinical", "treatment", "doctor"),
    "arts_and_culture": ("music", "film", "literature", "artist", "novel", "culture", "painting", "theatre"),
    "geography": ("country", "city", "river", "mountain", "region", "population", "province", "geography"),
}


def classify_topic(text: str) -> str:
    sample = text[:20_000].lower()
    scores: Counter[str] = Counter()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            scores[topic] += len(re.findall(rf"\b{re.escape(keyword)}\b", sample))
    if not scores:
        return "general"
    topic, score = scores.most_common(1)[0]
    return topic if score >= 2 else "general"
