from src.corpus.dedup import DuplicateIndex, hamming_distance, normalized_fingerprint, simhash64


def test_exact_duplicate_detection_ignores_spacing_and_case():
    index = DuplicateIndex()
    first = "Reliable software needs clear requirements and repeatable testing. " * 20
    second = "  RELIABLE   SOFTWARE needs clear requirements and repeatable testing. " * 20
    assert index.classify(first) is None
    assert index.classify(second) == "exact_duplicate"


def test_simhash_helpers_are_stable():
    text = "Network security uses authentication, encryption, logging, and monitoring. " * 20
    assert normalized_fingerprint(text) == normalized_fingerprint(text)
    assert simhash64(text) == simhash64(text)
    assert hamming_distance(1, 3) == 1
