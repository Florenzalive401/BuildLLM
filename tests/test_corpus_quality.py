from src.corpus.quality import assess_quality


def test_quality_accepts_well_formed_document():
    text = (
        "Software engineering is the disciplined design and operation of reliable systems. "
        "Engineers define requirements, evaluate tradeoffs, test behavior, and document decisions. "
        "A strong implementation uses clear interfaces, measurable outcomes, and repeatable validation. "
    ) * 8
    result = assess_quality(text, minimum_score=40)
    assert result.accepted
    assert result.score >= 40
    assert result.metrics["words"] >= 80


def test_quality_rejects_short_document():
    result = assess_quality("Too short.", minimum_score=0)
    assert not result.accepted
    assert "too_short" in result.reasons
