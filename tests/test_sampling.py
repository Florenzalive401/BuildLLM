import pytest
import torch

from src.sampling import apply_no_repeat_ngram
from src.sampling import SamplingResult
from src.sampling import apply_repetition_penalty
from src.sampling import apply_temperature
from src.sampling import filter_top_k
from src.sampling import filter_top_p
from src.sampling import sample_next_token


def test_apply_temperature_divides_logits() -> None:
    logits = torch.tensor([[2.0, 4.0]])

    result = apply_temperature(
        logits=logits,
        temperature=2.0,
    )

    assert torch.equal(
        result,
        torch.tensor([[1.0, 2.0]]),
    )


def test_apply_temperature_does_not_mutate_input() -> None:
    logits = torch.tensor([[2.0, 4.0]])
    original = logits.clone()

    apply_temperature(logits, 2.0)

    assert torch.equal(logits, original)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_apply_temperature_rejects_nonpositive_values(
    temperature: float,
) -> None:
    with pytest.raises(ValueError):
        apply_temperature(
            torch.tensor([[1.0]]),
            temperature,
        )


def test_apply_temperature_rejects_non_numeric_value() -> None:
    with pytest.raises(TypeError):
        apply_temperature(
            torch.tensor([[1.0]]),
            "warm",  # type: ignore[arg-type]
        )


def test_repetition_penalty_reduces_positive_logits() -> None:
    logits = torch.tensor([[4.0, 2.0, 1.0]])
    token_ids = torch.tensor([[0, 1]], dtype=torch.long)

    result = apply_repetition_penalty(
        logits,
        token_ids,
        2.0,
    )

    assert torch.equal(
        result,
        torch.tensor([[2.0, 1.0, 1.0]]),
    )


def test_repetition_penalty_makes_negative_logits_more_negative() -> None:
    logits = torch.tensor([[-2.0, 1.0]])
    token_ids = torch.tensor([[0]], dtype=torch.long)

    result = apply_repetition_penalty(
        logits,
        token_ids,
        2.0,
    )

    assert torch.equal(
        result,
        torch.tensor([[-4.0, 1.0]]),
    )


def test_repetition_penalty_handles_batch_rows_independently() -> None:
    logits = torch.tensor(
        [[4.0, 2.0], [6.0, 8.0]]
    )
    token_ids = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    result = apply_repetition_penalty(
        logits,
        token_ids,
        2.0,
    )

    assert torch.equal(
        result,
        torch.tensor(
            [[2.0, 2.0], [6.0, 4.0]]
        ),
    )


def test_repetition_penalty_one_returns_equal_copy() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    token_ids = torch.tensor([[0]], dtype=torch.long)

    result = apply_repetition_penalty(
        logits,
        token_ids,
        1.0,
    )

    assert torch.equal(result, logits)
    assert result.data_ptr() != logits.data_ptr()


def test_repetition_penalty_rejects_out_of_range_history() -> None:
    with pytest.raises(ValueError):
        apply_repetition_penalty(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[2]], dtype=torch.long),
            2.0,
        )


def test_no_repeat_ngram_bans_repeated_bigram_completion() -> None:
    logits = torch.tensor([[1.0, 3.0, 10.0, 2.0]])
    token_ids = torch.tensor(
        [[1, 2, 1]],
        dtype=torch.long,
    )

    result = apply_no_repeat_ngram(
        logits,
        token_ids,
        2,
    )

    assert torch.isneginf(result[0, 2])
    assert torch.isfinite(result[0, 1])


def test_no_repeat_ngram_bans_seen_tokens_when_size_is_one() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0]])
    token_ids = torch.tensor(
        [[0, 2]],
        dtype=torch.long,
    )

    result = apply_no_repeat_ngram(
        logits,
        token_ids,
        1,
    )

    assert torch.isneginf(result[0, 0])
    assert torch.isneginf(result[0, 2])
    assert torch.isfinite(result[0, 1])


def test_no_repeat_ngram_noop_values_return_equal_copy() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    token_ids = torch.tensor([[0]], dtype=torch.long)

    result = apply_no_repeat_ngram(
        logits,
        token_ids,
        0,
    )

    assert torch.equal(result, logits)
    assert result.data_ptr() != logits.data_ptr()


def test_no_repeat_ngram_restores_row_when_every_token_is_banned() -> None:
    logits = torch.tensor([[5.0, 4.0]])
    token_ids = torch.tensor(
        [[0, 1, 0, 1]],
        dtype=torch.long,
    )

    result = apply_no_repeat_ngram(
        logits,
        token_ids,
        1,
    )

    assert torch.equal(result, logits)


@pytest.mark.parametrize("ngram_size", [-1, "three"])
def test_no_repeat_ngram_rejects_invalid_size(
    ngram_size: int | str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_no_repeat_ngram(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0]], dtype=torch.long),
            ngram_size,  # type: ignore[arg-type]
        )


def test_top_k_keeps_only_requested_number_of_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])

    result = filter_top_k(logits, 2)

    assert torch.isneginf(result[0, 0])
    assert result[0, 1] == 4.0
    assert result[0, 2] == 3.0
    assert torch.isneginf(result[0, 3])


@pytest.mark.parametrize("top_k", [0, 4, 8])
def test_top_k_noop_values_return_equal_copy(
    top_k: int,
) -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = filter_top_k(logits, top_k)

    assert torch.equal(result, logits)
    assert result.data_ptr() != logits.data_ptr()


def test_top_k_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        filter_top_k(
            torch.tensor([[1.0]]),
            -1,
        )


def test_top_p_keeps_minimum_nucleus() -> None:
    logits = torch.log(
        torch.tensor([[0.60, 0.25, 0.10, 0.05]])
    )

    result = filter_top_p(logits, 0.70)

    assert torch.isfinite(result[0, 0])
    assert torch.isfinite(result[0, 1])
    assert torch.isneginf(result[0, 2])
    assert torch.isneginf(result[0, 3])


def test_top_p_always_keeps_at_least_one_token() -> None:
    logits = torch.tensor([[10.0, 0.0, -1.0]])

    result = filter_top_p(logits, 0.01)

    assert torch.isfinite(result[0, 0])
    assert torch.isneginf(result[0, 1:]).all()


def test_top_p_one_returns_equal_copy() -> None:
    logits = torch.tensor([[1.0, 2.0]])

    result = filter_top_p(logits, 1.0)

    assert torch.equal(result, logits)
    assert result.data_ptr() != logits.data_ptr()


@pytest.mark.parametrize("top_p", [0.0, -0.5, 1.1])
def test_top_p_rejects_invalid_values(
    top_p: float,
) -> None:
    with pytest.raises(ValueError):
        filter_top_p(
            torch.tensor([[1.0]]),
            top_p,
        )


def test_greedy_sampling_selects_argmax() -> None:
    result = sample_next_token(
        logits=torch.tensor(
            [[1.0, 5.0, 2.0], [9.0, 1.0, 0.0]]
        ),
        do_sample=False,
    )

    assert isinstance(result, SamplingResult)
    assert torch.equal(
        result.token_ids,
        torch.tensor([1, 0]),
    )
    assert result.log_probabilities.shape == (2,)


def test_greedy_sampling_ignores_temperature_and_filters() -> None:
    result = sample_next_token(
        logits=torch.tensor([[1.0, 5.0, 2.0]]),
        do_sample=False,
        temperature=0.0001,
        top_k=1,
        top_p=0.01,
    )

    assert result.token_ids.item() == 1


def test_seeded_sampling_is_reproducible() -> None:
    logits = torch.zeros((1, 5))
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first = sample_next_token(
        logits=logits,
        do_sample=True,
        generator=first_generator,
    )
    second = sample_next_token(
        logits=logits,
        do_sample=True,
        generator=second_generator,
    )

    assert torch.equal(
        first.token_ids,
        second.token_ids,
    )


def test_sampling_top_k_one_is_deterministic() -> None:
    result = sample_next_token(
        logits=torch.tensor([[1.0, 5.0, 2.0]]),
        do_sample=True,
        top_k=1,
    )

    assert result.token_ids.item() == 1


def test_sampling_requires_history_for_repetition_penalty() -> None:
    with pytest.raises(ValueError):
        sample_next_token(
            logits=torch.tensor([[1.0, 2.0]]),
            do_sample=False,
            repetition_penalty=2.0,
        )


def test_sampling_requires_history_for_no_repeat_ngram() -> None:
    with pytest.raises(ValueError):
        sample_next_token(
            logits=torch.tensor([[1.0, 2.0]]),
            do_sample=False,
            no_repeat_ngram_size=2,
        )


def test_greedy_sampling_respects_no_repeat_ngram() -> None:
    result = sample_next_token(
        logits=torch.tensor([[1.0, 5.0, 10.0, 3.0]]),
        do_sample=False,
        no_repeat_ngram_size=2,
        token_ids=torch.tensor(
            [[1, 2, 1]],
            dtype=torch.long,
        ),
    )

    assert result.token_ids.item() == 1


@pytest.mark.parametrize(
    "logits,exception_type",
    [
        (torch.tensor([1.0, 2.0]), ValueError),
        (torch.tensor([[1, 2]]), TypeError),
        (torch.tensor([[float("nan"), 1.0]]), ValueError),
        (torch.tensor([[float("inf"), 1.0]]), ValueError),
        (torch.tensor([[float("-inf"), float("-inf")]]), ValueError),
    ],
)
def test_sampling_rejects_invalid_logits(
    logits: torch.Tensor,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        sample_next_token(
            logits=logits,
            do_sample=False,
        )
