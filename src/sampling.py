from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingResult:
    token_ids: Tensor
    log_probabilities: Tensor


def apply_temperature(
    logits: Tensor,
    temperature: float,
) -> Tensor:
    _validate_logits(logits)

    if not isinstance(temperature, (int, float)):
        raise TypeError(
            "temperature must be a number"
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero"
        )

    return logits / float(temperature)


def apply_repetition_penalty(
    logits: Tensor,
    token_ids: Tensor,
    penalty: float,
) -> Tensor:
    _validate_logits(logits)
    _validate_token_history(
        token_ids=token_ids,
        batch_size=logits.shape[0],
    )

    if not isinstance(penalty, (int, float)):
        raise TypeError(
            "penalty must be a number"
        )

    if penalty <= 0:
        raise ValueError(
            "penalty must be greater than zero"
        )

    if penalty == 1.0:
        return logits.clone()

    adjusted_logits = logits.clone()
    vocabulary_size = logits.shape[-1]

    if token_ids.numel() == 0:
        return adjusted_logits

    minimum_token_id = int(
        token_ids.min().item()
    )
    maximum_token_id = int(
        token_ids.max().item()
    )

    if minimum_token_id < 0:
        raise ValueError(
            "token_ids cannot contain negative values"
        )

    if maximum_token_id >= vocabulary_size:
        raise ValueError(
            "token_ids contain a value outside the logits vocabulary"
        )

    for batch_index in range(logits.shape[0]):
        unique_token_ids = torch.unique(
            token_ids[batch_index]
        )

        selected_logits = adjusted_logits[
            batch_index,
            unique_token_ids,
        ]

        penalized_logits = torch.where(
            selected_logits < 0,
            selected_logits * float(penalty),
            selected_logits / float(penalty),
        )

        adjusted_logits[
            batch_index,
            unique_token_ids,
        ] = penalized_logits

    return adjusted_logits


def apply_no_repeat_ngram(
    logits: Tensor,
    token_ids: Tensor,
    ngram_size: int,
) -> Tensor:
    _validate_logits(logits)
    _validate_token_history(
        token_ids=token_ids,
        batch_size=logits.shape[0],
    )

    if not isinstance(ngram_size, int):
        raise TypeError(
            "ngram_size must be an integer"
        )

    if ngram_size < 0:
        raise ValueError(
            "ngram_size cannot be negative"
        )

    if ngram_size == 0 or token_ids.numel() == 0:
        return logits.clone()

    adjusted_logits = logits.clone()

    for batch_index in range(logits.shape[0]):
        history = [
            int(token_id)
            for token_id in token_ids[
                batch_index
            ].tolist()
        ]
        banned_token_ids = _banned_ngram_tokens(
            history=history,
            ngram_size=ngram_size,
        )

        if not banned_token_ids:
            continue

        adjusted_logits[
            batch_index,
            banned_token_ids,
        ] = float("-inf")

        if torch.all(
            torch.isneginf(
                adjusted_logits[batch_index]
            )
        ):
            adjusted_logits[batch_index] = logits[
                batch_index
            ]

    return adjusted_logits


def filter_top_k(
    logits: Tensor,
    top_k: int,
) -> Tensor:
    _validate_logits(logits)

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k must be an integer"
        )

    if top_k < 0:
        raise ValueError(
            "top_k cannot be negative"
        )

    vocabulary_size = logits.shape[-1]

    if top_k == 0 or top_k >= vocabulary_size:
        return logits.clone()

    threshold = torch.topk(
        logits,
        k=top_k,
        dim=-1,
    ).values[..., -1, None]

    return logits.masked_fill(
        logits < threshold,
        float("-inf"),
    )


def filter_top_p(
    logits: Tensor,
    top_p: float,
) -> Tensor:
    _validate_logits(logits)

    if not isinstance(top_p, (int, float)):
        raise TypeError(
            "top_p must be a number"
        )

    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(
            "top_p must be greater than zero and less than or equal to one"
        )

    if float(top_p) == 1.0:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    sorted_remove_mask = (
        cumulative_probabilities > float(top_p)
    )

    sorted_remove_mask[..., 1:] = (
        sorted_remove_mask[..., :-1].clone()
    )
    sorted_remove_mask[..., 0] = False

    remove_mask = torch.zeros_like(
        sorted_remove_mask
    )
    remove_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_remove_mask,
    )

    return logits.masked_fill(
        remove_mask,
        float("-inf"),
    )


def sample_next_token(
    logits: Tensor,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    token_ids: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SamplingResult:
    _validate_logits(logits)

    processed_logits = logits

    if token_ids is not None:
        processed_logits = apply_repetition_penalty(
            logits=processed_logits,
            token_ids=token_ids,
            penalty=repetition_penalty,
        )
        processed_logits = apply_no_repeat_ngram(
            logits=processed_logits,
            token_ids=token_ids,
            ngram_size=no_repeat_ngram_size,
        )
    elif repetition_penalty != 1.0:
        raise ValueError(
            "token_ids are required when repetition_penalty is not one"
        )
    elif no_repeat_ngram_size != 0:
        raise ValueError(
            "token_ids are required when no_repeat_ngram_size is not zero"
        )

    if do_sample:
        processed_logits = apply_temperature(
            logits=processed_logits,
            temperature=temperature,
        )
        processed_logits = filter_top_k(
            logits=processed_logits,
            top_k=top_k,
        )
        processed_logits = filter_top_p(
            logits=processed_logits,
            top_p=top_p,
        )

    log_probabilities = torch.log_softmax(
        processed_logits,
        dim=-1,
    )

    if do_sample:
        probabilities = torch.softmax(
            processed_logits,
            dim=-1,
        )

        if not torch.isfinite(probabilities).all():
            raise RuntimeError(
                "sampling probabilities contain nonfinite values"
            )

        if torch.any(probabilities.sum(dim=-1) <= 0):
            raise RuntimeError(
                "sampling probabilities contain an empty distribution"
            )

        selected_token_ids = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
    else:
        selected_token_ids = torch.argmax(
            processed_logits,
            dim=-1,
        )

    selected_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=selected_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    return SamplingResult(
        token_ids=selected_token_ids,
        log_probabilities=selected_log_probabilities,
    )


def _validate_logits(
    logits: Tensor,
) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError(
            "logits must be a torch.Tensor"
        )

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape batch_size, vocabulary_size"
        )

    if logits.shape[0] <= 0:
        raise ValueError(
            "batch_size must be greater than zero"
        )

    if logits.shape[1] <= 0:
        raise ValueError(
            "vocabulary_size must be greater than zero"
        )

    if not logits.is_floating_point():
        raise TypeError(
            "logits must use a floating point dtype"
        )

    if torch.isnan(logits).any():
        raise ValueError(
            "logits cannot contain NaN values"
        )

    if torch.isposinf(logits).any():
        raise ValueError(
            "logits cannot contain positive infinity"
        )

    if torch.all(torch.isneginf(logits), dim=-1).any():
        raise ValueError(
            "each logits row must contain at least one finite value"
        )


def _banned_ngram_tokens(
    *,
    history: list[int],
    ngram_size: int,
) -> list[int]:
    if ngram_size == 1:
        return sorted(set(history))

    prefix_length = ngram_size - 1

    if len(history) < prefix_length:
        return []

    prefix = tuple(history[-prefix_length:])
    banned_token_ids: set[int] = set()

    for start_index in range(
        len(history) - ngram_size + 1
    ):
        ngram = history[
            start_index : start_index + ngram_size
        ]

        if tuple(ngram[:-1]) == prefix:
            banned_token_ids.add(int(ngram[-1]))

    return sorted(banned_token_ids)


def _validate_token_history(
    *,
    token_ids: Tensor,
    batch_size: int,
) -> None:
    if not isinstance(token_ids, Tensor):
        raise TypeError(
            "token_ids must be a torch.Tensor"
        )

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape batch_size, sequence_length"
        )

    if token_ids.shape[0] != batch_size:
        raise ValueError(
            "token_ids batch_size must match logits batch_size"
        )

    if token_ids.dtype != torch.long:
        raise TypeError(
            "token_ids must use torch.long"
        )


def _validate_token_history(
    token_ids: Tensor,
    batch_size: int,
) -> None:
    if not isinstance(token_ids, Tensor):
        raise TypeError(
            "token_ids must be a torch.Tensor"
        )

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape batch_size, sequence_length"
        )

    if token_ids.shape[0] != batch_size:
        raise ValueError(
            "token_ids and logits must have matching batch sizes"
        )

    if token_ids.dtype != torch.long:
        raise TypeError(
            "token_ids must use torch.long"
        )
