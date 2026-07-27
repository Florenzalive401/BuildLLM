from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch import nn

from src.config import config
from src.kv_cache import ModelKVCache
from src.sampling import sample_next_token


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = config.generation_max_tokens
    do_sample: bool = True
    temperature: float = config.generation_temperature
    top_k: int = config.generation_top_k
    top_p: float = config.generation_top_p
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    stop_token_ids: tuple[int, ...] = ()
    stop_sequences: tuple[tuple[int, ...], ...] = ()
    random_seed: int | None = None
    use_kv_cache: bool = False

    def validate(
        self,
        vocabulary_size: int,
    ) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero"
            )

        if not isinstance(self.do_sample, bool):
            raise TypeError(
                "do_sample must be a boolean"
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero"
            )

        if self.top_k < 0:
            raise ValueError(
                "top_k cannot be negative"
            )

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(
                "top_p must be greater than zero and less than or equal to one"
            )

        if self.repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty must be greater than zero"
            )

        if not isinstance(
            self.no_repeat_ngram_size,
            int,
        ):
            raise TypeError(
                "no_repeat_ngram_size must be an integer"
            )

        if self.no_repeat_ngram_size < 0:
            raise ValueError(
                "no_repeat_ngram_size cannot be negative"
            )

        if not isinstance(self.use_kv_cache, bool):
            raise TypeError(
                "use_kv_cache must be a boolean"
            )

        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise TypeError(
                    "random_seed must be an integer or None"
                )

            if self.random_seed < 0:
                raise ValueError(
                    "random_seed cannot be negative"
                )

        self._validate_stop_token_ids(
            vocabulary_size=vocabulary_size
        )
        self._validate_stop_sequences(
            vocabulary_size=vocabulary_size
        )

    def _validate_stop_token_ids(
        self,
        vocabulary_size: int,
    ) -> None:
        if not isinstance(self.stop_token_ids, tuple):
            raise TypeError(
                "stop_token_ids must be a tuple"
            )

        if len(set(self.stop_token_ids)) != len(
            self.stop_token_ids
        ):
            raise ValueError(
                "stop_token_ids cannot contain duplicates"
            )

        for token_id in self.stop_token_ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    "stop_token_ids must contain integers"
                )

            if not 0 <= token_id < vocabulary_size:
                raise ValueError(
                    "stop_token_ids contain a value outside the vocabulary"
                )

    def _validate_stop_sequences(
        self,
        vocabulary_size: int,
    ) -> None:
        if not isinstance(self.stop_sequences, tuple):
            raise TypeError(
                "stop_sequences must be a tuple"
            )

        if len(set(self.stop_sequences)) != len(
            self.stop_sequences
        ):
            raise ValueError(
                "stop_sequences cannot contain duplicates"
            )

        for sequence in self.stop_sequences:
            if not isinstance(sequence, tuple):
                raise TypeError(
                    "each stop sequence must be a tuple"
                )

            if not sequence:
                raise ValueError(
                    "stop sequences cannot be empty"
                )

            for token_id in sequence:
                if not isinstance(token_id, int):
                    raise TypeError(
                        "stop sequences must contain integers"
                    )

                if not 0 <= token_id < vocabulary_size:
                    raise ValueError(
                        "stop sequences contain a value outside the vocabulary"
                    )


@dataclass(frozen=True)
class GenerationResult:
    sequences: tuple[Tensor, ...]
    generated_token_ids: tuple[Tensor, ...]
    log_probabilities: tuple[Tensor, ...]
    finish_reasons: tuple[str, ...]
    prompt_length: int

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def generated_lengths(self) -> tuple[int, ...]:
        return tuple(
            int(token_ids.numel())
            for token_ids in self.generated_token_ids
        )


@dataclass(frozen=True)
class GenerationStep:
    token_id: int
    log_probability: float
    generated_token_count: int
    is_finished: bool
    finish_reason: str | None


class Generator:
    def __init__(
        self,
        model: nn.Module,
        model_config: Any | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be a torch.nn.Module"
            )

        resolved_config = (
            model_config
            if model_config is not None
            else getattr(model, "model_config", None)
        )

        if resolved_config is None:
            raise ValueError(
                "model_config is required when the model does not expose model_config"
            )

        resolved_config.validate()

        self.model = model
        self.model_config = resolved_config
        self.vocabulary_size = int(
            resolved_config.vocabulary_size
        )
        self.maximum_sequence_length = int(
            resolved_config.maximum_sequence_length
        )

    def generate(
        self,
        token_ids: Tensor,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        resolved_config = generation_config or GenerationConfig()
        resolved_config.validate(
            vocabulary_size=self.vocabulary_size
        )
        self._validate_prompt(token_ids)

        model_device = self._model_device()
        active_token_ids = token_ids.to(
            device=model_device
        )
        prompt_length = active_token_ids.shape[1]
        batch_size = active_token_ids.shape[0]

        generated_tokens: list[list[Tensor]] = [
            [] for _ in range(batch_size)
        ]
        generated_log_probabilities: list[list[Tensor]] = [
            [] for _ in range(batch_size)
        ]
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=model_device,
        )
        finish_reasons = [
            "maximum_new_tokens"
            for _ in range(batch_size)
        ]

        random_generator = self._create_random_generator(
            generation_config=resolved_config,
            device=model_device,
        )
        model_cache: ModelKVCache | None = None

        original_training_state = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                for _ in range(
                    resolved_config.max_new_tokens
                ):
                    (
                        logits,
                        model_cache,
                        model_input_length,
                    ) = self._forward_model(
                        active_token_ids=active_token_ids,
                        model_cache=model_cache,
                        use_kv_cache=resolved_config.use_kv_cache,
                    )
                    self._validate_model_output(
                        logits=logits,
                        batch_size=batch_size,
                        sequence_length=model_input_length,
                    )

                    next_token_logits = logits[:, -1, :]

                    sampling_result = sample_next_token(
                        logits=next_token_logits,
                        do_sample=resolved_config.do_sample,
                        temperature=resolved_config.temperature,
                        top_k=resolved_config.top_k,
                        top_p=resolved_config.top_p,
                        repetition_penalty=(
                            resolved_config.repetition_penalty
                        ),
                        no_repeat_ngram_size=(
                            resolved_config.no_repeat_ngram_size
                        ),
                        token_ids=active_token_ids,
                        generator=random_generator,
                    )

                    next_token_ids = sampling_result.token_ids
                    next_log_probabilities = (
                        sampling_result.log_probabilities
                    )

                    if finished.any():
                        next_token_ids = torch.where(
                            finished,
                            active_token_ids[:, -1],
                            next_token_ids,
                        )
                        next_log_probabilities = torch.where(
                            finished,
                            torch.zeros_like(
                                next_log_probabilities
                            ),
                            next_log_probabilities,
                        )

                    active_token_ids = torch.cat(
                        (
                            active_token_ids,
                            next_token_ids.unsqueeze(-1),
                        ),
                        dim=-1,
                    )

                    for batch_index in range(batch_size):
                        if bool(finished[batch_index].item()):
                            continue

                        generated_tokens[batch_index].append(
                            next_token_ids[batch_index].detach().cpu()
                        )
                        generated_log_probabilities[batch_index].append(
                            next_log_probabilities[batch_index]
                            .detach()
                            .cpu()
                        )

                        finish_reason = self._finish_reason(
                            generated_token_ids=generated_tokens[
                                batch_index
                            ],
                            latest_token_id=int(
                                next_token_ids[
                                    batch_index
                                ].item()
                            ),
                            generation_config=resolved_config,
                        )

                        if finish_reason is not None:
                            finished[batch_index] = True
                            finish_reasons[
                                batch_index
                            ] = finish_reason

                    if bool(finished.all().item()):
                        break
        finally:
            self.model.train(
                original_training_state
            )

        result_sequences: list[Tensor] = []
        result_generated_tokens: list[Tensor] = []
        result_log_probabilities: list[Tensor] = []
        prompt_cpu = token_ids.detach().cpu()

        for batch_index in range(batch_size):
            generated_tensor = self._stack_token_values(
                generated_tokens[batch_index],
                dtype=torch.long,
            )
            log_probability_tensor = self._stack_token_values(
                generated_log_probabilities[batch_index],
                dtype=torch.float32,
            )

            result_generated_tokens.append(
                generated_tensor
            )
            result_log_probabilities.append(
                log_probability_tensor
            )
            result_sequences.append(
                torch.cat(
                    (
                        prompt_cpu[batch_index],
                        generated_tensor,
                    ),
                    dim=0,
                )
            )

        return GenerationResult(
            sequences=tuple(result_sequences),
            generated_token_ids=tuple(
                result_generated_tokens
            ),
            log_probabilities=tuple(
                result_log_probabilities
            ),
            finish_reasons=tuple(finish_reasons),
            prompt_length=prompt_length,
        )

    def stream(
        self,
        token_ids: Tensor,
        generation_config: GenerationConfig | None = None,
    ) -> Iterator[GenerationStep]:
        self._validate_prompt(token_ids)

        if token_ids.shape[0] != 1:
            raise ValueError(
                "streaming generation requires batch_size equal to one"
            )

        resolved_config = generation_config or GenerationConfig()
        resolved_config.validate(
            vocabulary_size=self.vocabulary_size
        )

        model_device = self._model_device()
        active_token_ids = token_ids.to(
            device=model_device
        )
        generated_token_ids: list[Tensor] = []
        random_generator = self._create_random_generator(
            generation_config=resolved_config,
            device=model_device,
        )
        model_cache: ModelKVCache | None = None

        original_training_state = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                for generated_index in range(
                    resolved_config.max_new_tokens
                ):
                    (
                        logits,
                        model_cache,
                        model_input_length,
                    ) = self._forward_model(
                        active_token_ids=active_token_ids,
                        model_cache=model_cache,
                        use_kv_cache=resolved_config.use_kv_cache,
                    )
                    self._validate_model_output(
                        logits=logits,
                        batch_size=1,
                        sequence_length=model_input_length,
                    )

                    sampling_result = sample_next_token(
                        logits=logits[:, -1, :],
                        do_sample=resolved_config.do_sample,
                        temperature=resolved_config.temperature,
                        top_k=resolved_config.top_k,
                        top_p=resolved_config.top_p,
                        repetition_penalty=(
                            resolved_config.repetition_penalty
                        ),
                        no_repeat_ngram_size=(
                            resolved_config.no_repeat_ngram_size
                        ),
                        token_ids=active_token_ids,
                        generator=random_generator,
                    )

                    next_token_id = sampling_result.token_ids
                    active_token_ids = torch.cat(
                        (
                            active_token_ids,
                            next_token_id.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                    generated_token_ids.append(
                        next_token_id[0].detach().cpu()
                    )

                    finish_reason = self._finish_reason(
                        generated_token_ids=generated_token_ids,
                        latest_token_id=int(
                            next_token_id[0].item()
                        ),
                        generation_config=resolved_config,
                    )

                    if (
                        finish_reason is None
                        and generated_index + 1
                        == resolved_config.max_new_tokens
                    ):
                        finish_reason = "maximum_new_tokens"

                    yield GenerationStep(
                        token_id=int(
                            next_token_id[0].item()
                        ),
                        log_probability=float(
                            sampling_result.log_probabilities[
                                0
                            ].item()
                        ),
                        generated_token_count=(
                            generated_index + 1
                        ),
                        is_finished=(
                            finish_reason is not None
                        ),
                        finish_reason=finish_reason,
                    )

                    if finish_reason is not None:
                        break
        finally:
            self.model.train(
                original_training_state
            )

    def _forward_model(
        self,
        *,
        active_token_ids: Tensor,
        model_cache: ModelKVCache | None,
        use_kv_cache: bool,
    ) -> tuple[Tensor, ModelKVCache | None, int]:
        if not use_kv_cache:
            context_token_ids = active_token_ids[
                :,
                -self.maximum_sequence_length :,
            ]
            logits = self.model(context_token_ids)
            return logits, None, context_token_ids.shape[1]

        if (
            model_cache is None
            or model_cache.sequence_length
            >= self.maximum_sequence_length
        ):
            context_token_ids = active_token_ids[
                :,
                -self.maximum_sequence_length :,
            ]
            logits, updated_cache = self.model(
                context_token_ids,
                cache=None,
                use_cache=True,
            )
            return (
                logits,
                updated_cache,
                context_token_ids.shape[1],
            )

        next_input_token_ids = active_token_ids[:, -1:]
        logits, updated_cache = self.model(
            next_input_token_ids,
            cache=model_cache,
            use_cache=True,
        )
        return logits, updated_cache, 1

    def _validate_prompt(
        self,
        token_ids: Tensor,
    ) -> None:
        if not isinstance(token_ids, Tensor):
            raise TypeError(
                "token_ids must be a torch.Tensor"
            )

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape batch_size, sequence_length"
            )

        if token_ids.dtype != torch.long:
            raise TypeError(
                "token_ids must use torch.long"
            )

        if token_ids.shape[0] <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        if token_ids.shape[1] <= 0:
            raise ValueError(
                "sequence_length must be greater than zero"
            )

        if token_ids.shape[1] > self.maximum_sequence_length:
            raise ValueError(
                "prompt sequence_length exceeds maximum_sequence_length"
            )

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

        if maximum_token_id >= self.vocabulary_size:
            raise ValueError(
                "token_ids contain a value outside the configured vocabulary"
            )

    def _validate_model_output(
        self,
        logits: Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if not isinstance(logits, Tensor):
            raise TypeError(
                "model output must be a torch.Tensor"
            )

        expected_shape = (
            batch_size,
            sequence_length,
            self.vocabulary_size,
        )

        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                "model output must have shape batch_size, "
                "sequence_length, vocabulary_size"
            )

        if not logits.is_floating_point():
            raise TypeError(
                "model output logits must use a floating point dtype"
            )

    def _model_device(self) -> torch.device:
        try:
            return next(
                self.model.parameters()
            ).device
        except StopIteration:
            return torch.device(
                self.model_config.device
            )

    @staticmethod
    def _create_random_generator(
        generation_config: GenerationConfig,
        device: torch.device,
    ) -> torch.Generator | None:
        if generation_config.random_seed is None:
            return None

        generator_device = (
            device.type
            if device.type in {"cpu", "cuda"}
            else "cpu"
        )
        random_generator = torch.Generator(
            device=generator_device
        )
        random_generator.manual_seed(
            generation_config.random_seed
        )
        return random_generator

    @staticmethod
    def _finish_reason(
        generated_token_ids: list[Tensor],
        latest_token_id: int,
        generation_config: GenerationConfig,
    ) -> str | None:
        if latest_token_id in generation_config.stop_token_ids:
            return "stop_token"

        for stop_sequence in generation_config.stop_sequences:
            if len(generated_token_ids) < len(stop_sequence):
                continue

            suffix = tuple(
                int(token_id.item())
                for token_id in generated_token_ids[
                    -len(stop_sequence) :
                ]
            )

            if suffix == stop_sequence:
                return "stop_sequence"

        return None

    @staticmethod
    def _stack_token_values(
        values: list[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        if not values:
            return torch.empty(
                0,
                dtype=dtype,
            )

        return torch.stack(values).to(
            dtype=dtype
        )
