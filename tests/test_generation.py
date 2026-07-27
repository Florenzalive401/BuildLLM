from dataclasses import replace

import pytest
import torch
from torch import nn

from src.config import ModelConfig
from src.generation import GenerationConfig
from src.generation import GenerationResult
from src.generation import GenerationStep
from src.generation import Generator


class PredictableModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int = 8,
        maximum_sequence_length: int = 4,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model_config = ModelConfig(
            vocabulary_size=vocabulary_size,
            maximum_sequence_length=maximum_sequence_length,
            embedding_dimension=8,
            number_of_layers=1,
            number_of_attention_heads=1,
            feed_forward_dimension=16,
            dropout=0.0,
            batch_size=1,
            epochs=1,
            checkpoint_frequency=1,
            generation_max_tokens=4,
            generation_top_k=0,
            generation_top_p=1.0,
        )
        self.seen_sequence_lengths: list[int] = []

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_sequence_lengths.append(
            token_ids.shape[1]
        )
        batch_size, sequence_length = token_ids.shape
        logits = torch.full(
            (
                batch_size,
                sequence_length,
                self.model_config.vocabulary_size,
            ),
            -100.0,
            device=token_ids.device,
        )
        next_token_ids = (
            token_ids[:, -1] + 1
        ) % self.model_config.vocabulary_size
        logits[
            torch.arange(batch_size),
            sequence_length - 1,
            next_token_ids,
        ] = 100.0
        return logits


class UniformModel(PredictableModel):
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        return torch.zeros(
            (
                batch_size,
                sequence_length,
                self.model_config.vocabulary_size,
            ),
            device=token_ids.device,
        )


class InvalidShapeModel(PredictableModel):
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            token_ids.shape[0],
            self.model_config.vocabulary_size,
        )


def greedy_config(**changes: object) -> GenerationConfig:
    base = GenerationConfig(
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    return replace(base, **changes)


def test_generator_uses_model_config_automatically() -> None:
    model = PredictableModel()

    generator = Generator(model)

    assert generator.vocabulary_size == 8
    assert generator.maximum_sequence_length == 4


def test_generator_requires_config_when_model_has_none() -> None:
    with pytest.raises(ValueError):
        Generator(nn.Linear(2, 2))


def test_generate_returns_complete_result() -> None:
    model = PredictableModel()
    generator = Generator(model)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    result = generator.generate(
        prompt,
        greedy_config(max_new_tokens=3),
    )

    assert isinstance(result, GenerationResult)
    assert result.batch_size == 1
    assert result.prompt_length == 2
    assert result.generated_lengths == (3,)
    assert result.finish_reasons == (
        "maximum_new_tokens",
    )
    assert torch.equal(
        result.generated_token_ids[0],
        torch.tensor([3, 4, 5]),
    )
    assert torch.equal(
        result.sequences[0],
        torch.tensor([1, 2, 3, 4, 5]),
    )
    assert result.log_probabilities[0].shape == (3,)


def test_generate_supports_batches() -> None:
    model = PredictableModel()
    generator = Generator(model)
    prompt = torch.tensor(
        [[1, 2], [4, 5]],
        dtype=torch.long,
    )

    result = generator.generate(
        prompt,
        greedy_config(max_new_tokens=2),
    )

    assert result.batch_size == 2
    assert torch.equal(
        result.generated_token_ids[0],
        torch.tensor([3, 4]),
    )
    assert torch.equal(
        result.generated_token_ids[1],
        torch.tensor([6, 7]),
    )


def test_generate_stops_on_stop_token() -> None:
    model = PredictableModel()
    generator = Generator(model)

    result = generator.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        greedy_config(
            max_new_tokens=4,
            stop_token_ids=(4,),
        ),
    )

    assert torch.equal(
        result.generated_token_ids[0],
        torch.tensor([3, 4]),
    )
    assert result.finish_reasons == ("stop_token",)


def test_generate_stops_on_stop_sequence() -> None:
    model = PredictableModel()
    generator = Generator(model)

    result = generator.generate(
        torch.tensor([[1]], dtype=torch.long),
        greedy_config(
            max_new_tokens=5,
            stop_sequences=((2, 3, 4),),
        ),
    )

    assert torch.equal(
        result.generated_token_ids[0],
        torch.tensor([2, 3, 4]),
    )
    assert result.finish_reasons == (
        "stop_sequence",
    )


def test_generate_respects_no_repeat_ngram() -> None:
    model = PredictableModel()
    generator = Generator(model)

    result = generator.generate(
        torch.tensor([[1, 2, 1]], dtype=torch.long),
        greedy_config(
            max_new_tokens=1,
            no_repeat_ngram_size=2,
        ),
    )

    assert result.generated_token_ids[0].item() != 2


def test_batch_items_can_stop_at_different_times() -> None:
    model = PredictableModel()
    generator = Generator(model)
    prompt = torch.tensor(
        [[1], [4]],
        dtype=torch.long,
    )

    result = generator.generate(
        prompt,
        greedy_config(
            max_new_tokens=4,
            stop_token_ids=(3, 7),
        ),
    )

    assert torch.equal(
        result.generated_token_ids[0],
        torch.tensor([2, 3]),
    )
    assert torch.equal(
        result.generated_token_ids[1],
        torch.tensor([5, 6, 7]),
    )
    assert result.finish_reasons == (
        "stop_token",
        "stop_token",
    )


def test_generate_crops_context_to_model_limit() -> None:
    model = PredictableModel(
        maximum_sequence_length=3
    )
    generator = Generator(model)

    result = generator.generate(
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        greedy_config(max_new_tokens=4),
    )

    assert result.generated_lengths == (4,)
    assert model.seen_sequence_lengths == [3, 3, 3, 3]


def test_generate_restores_training_mode() -> None:
    model = PredictableModel()
    model.train()
    generator = Generator(model)

    generator.generate(
        torch.tensor([[1]], dtype=torch.long),
        greedy_config(max_new_tokens=1),
    )

    assert model.training is True


def test_generate_preserves_eval_mode() -> None:
    model = PredictableModel()
    model.eval()
    generator = Generator(model)

    generator.generate(
        torch.tensor([[1]], dtype=torch.long),
        greedy_config(max_new_tokens=1),
    )

    assert model.training is False


def test_seeded_generation_is_reproducible() -> None:
    model = UniformModel()
    generator = Generator(model)
    prompt = torch.tensor([[1]], dtype=torch.long)
    configuration = GenerationConfig(
        max_new_tokens=8,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        random_seed=42,
    )

    first = generator.generate(
        prompt,
        configuration,
    )
    second = generator.generate(
        prompt,
        configuration,
    )

    assert torch.equal(
        first.generated_token_ids[0],
        second.generated_token_ids[0],
    )


def test_generate_rejects_invalid_model_output_shape() -> None:
    generator = Generator(InvalidShapeModel())

    with pytest.raises(ValueError):
        generator.generate(
            torch.tensor([[1]], dtype=torch.long),
            greedy_config(max_new_tokens=1),
        )


def test_stream_yields_generation_steps() -> None:
    model = PredictableModel()
    generator = Generator(model)

    steps = list(
        generator.stream(
            torch.tensor([[1]], dtype=torch.long),
            greedy_config(max_new_tokens=3),
        )
    )

    assert all(
        isinstance(step, GenerationStep)
        for step in steps
    )
    assert [step.token_id for step in steps] == [
        2,
        3,
        4,
    ]
    assert steps[-1].is_finished is True
    assert (
        steps[-1].finish_reason
        == "maximum_new_tokens"
    )


def test_stream_stops_on_stop_token() -> None:
    generator = Generator(PredictableModel())

    steps = list(
        generator.stream(
            torch.tensor([[1]], dtype=torch.long),
            greedy_config(
                max_new_tokens=5,
                stop_token_ids=(3,),
            ),
        )
    )

    assert [step.token_id for step in steps] == [2, 3]
    assert steps[-1].finish_reason == "stop_token"


def test_stream_rejects_batch_generation() -> None:
    generator = Generator(PredictableModel())

    with pytest.raises(ValueError):
        list(
            generator.stream(
                torch.tensor(
                    [[1], [2]],
                    dtype=torch.long,
                ),
                greedy_config(),
            )
        )


@pytest.mark.parametrize(
    "token_ids,exception_type",
    [
        (torch.tensor([1, 2]), ValueError),
        (torch.tensor([[1.0]]), TypeError),
        (torch.empty((1, 0), dtype=torch.long), ValueError),
        (torch.tensor([[-1]], dtype=torch.long), ValueError),
        (torch.tensor([[8]], dtype=torch.long), ValueError),
        (torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long), ValueError),
    ],
)
def test_generate_rejects_invalid_prompts(
    token_ids: torch.Tensor,
    exception_type: type[Exception],
) -> None:
    generator = Generator(PredictableModel())

    with pytest.raises(exception_type):
        generator.generate(
            token_ids,
            greedy_config(max_new_tokens=1),
        )


@pytest.mark.parametrize(
    "configuration,exception_type",
    [
        (GenerationConfig(max_new_tokens=0), ValueError),
        (GenerationConfig(temperature=0.0), ValueError),
        (GenerationConfig(top_k=-1), ValueError),
        (GenerationConfig(top_p=0.0), ValueError),
        (GenerationConfig(top_p=1.1), ValueError),
        (GenerationConfig(repetition_penalty=0.0), ValueError),
        (GenerationConfig(no_repeat_ngram_size=-1), ValueError),
        (GenerationConfig(random_seed=-1), ValueError),
        (GenerationConfig(stop_token_ids=(8,)), ValueError),
        (GenerationConfig(stop_token_ids=(1, 1)), ValueError),
        (GenerationConfig(stop_sequences=((),)), ValueError),
        (GenerationConfig(stop_sequences=((1, 8),)), ValueError),
        (GenerationConfig(stop_sequences=((1,), (1,))), ValueError),
    ],
)
def test_generation_config_rejects_invalid_values(
    configuration: GenerationConfig,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        configuration.validate(vocabulary_size=8)
