from pathlib import Path

from src.config import ModelConfig
from src.model import GPTModel
from src.model_profiles import ModelProfile


def test_42m_profile_loads() -> None:
    profile = ModelProfile.load(Path("configs/models/gpt_42m.json"))
    assert profile.course_iteration == 2
    assert profile.cpu.batch_size == 1
    assert profile.gpu.precision == "bf16"
    assert profile.embedding_dimension == 512
    assert profile.layers == 8


def test_100m_profile_is_near_target_parameter_count() -> None:
    profile = ModelProfile.load(Path("configs/models/gpt_100m.json"))
    config = ModelConfig(
        vocabulary_size=32_768,
        maximum_sequence_length=profile.sequence_length,
        embedding_dimension=profile.embedding_dimension,
        number_of_layers=profile.layers,
        number_of_attention_heads=profile.attention_heads,
        feed_forward_dimension=profile.feed_forward_dimension,
        batch_size=profile.batch_size,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
    )
    model = GPTModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert 95_000_000 <= parameter_count <= 105_000_000
