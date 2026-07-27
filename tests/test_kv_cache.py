from dataclasses import replace

import pytest
import torch

from src.attention import CausalSelfAttention
from src.config import ModelConfig
from src.generation import GenerationConfig
from src.generation import Generator
from src.kv_cache import AttentionKVCache
from src.kv_cache import ModelKVCache
from src.model import GPTModel


def small_config(**changes) -> ModelConfig:
    base = ModelConfig(
        vocabulary_size=32,
        maximum_sequence_length=12,
        embedding_dimension=16,
        number_of_layers=2,
        number_of_attention_heads=4,
        feed_forward_dimension=32,
        dropout=0.0,
        batch_size=2,
        epochs=1,
        generation_max_tokens=4,
        generation_top_k=0,
        generation_top_p=1.0,
    )
    return replace(base, **changes)


def test_attention_cache_properties() -> None:
    cache = AttentionKVCache(
        key=torch.zeros(2, 4, 3, 5),
        value=torch.zeros(2, 4, 3, 5),
    )

    assert cache.batch_size == 2
    assert cache.number_of_heads == 4
    assert cache.sequence_length == 3
    assert cache.head_dimension == 5


def test_attention_cache_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        AttentionKVCache(
            key=torch.zeros(2, 4, 3, 5),
            value=torch.zeros(2, 4, 2, 5),
        )


def test_model_cache_rejects_layer_length_mismatch() -> None:
    first = AttentionKVCache(
        key=torch.zeros(1, 2, 2, 4),
        value=torch.zeros(1, 2, 2, 4),
    )
    second = AttentionKVCache(
        key=torch.zeros(1, 2, 3, 4),
        value=torch.zeros(1, 2, 3, 4),
    )

    with pytest.raises(ValueError):
        ModelKVCache(layers=(first, second))


def test_attention_cached_output_matches_full_output() -> None:
    torch.manual_seed(7)
    model_config = small_config(number_of_layers=1)
    attention = CausalSelfAttention(model_config)
    attention.eval()
    hidden_states = torch.randn(2, 6, 16)

    full_output = attention(hidden_states)
    prefix_output, cache = attention(
        hidden_states[:, :4],
        use_cache=True,
    )
    suffix_output, updated_cache = attention(
        hidden_states[:, 4:],
        cache=cache,
        use_cache=True,
    )

    assert prefix_output.shape == (2, 4, 16)
    assert updated_cache.sequence_length == 6
    assert torch.allclose(
        suffix_output,
        full_output[:, 4:],
        atol=1e-6,
        rtol=1e-5,
    )


def test_model_cached_logits_match_full_logits() -> None:
    torch.manual_seed(11)
    model_config = small_config()
    model = GPTModel(model_config)
    model.eval()
    token_ids = torch.randint(0, 32, (2, 7))

    full_logits = model(token_ids)
    prefix_logits, cache = model(
        token_ids[:, :5],
        use_cache=True,
    )
    suffix_logits, updated_cache = model(
        token_ids[:, 5:],
        cache=cache,
        use_cache=True,
    )

    assert prefix_logits.shape == (2, 5, 32)
    assert updated_cache.number_of_layers == 2
    assert updated_cache.sequence_length == 7
    assert torch.allclose(
        suffix_logits,
        full_logits[:, 5:],
        atol=1e-6,
        rtol=1e-5,
    )


def test_incremental_single_token_logits_match_full_logits() -> None:
    torch.manual_seed(19)
    model_config = small_config()
    model = GPTModel(model_config)
    model.eval()
    token_ids = torch.randint(0, 32, (1, 8))

    full_logits = model(token_ids)
    cache = None
    incremental_logits = []

    for index in range(token_ids.shape[1]):
        step_logits, cache = model(
            token_ids[:, index : index + 1],
            cache=cache,
            use_cache=True,
        )
        incremental_logits.append(step_logits)

    concatenated_logits = torch.cat(incremental_logits, dim=1)

    assert cache.sequence_length == token_ids.shape[1]
    assert torch.allclose(
        concatenated_logits,
        full_logits,
        atol=1e-6,
        rtol=1e-5,
    )


def test_model_rejects_cache_that_exceeds_context_window() -> None:
    model_config = small_config(maximum_sequence_length=4)
    model = GPTModel(model_config)
    token_ids = torch.randint(0, 32, (1, 4))
    _, cache = model(token_ids, use_cache=True)

    with pytest.raises(ValueError):
        model(
            torch.randint(0, 32, (1, 1)),
            cache=cache,
            use_cache=True,
        )


def test_generation_cached_and_uncached_greedy_match() -> None:
    torch.manual_seed(23)
    model_config = small_config(
        maximum_sequence_length=10,
        generation_max_tokens=5,
    )
    model = GPTModel(model_config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generator = Generator(model, model_config)

    uncached = generator.generate(
        prompt,
        GenerationConfig(
            max_new_tokens=5,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            use_kv_cache=False,
        ),
    )
    cached = generator.generate(
        prompt,
        GenerationConfig(
            max_new_tokens=5,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            use_kv_cache=True,
        ),
    )

    assert torch.equal(
        cached.generated_token_ids[0],
        uncached.generated_token_ids[0],
    )
    assert torch.allclose(
        cached.log_probabilities[0],
        uncached.log_probabilities[0],
        atol=1e-6,
        rtol=1e-5,
    )


def test_generation_cache_resets_at_context_limit() -> None:
    torch.manual_seed(29)
    model_config = small_config(
        maximum_sequence_length=4,
        generation_max_tokens=6,
    )
    model = GPTModel(model_config)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generator = Generator(model, model_config)

    result = generator.generate(
        prompt,
        GenerationConfig(
            max_new_tokens=6,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            use_kv_cache=True,
        ),
    )

    assert result.generated_lengths == (6,)
    assert result.finish_reasons == ("maximum_new_tokens",)


def test_generation_config_rejects_non_boolean_cache_setting() -> None:
    generation_config = GenerationConfig(use_kv_cache="yes")

    with pytest.raises(TypeError):
        generation_config.validate(vocabulary_size=32)
