from pathlib import Path

import pytest
import torch

from src.dataset import TokenDataset


@pytest.fixture
def token_file(
    tmp_path: Path,
):

    tokens = torch.arange(
        100,
        dtype=torch.long,
    )

    path = tmp_path / "tokens.pt"

    torch.save(
        tokens,
        path,
    )

    return path


def test_length(
    token_file,
):

    dataset = TokenDataset(
        token_file,
        sequence_length=8,
    )

    assert len(dataset) == 92


def test_shapes(
    token_file,
):

    dataset = TokenDataset(
        token_file,
        sequence_length=8,
    )

    x, y = dataset[0]

    assert x.shape == (8,)
    assert y.shape == (8,)


def test_shift(
    token_file,
):

    dataset = TokenDataset(
        token_file,
        sequence_length=8,
    )

    x, y = dataset[0]

    assert torch.equal(
        y[:-1],
        x[1:],
    )


def test_dtype(
    token_file,
):

    dataset = TokenDataset(
        token_file,
        sequence_length=8,
    )

    x, y = dataset[0]

    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_invalid_shape(
    tmp_path,
):

    tensor = torch.zeros(
        (10, 10)
    )

    path = tmp_path / "bad.pt"

    torch.save(
        tensor,
        path,
    )

    with pytest.raises(
        ValueError,
    ):
        TokenDataset(
            path,
            8,
        )


def test_small_corpus(
    tmp_path,
):

    tensor = torch.arange(
        5
    )

    path = tmp_path / "small.pt"

    torch.save(
        tensor,
        path,
    )

    with pytest.raises(
        ValueError,
    ):
        TokenDataset(
            path,
            8,
        )


def test_last_item(
    token_file,
):

    dataset = TokenDataset(
        token_file,
        8,
    )

    x, y = dataset[
        len(dataset) - 1
    ]

    assert x[-1] == 98
    assert y[-1] == 99

def test_zero_maximum_examples_uses_full_dataset(token_file):
    dataset = TokenDataset(
        token_file,
        sequence_length=8,
        maximum_examples=0,
    )

    assert len(dataset) == 92


def test_negative_maximum_examples_is_rejected(token_file):
    with pytest.raises(ValueError, match="cannot be negative"):
        TokenDataset(
            token_file,
            sequence_length=8,
            maximum_examples=-1,
        )
