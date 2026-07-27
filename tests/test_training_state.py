import pytest

from src.training_state import TrainingState


def test_defaults():
    state = TrainingState()

    assert state.epoch == 0
    assert state.global_step == 0
    assert state.best_validation_loss == float("inf")


def test_increment_step():
    state = TrainingState()

    state.increment_step(
        tokens=512,
        examples=8,
    )

    assert state.global_step == 1
    assert state.tokens_processed == 512
    assert state.examples_processed == 8


def test_increment_epoch():
    state = TrainingState()

    state.increment_epoch()

    assert state.epoch == 1


def test_best_validation_loss():
    state = TrainingState()

    state.update_validation_loss(4.0)
    state.update_validation_loss(5.0)
    state.update_validation_loss(3.0)

    assert state.best_validation_loss == 3.0


def test_serialization():
    state = TrainingState()

    state.increment_epoch()
    state.increment_step(
        100,
        4,
    )

    restored = (
        TrainingState.from_dict(
            state.to_dict()
        )
    )

    assert restored == state


def test_negative_learning_rate():
    state = TrainingState()

    with pytest.raises(ValueError):
        state.update_learning_rate(-1)


def test_negative_training_loss():
    state = TrainingState()

    with pytest.raises(ValueError):
        state.update_training_loss(-1)


def test_negative_validation_loss():
    state = TrainingState()

    with pytest.raises(ValueError):
        state.update_validation_loss(-1)


def test_negative_tokens():
    state = TrainingState()

    with pytest.raises(ValueError):
        state.increment_step(
            -1,
            1,
        )


def test_negative_examples():
    state = TrainingState()

    with pytest.raises(ValueError):
        state.increment_step(
            1,
            -1,
        )