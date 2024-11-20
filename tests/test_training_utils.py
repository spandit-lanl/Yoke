"""Tests for torch training utilities."""

import pytest
import torch
import torch.nn as nn
from collections.abc import Generator
from yoke.torch_training_utils import count_torch_params


class SimpleModel(nn.Module):
    """A simple test model for parameter counting."""

    def __init__(self) -> None:
        """Setup the model."""
        super().__init__()
        self.fc1 = nn.Linear(10, 20)  # 10*20 + 20 = 220 parameters
        self.fc2 = nn.Linear(20, 5)  # 20*5 + 5 = 105 parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward method for module."""
        return self.fc2(self.fc1(x))


@pytest.fixture
def simple_model() -> Generator[SimpleModel, None, None]:
    """Fixture for creating a simple model."""
    model = SimpleModel()
    yield model


def test_count_torch_params_trainable(simple_model: SimpleModel) -> None:
    """Test count_torch_params for trainable parameters."""
    # By default, all parameters are trainable
    assert count_torch_params(simple_model) == 325  # Total parameters = 220 + 105


def test_count_torch_params_non_trainable(simple_model: SimpleModel) -> None:
    """Test count_torch_params for non-trainable parameters."""
    # Freeze all parameters
    for param in simple_model.parameters():
        param.requires_grad = False
    assert count_torch_params(simple_model) == 0  # No trainable parameters
    assert count_torch_params(simple_model, trainable=False) == 325  # Total parameters


def test_count_torch_params_partial_trainable(simple_model: SimpleModel) -> None:
    """Test count_torch_params when only some parameters are trainable."""
    # Freeze only fc2 layer
    for param in simple_model.fc2.parameters():
        param.requires_grad = False
    assert count_torch_params(simple_model) == 220  # Only fc1 parameters are trainable
    assert count_torch_params(simple_model, trainable=False) == 325  # Total parameters
