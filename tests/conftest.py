"""Small, deterministic mixtures shared by the scientific tests."""
import pytest
import torch


@pytest.fixture
def mixture():
    dtype = torch.float64
    c = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]], dtype=dtype)
    st = torch.tensor([[1.0, 1.5, 2.0, 1.5, 1.0], [2.0, 1.0, 0.5, 1.0, 2.0]], dtype=dtype)
    return c, st, c @ st
