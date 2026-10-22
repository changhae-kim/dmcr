"""Repeatability on CPU with identical initialization and optimizer."""
import torch
from dmcr.mcr import DifferentiableMCR


def test_same_initialization_produces_same_fit(mixture, capsys):
    c, st, d = mixture
    outputs = []
    for _ in range(2):
        m = DifferentiableMCR(C_guess=c.clone(), ST_guess=(st * 1.05).clone())
        m.fit(d, torch.optim.SGD(m.parameters(), lr=0.01), None, max_epochs=4, verbose=100)
        outputs.append((m.concentrations().detach().clone(), m.spectra().detach().clone()))
    for a, b in zip(*outputs):
        assert torch.equal(a, b)


def test_fixed_seed_reproduces_random_initialization():
    torch.manual_seed(42)
    a = torch.rand((4, 2), dtype=torch.float64)
    torch.manual_seed(42)
    b = torch.rand((4, 2), dtype=torch.float64)
    assert torch.equal(a, b)
