"""Scientific invariants of the differentiable factorization."""
import torch
from dmcr.mcr import DifferentiableMCR


def test_exact_reconstruction_from_known_factors(mixture):
    c, st, d = mixture
    m = DifferentiableMCR(C_guess=c, ST_guess=st)
    assert m.forward().shape == d.shape
    assert torch.allclose(m.forward(), d, rtol=0, atol=1e-12)
    assert m.reconstruction_mse(m.concentrations(), m.spectra(), d) < 1e-24


def test_one_step_optimization_reduces_loss(mixture, capsys):
    c, st, d = mixture
    m = DifferentiableMCR(C_guess=c, ST_guess=st * 1.1)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    before = torch.mean((m.forward() - d) ** 2).item()
    m.fit(d, opt, None, max_epochs=1, verbose=100)
    after = torch.mean((m.forward() - d) ** 2).item()
    assert after < before


def test_reconstruction_mae_and_mse_are_zero_for_exact_data(mixture):
    c, st, d = mixture
    m = DifferentiableMCR(C_guess=c, ST_guess=st)
    assert m.reconstruction_mae(m.concentrations(), m.spectra(), d).item() < 1e-12
    assert m.reconstruction_mse(m.concentrations(), m.spectra(), d).item() < 1e-24
