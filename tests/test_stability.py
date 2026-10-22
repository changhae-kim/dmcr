"""Finite reconstruction for modest noise and overlapping spectra."""
import pytest
import torch
from dmcr.mcr import DifferentiableMCR


@pytest.mark.parametrize('noise', [0., 0.001, 0.01])
def test_noisy_data_remains_finite(mixture, noise, capsys):
    c, st, d = mixture
    torch.manual_seed(42)
    noisy = d + noise * torch.randn(d.shape, dtype=d.dtype)
    m = DifferentiableMCR(C_guess=c, ST_guess=st * 1.02)
    m.fit(noisy, torch.optim.SGD(m.parameters(), lr=0.01), None, max_epochs=3, verbose=100)
    assert torch.isfinite(m.forward()).all()
    assert torch.allclose(m.concentrations().sum(dim=1), torch.ones(c.shape[0], dtype=c.dtype))


def test_nearly_identical_spectra_do_not_create_nonfinite_values(mixture, capsys):
    c, st, _ = mixture
    st[1] = st[0] + 1e-5
    d = c @ st
    m = DifferentiableMCR(C_guess=c, ST_guess=st * 1.01)
    m.fit(d, torch.optim.SGD(m.parameters(), lr=0.01), None, max_epochs=3, verbose=100)
    assert torch.isfinite(m.forward()).all()
