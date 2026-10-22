"""Tests of the constraints actually exposed by dmcr.mcr."""
import torch
from dmcr.mcr import (DifferentiableMCR, AnchorLoss, TemporalSmoothnessLoss,
                      SpectralSmoothnessLoss, KendallRankCorrCoeff)


def test_simplex_and_positive_spectra(mixture):
    c, st, _ = mixture
    m = DifferentiableMCR(C_guess=c, ST_guess=st)
    assert torch.all(m.concentrations() >= 0)
    assert torch.allclose(m.concentrations().sum(dim=1), torch.ones(c.shape[0], dtype=c.dtype))
    assert torch.all(m.spectra() > 0)


def test_anchor_loss_zero_when_species_is_at_target(mixture):
    c, st, d = mixture
    c[:, 0] = 0.4
    assert torch.allclose(AnchorLoss(2.0, 0, 0.4)(c, st, d), torch.zeros((), dtype=c.dtype))


def test_temporal_smoothness_constant_and_break(mixture):
    c, st, d = mixture
    constant = c[:1].expand_as(c).clone()
    assert TemporalSmoothnessLoss(1.0)(constant, st, d).item() == 0
    assert TemporalSmoothnessLoss(1.0)(c, st, d).item() > 0
    # All adjacent differences are excluded when every boundary is a break.
    assert TemporalSmoothnessLoss(1.0, [0, 1, 2])(c, st, d).item() == 0


def test_spectral_smoothness_constant_and_varying(mixture):
    c, st, d = mixture
    assert SpectralSmoothnessLoss([1., 1.])(c, torch.ones_like(st), d).item() == 0
    assert SpectralSmoothnessLoss([1., 1.])(c, st, d).item() > 0
    assert SpectralSmoothnessLoss([0., 0.])(c, st, d).item() == 0


def test_kendall_loss_is_finite_and_order_sensitive(mixture):
    c, st, d = mixture
    loss = KendallRankCorrCoeff(1.0, c)
    matched = loss(c.clone(), st, d)
    reversed_order = loss(c.flip(0).clone(), st, d)
    assert torch.isfinite(matched) and torch.isfinite(reversed_order)
    assert matched < reversed_order


def test_combined_constraint_loss_is_additive(mixture):
    c, st, d = mixture
    m = DifferentiableMCR(C_guess=c, ST_guess=st)
    penalties = [AnchorLoss(0.1, 0, 0.3), TemporalSmoothnessLoss(0.2),
                 SpectralSmoothnessLoss([0.1, 0.1])]
    base = m.reconstruction_mse(c, st, d)
    expected = base + sum((p(c, st, d) for p in penalties), torch.zeros_like(base))
    assert torch.allclose(m.total_loss(c, st, d, penalties), expected)
