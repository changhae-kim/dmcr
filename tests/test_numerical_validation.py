import pytest
import torch

from dmcr.mcr import DifferentiableMCR


def model():
    return DifferentiableMCR(n_timesteps=3, n_species=2, n_energies=4)


def test_total_loss_default_constraints():
    m = model()
    c, st = m.concentrations(), m.spectra()
    d = c @ st
    assert torch.allclose(m.total_loss(c, st, d), torch.zeros(()))


@pytest.mark.parametrize('shape', [(3, 5), (2, 4), (3,)])
def test_rejects_wrong_data_shape(shape):
    m = model()
    with pytest.raises(ValueError, match='D must have shape'):
        m.fit(torch.zeros(shape), torch.optim.SGD(m.parameters(), lr=0.01), None, max_epochs=1)


def test_rejects_nonfinite_data():
    m = model()
    d = torch.ones(3, 4)
    d[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        m.fit(d, torch.optim.SGD(m.parameters(), lr=0.01), None, max_epochs=1)


def test_rejects_incompatible_guesses():
    with pytest.raises(ValueError, match='Incompatible'):
        DifferentiableMCR(C_guess=torch.ones(3, 2), ST_guess=torch.ones(3, 4))


def test_rejects_nonfinite_loss():
    m = model()
    d = torch.ones(3, 4)
    def bad_constraint(c, st, data):
        return torch.tensor(float('nan'))
    with pytest.raises(FloatingPointError, match='loss'):
        m.fit(d, torch.optim.SGD(m.parameters(), lr=0.01), None, constraints=[bad_constraint], max_epochs=1)


def test_rejects_nonfinite_gradient():
    m = model()
    d = torch.ones(3, 4)
    def bad_constraint(c, st, data):
        return c.sum() * torch.tensor(float('nan'))
    with pytest.raises(FloatingPointError):
        m.fit(d, torch.optim.SGD(m.parameters(), lr=0.01), None, constraints=[bad_constraint], max_epochs=1)


def test_finite_one_step_fit():
    m = model()
    m.fit(torch.ones(3, 4), torch.optim.SGD(m.parameters(), lr=0.01), None, max_epochs=1)
    assert torch.isfinite(m.forward()).all()
