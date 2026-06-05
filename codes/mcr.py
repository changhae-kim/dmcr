import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorLoss(nn.Module):
    """
    Constrain one species to a target concentration.

    This is a soft constraint used to anchor a chosen species at a known
    concentration level or to remove an otherwise arbitrary rotational
    ambiguity in the matrix factorization.
    """

    def __init__(
        self,
        weight: float,
        i_species: int,
        C0: float,
    ) -> None:
        super(AnchorLoss, self).__init__()
        self.weight = weight
        self.i = i_species
        self.C0 = C0
        return

    def forward(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        dC = C[:, self.i] - self.C0
        loss = self.weight * torch.mean(dC ** 2)
        return loss

class TemporalSmoothnessLoss(nn.Module):
    """
    Encourage concentrations to vary smoothly.

    This loss penalizes differences between adjacent time points. Optional break
    indices can be exempted so that known discontinuities do not contribute.
    """
    def __init__(
        self,
        weight: float,
        i_breaks: list[int] = None,
    ) -> None:
        super(TemporalSmoothnessLoss, self).__init__()
        self.weight = weight
        self.i = i_breaks
        return

    def forward(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        dC = C[1:, :] - C[:-1, :]
        if self.i is not None:
            dC[self.i, :] = 0.0
        loss = self.weight * torch.mean(dC ** 2)
        return loss

class SpectralSmoothnessLoss(nn.Module):
    """Encourage each component spectrum to be smooth."""

    def __init__(
        self,
        weights: list[float],
    ) -> None:
        super(SpectralSmoothnessLoss, self).__init__()
        self.weights = weights
        return

    def forward(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=C.dtype, device=C.device)
        for w, st in zip(self.weights, ST):
            loss += w * F.mse_loss(st[1:], st[:-1])
        return loss

class KendallRankCorrCoeff(nn.Module):
    """
    Encourage the concentration ordering to match a reference.

    The target is the pairwise sign comparison of a reference concentration
    matrix. The loss matches pairwise ordering in the concentrations.
    """

    def __init__(
        self,
        weight: float,
        C_ref: torch.Tensor,
    ) -> None:
        super(KendallRankCorrCoeff, self).__init__()
        self.weight = weight
        self.target = torch.sign(C_ref[:,None,:] - C_ref[None,:,:])
        return

    def forward(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        n_timesteps, n_species = C.size()
        pred = C[:,None,:] - C[None,:,:]
        product = pred * self.target
        same_sign = (product > 0.0)
        opp_sign = (product < 0.0)
        indices = torch.arange(n_species, dtype=torch.int64, device=C.device)
        for i in range(n_species):
            selector = (indices == i)
            n_same = max(torch.numel(product[same_sign & selector]), 1)
            n_opp = max(torch.numel(product[opp_sign & selector]), 1)
            product[same_sign & selector] *= n_opp / max(n_same, n_opp)
        loss = self.weight * (-torch.sum(product) / ((n_timesteps ** 2 - n_timesteps) * n_species))
        return loss

class DifferentiableMCR(nn.Module):
    """
    Differentiable multivariate curve resolution.

    This model stores unconstrained latent variables and maps them to physically
    meaningful quantities:
      - concentrations are normalized with a softmax so each row sums to one,
      - spectra are exponentiated so they remain strictly positive.
    """

    def __init__(
        self,
        n_timesteps: int = None,
        n_species: int = None,
        n_energies: int = None,
        C_guess: torch.Tensor = None,
        ST_guess: torch.Tensor = None,
        C_min: float = 1e-3,
        ST_min: float = 1e-3,
        c_requires_grad: bool = True,
        st_requires_grad: bool = True,
    ) -> None:
        super(DifferentiableMCR, self).__init__()

        assert (n_timesteps is not None and n_species is not None) or (C_guess is not None)
        assert (n_species is not None and n_energies is not None) or (ST_guess is not None)

        self.dtype = torch.get_default_dtype()
        self.device = torch.get_default_device()

        # If guess are provided, then initialize the latent variables with their logarithms.
        if C_guess is not None:
            C0 = C_guess.clamp_min(C_min)
            c0 = torch.log(C0)
            self.dtype = c0.dtype
            self.device = c0.device
        if ST_guess is not None:
            ST0 = ST_guess.clamp_min(ST_min)
            st0 = torch.log(ST0)
            self.dtype = st0.dtype
            self.device = st0.device

        # If no guesses are provided, then fall back to zero latent variables.
        if C_guess is None:
            c0 = torch.zeros((n_timesteps, n_species), dtype=self.dtype, device=self.device)
        if ST_guess is None:
            st0 = torch.zeros((n_species, n_energies), dtype=self.dtype, device=self.device)

        self.c = nn.Parameter(c0, requires_grad=c_requires_grad)
        self.st = nn.Parameter(st0, requires_grad=st_requires_grad)

        return

    def concentrations(self) -> torch.Tensor:
        return F.softmax(self.c, dim=1)

    def spectra(self) -> torch.Tensor:
        return torch.exp(self.st)

    def forward(self) -> torch.Tensor:
        C = self.concentrations()
        ST = self.spectra()
        return C @ ST

    def reconstruction_mse(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(C @ ST, D)

    def reconstruction_mae(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        return F.l1_loss(C @ ST, D)

    def total_loss(
        self,
        C: torch.Tensor,
        ST: torch.Tensor,
        D: torch.Tensor,
        constraints: list[callable] = [],
    ) -> torch.Tensor:
        loss = self.reconstruction_mse(C, ST, D)
        for constraint in constraints:
            loss += constraint(C, ST, D)
        return loss

    def fit(
        self,
        D: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        constraints: list[callable] = [],
        max_epochs: int = 1000,
        verbose: int = 10,
        gtol: float = 1e-7,
        xtol: float = 1e-2,
    ) -> None:
        """Optimize the latent variables against the data matrix."""

        def closure():
            optimizer.zero_grad()
            C = self.concentrations()
            ST = self.spectra()
            loss = self.total_loss(C, ST, D, constraints)
            loss.backward()
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad = param.grad.contiguous()
            return loss

        for epoch in range(max_epochs):

            # Save state before step
            C_pre = self.concentrations()
            ST_pre = self.spectra()
            loss = closure()
            gmax = max([
                param.grad.abs().max()
                for group in optimizer.param_groups
                for param in group["params"]
            ])

            # Optimization step
            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                optimizer.step()

            # Recompute state after step
            C_post = self.concentrations()
            ST_post = self.spectra()
            dCmax = (C_post - C_pre).abs().max()
            dSTmax = (ST_post - ST_pre).abs().max()
            lr = optimizer.param_groups[0]["lr"]

            # Update learning-rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()

            # Report progress
            if epoch % verbose == 0:
                print(f"Epoch {epoch} Loss = {loss.item():.6g} Gmax = {gmax:.6g} LR = {lr:.6g} dCmax = {dCmax:.6g} dSTmax = {dSTmax:.6g}")

            # Check convergence
            if gmax < gtol and max(dCmax, dSTmax) < xtol * lr:
                break
            elif max(dCmax, dSTmax) == 0.0:
                break
            else:
                continue

        # Final report
        optimizer.zero_grad()
        C = self.concentrations()
        ST = self.spectra()
        loss = self.total_loss(C, ST, D, constraints)
        loss.backward()
        gmax = max([
            param.grad.abs().max()
            for group in optimizer.param_groups
            for param in group["params"]
        ])
        recon_mse = self.reconstruction_mse(C, ST, D)
        recon_mae = self.reconstruction_mae(C, ST, D)
        print(f"Final Epoch {epoch}")
        print(f"Loss = {loss.item():.6g} Gmax = {gmax:.6g} LR = {lr:.6g} dCmax = {dCmax:.6g} dSTmax = {dSTmax:.6g}")
        print(f"Reconstruction MSE {recon_mse:.6g} MAE {recon_mae:.6g}")

        return
