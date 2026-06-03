import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field

class SpectralDecomposition(nn.Module):
    def __init__(self,
                 n_timesteps: int,
                 n_energies: int,
                 n_species: int,
                 C_guess: torch.Tensor = None,
                 ST_guess: torch.Tensor = None,
                 C_min: float = 1E-12,
                 ST_min: float = 1E-12,
                 c_requires_grad: bool = True,
                 s_requires_grad: bool = True,
                 ) -> None:
        super().__init__()

        self.dtype = torch.get_default_dtype()

        if C_guess is not None:
            C0 = torch.as_tensor(C_guess).clamp_min(C_min)
            c0 = torch.log(C0)
            self.dtype = c0.dtype
        if ST_guess is not None:
            ST0 = torch.as_tensor(ST_guess).clamp_min(ST_min)
            st0 = torch.log(ST0)
            self.dtype = st0.dtype

        if C_guess is None:
            c0 = torch.zeros((n_timesteps, n_species), dtype=self.dtype)
        if ST_guess is None:
            st0 = torch.zeros((n_species, n_energies), dtype=self.dtype)

        self.c = nn.Parameter(c0, requires_grad=c_requires_grad)
        self.st = nn.Parameter(st0, requires_grad=s_requires_grad)

        return

    def concentrations(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.c, dim=1)

    def spectra(self) -> torch.Tensor:
        return torch.exp(self.st)

    def forward(self) -> torch.Tensor:
        C = self.concentrations()
        ST = self.spectra()
        return C @ ST

    def reconstruction_mse(self,
            C: torch.Tensor,
            ST: torch.Tensor,
            D: torch.Tensor,
            ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(C @ ST, D)

    def reconstruction_mae(self,
            C: torch.Tensor,
            ST: torch.Tensor,
            D: torch.Tensor,
            ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(C @ ST, D)

    def total_loss(self,
            C: torch.Tensor,
            ST: torch.Tensor,
            D: torch.Tensor,
            constraints: list[callable] = [],
            ) -> torch.Tensor:
        loss = self.reconstruction_mse(C, ST, D)
        for constraint in constraints:
            loss += constraint(C, ST, D)
        return loss

    def train(self,
              D: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler,
              constraints: list[callable],
              max_epoches: int = 5000,
              verbose: int = 50,
              gtol: float = 1e-7,
              xtol: float = 1e-2,
              ) -> None:

        D = torch.as_tensor(D)

        def closure():
            optimizer.zero_grad()
            C = self.concentrations()
            ST = self.spectra()
            loss = self.total_loss(C, ST, D, constraints)
            loss.backward()
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad = param.grad.contiguous()
            return loss

        for epoch in range(max_epoches):
            C_pre = self.concentrations()
            ST_pre = self.spectra()
            loss = closure()
            gmax = max([param.grad.abs().max()
                        for group in optimizer.param_groups
                        for param in group['params']])
            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                optimizer.step()
            C_post = self.concentrations()
            ST_post = self.spectra()
            dCmax = (C_post - C_pre).abs().max()
            dSTmax = (ST_post - ST_pre).abs().max()
            lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()
            if epoch % verbose == 0:
                print(f'Epoch {epoch} Loss = {loss.item():.6g} Gmax = {gmax:.6g} LR = {lr:.6g} dCmax = {dCmax:.6g} dSTmax = {dSTmax:.6g}')
            if gmax < gtol and max(dCmax, dSTmax) < xtol * lr:
                break
            elif max(dCmax, dSTmax) == 0.0:
                break
            else:
                continue

        optimizer.zero_grad()
        C = self.concentrations()
        ST = self.spectra()
        loss = self.total_loss(C, ST, D, constraints)
        loss.backward()
        gmax = max([param.grad.abs().max()
                    for group in optimizer.param_groups
                    for param in group['params']])
        recon_mse = self.reconstruction_mse(C, ST, D)
        recon_mae = self.reconstruction_mae(C, ST, D)
        print(f'Final Epoch {epoch}')
        print(f'Loss = {loss.item():.6g} Gmax = {gmax:.6g} LR = {lr:.6g} dCmax = {dCmax:.6g} dSTmax = {dSTmax:.6g}')
        print(f'Reconstruction MSE {recon_mse:.6g} MAE {recon_mae:.6g}')

        return

class MultipleSpectraDecomposition(nn.Module):
    def __init__(self,
                 models: list[SpectralDecomposition]
                 ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        return

    def concentrations(self) -> list[torch.Tensor]:
        return [model.concentrations() for model in self.models]

    def spectra(self) -> list[torch.Tensor]:
        return [model.spectra() for model in self.models]

    def forward(self) -> list[torch.Tensor]:
        return [model.forward() for model in self.models]

    def reconstruction_mse(self,
            C: list[torch.Tensor],
            ST: list[torch.Tensor],
            D: list[torch.Tensor],
            ) -> torch.Tensor:
        return [torch.nn.functional.mse_loss(c @ st, d) for c, st, d in zip(C, ST, D)]

    def reconstruction_mae(self,
            C: list[torch.Tensor],
            ST: list[torch.Tensor],
            D: list[torch.Tensor],
            ) -> torch.Tensor:
        return [torch.nn.functional.l1_loss(c @ st, d) for c, st, d in zip(C, ST, D)]

    def total_loss(self,
            C: list[torch.Tensor],
            ST: list[torch.Tensor],
            D: list[torch.Tensor],
            constraints: list[callable] = [],
            ) -> torch.Tensor:
        loss = 0.0
        for mse in self.reconstruction_mse(C, ST, D):
            loss += mse
        for constraint in constraints:
            for cons in constraint(C, ST, D):
                loss += cons
        return loss

    def train(self,
              D: list[torch.Tensor],
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler,
              constraints: list[callable],
              max_epoches: int = 5000,
              verbose: int = 50,
              gtol: float = 1e-7,
              xtol: float = 1e-2,
              ) -> None:

        D = [torch.as_tensor(d) for d in D]

        def closure():
            optimizer.zero_grad()
            C = self.concentrations()
            ST = self.spectra()
            loss = self.total_loss(C, ST, D, constraints)
            loss.backward()
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad = param.grad.contiguous()
            return loss

        for epoch in range(max_epoches):
            C_pre = self.concentrations()
            ST_pre = self.spectra()
            loss = closure()
            gmax = max([param.grad.abs().max()
                        for group in optimizer.param_groups
                        for param in group['params']])
            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                optimizer.step()
            C_post = self.concentrations()
            ST_post = self.spectra()
            dCmax = max([(c_post - c_pre).abs().max() for c_post, c_pre in zip(C_pre, C_post)])
            dSTmax = max([(st_post - st_pre).abs().max() for st_post, st_pre in zip(ST_pre, ST_post)])
            lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()
            if epoch % verbose == 0:
                print(f'Epoch {epoch} Loss = {loss.item():.6g} Gmax = {gmax:.6g} LR = {lr:.6g} dCmax = {dCmax:.6g} dSTmax = {dSTmax:.6g}')
            if gmax < gtol and max(dCmax, dSTmax) < xtol * lr:
                break
            elif max(dCmax, dSTmax) == 0.0:
                break
            else:
                continue

        optimizer.zero_grad()
        C = self.concentrations()
        ST = self.spectra()
        loss = self.total_loss(C, ST, D, constraints)
        loss.backward()
        gmax = max([param.grad.abs().max()
                    for group in optimizer.param_groups
                    for param in group['params']])
        recon_mse = self.reconstruction_mse(C, ST, D)
        recon_mae = self.reconstruction_mae(C, ST, D)
        print(f'Final Epoch {epoch}')
        print(f'Loss = {loss.item():.6g} Gmax = {gmax:.6g} LR = {lr:.6g} dCmax = {dCmax:.6g} dSTmax = {dSTmax:.6g}')
        for i, (mse, mae) in enumerate(zip(recon_mse, recon_mae)):
            print(f'Model {i} Reconstruction MSE {mse:.6g} MAE {mae:.6g}')

        return

