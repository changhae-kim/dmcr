import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dacite import from_dict
from dataclasses import dataclass, field
from larch.io.columnfile import read_ascii, write_ascii

from mcr import (
    AnchorLoss,
    TemporalSmoothnessLoss,
    SpectralSmoothnessLoss,
    KendallRankCorrCoeff,
    DifferentiableMCR,
)

@dataclass
class ModelParams:
    n_timesteps: int = None
    n_species: int = None
    n_energies: int = None
    C_guess: str = None
    ST_guess: list[str] = None
    C_min: float = 1e-3
    ST_min: float = 1e-3
    c_requires_grad: bool = True
    st_requires_grad: bool = True

@dataclass
class FitParams:
    max_epochs: int = 1000
    verbose: int = 10
    gtol: float = 1e-7
    xtol: float = 1e-2

@dataclass
class AnchorParams:
    i_species: int = None
    C0: float = None
    weight: float = 0.0

@dataclass
class TemporalSmoothnessParams:
    i_breaks: list[int] = None
    weight: float = 0.0

@dataclass
class SpectralSmoothnessParams:
    weights: list[float] = None

@dataclass
class KendallRankCorrParams:
    C_ref: str = None
    weight: float = 0.0

@dataclass
class Config:
    filepaths: list[str]
    e0: float = 0.0
    pre_edge: float = -np.inf
    post_edge: float = +np.inf
    model: ModelParams = field(default_factory=ModelParams)
    fit: FitParams = field(default_factory=FitParams)
    anchor: AnchorParams = field(default_factory=AnchorParams)
    temporal_smoothness: TemporalSmoothnessParams = field(default_factory=TemporalSmoothnessParams)
    spectral_smoothness: SpectralSmoothnessParams = field(default_factory=SpectralSmoothnessParams)
    kendall_rank: KendallRankCorrParams = field(default_factory=KendallRankCorrParams)

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Shared Differentiable MCR Driver")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("-i", "--inpdir", type=str, default=None,
        help="Input directory with data files (default = same as config YAML)")
    parser.add_argument("-o", "--outdir", type=str, default=".",
        help="Output directory for exports (default = current directory)")
    args = parser.parse_args()

    with open(args.config, "rt") as f:
        data = yaml.safe_load(f)
    config = from_dict(data_class=Config, data=data)
    if args.inpdir is not None:
        inpdir = args.inpdir
    else:
        inpdir = os.path.dirname(args.config)
    outdir = args.outdir

    # Load spectra
    emin = config.e0 + config.pre_edge
    emax = config.e0 + config.post_edge
    grids = []
    specs = []
    for filepath in config.filepaths:
        filepath = os.path.join(inpdir, filepath)
        dat = read_ascii(filepath, labels="energy flat")
        select = (dat.energy > emin) & (dat.energy < emax)
        grids.append(dat.energy[select])
        specs.append(dat.flat[select])
    grids = np.array(grids)
    specs = np.array(specs)

    # Set up guess concentrations & spectra
    if config.model.C_guess is None:
        C_guess = None
    else:
        filepath = os.path.join(inpdir, config.model.C_guess)
        C_guess = np.load(filepath)
        C_guess = torch.tensor(C_guess)

    if config.model.ST_guess is None:
        ST_guess = None
    else:
        ST_guess = []
        for filepath in config.model.ST_guess:
            filepath = os.path.join(inpdir, filepath)
            dat = read_ascii(filepath, labels="energy flat")
            select = (dat.energy > emin) & (dat.energy < emax)
            ST_guess.append(dat.flat[select])
        ST_guess = torch.tensor(ST_guess)

    # Set up model & optimizer
    model = DifferentiableMCR(
        n_timesteps=config.model.n_timesteps,
        n_species=config.model.n_species,
        n_energies=config.model.n_energies,
        C_guess=C_guess,
        ST_guess=ST_guess,
        C_min=config.model.C_min,
        ST_min=config.model.ST_min,
        c_requires_grad=config.model.c_requires_grad,
        st_requires_grad=config.model.st_requires_grad,
    )
    optimizer = optim.LBFGS(model.parameters(), history_size=10000)
    scheduler = None

    # Set up constraints
    constraints = []
    if config.anchor.weight > 0.0:
        anchor = AnchorLoss(
            weight=config.anchor.weight,
            i_species=config.anchor.i_species,
            C0=config.anchor.C0,
        )
        constraints.append(anchor)
    if config.temporal_smoothness.weight > 0.0:
        temporal_smoothness = TemporalSmoothnessLoss(
            weight=config.temporal_smoothness.weight,
            i_breaks=config.temporal_smoothness.i_breaks,
        )
        constraints.append(temporal_smoothness)
    if config.spectral_smoothness.weights is not None:
        spectral_smoothness = SpectralSmoothnessLoss(
            weights=config.spectral_smoothness.weights,
        )
        constraints.append(spectral_smoothness)
    if config.kendall_rank.weight > 0.0:
        filepath = os.path.join(inpdir, config.kendall_rank.C_ref)
        C_ref = np.load(filepath)
        C_ref = torch.tensor(C_ref)
        kendall_rank = KendallRankCorrCoeff(
            weight=config.kendall_rank.weight,
            C_ref=C_ref,
        )
        constraints.append(kendall_rank)

    # Run MCR
    D = torch.tensor(specs)
    model.fit(
        D,
        optimizer,
        scheduler,
        constraints=constraints,
        max_epochs=config.fit.max_epochs,
        verbose=config.fit.verbose,
        gtol=config.fit.gtol,
        xtol=config.fit.xtol,
    )
    C_opt = model.concentrations().detach().numpy()
    ST_opt = model.spectra().detach().numpy()

    # Save results
    np.save("C_opt.npy", C_opt)
    np.save("ST_opt.npy", ST_opt)
    for k, _ in enumerate(ST_opt):
        write_ascii(f"ST_opt_{k}.dat", grids[0], ST_opt[k])

if __name__ == "__main__":
    exit(main())
