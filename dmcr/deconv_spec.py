"""
Examples:

  python deconv_spec.py ../examples/CuGaXAS/cgo_cu.yaml
  python deconv_spec.py ../examples/CuGaXAS/cgo_cu.yaml
  python deconv_spec.py ../examples/CuGaXAS/mix_cu.yaml -i ../examples/CuGaXAS -o .

Notes:
  * The config file path is required.
  * By default, input files are read relative to the config file path.
  * Results are written to the current directory unless --outdir is specified.
  * In the CuGaO2 workflow, `cgo_cu.yaml` should be run before `cgo_ga.yaml`,
    because Ga uses the Cu concentrations as a reference for the Kendall rank
    correlation coefficient.
"""

import argparse
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dacite import from_dict
from dataclasses import dataclass, field
from larch.io.columnfile import read_ascii, write_ascii

from dmcr.mcr import (
    AnchorLoss,
    TemporalSmoothnessLoss,
    SpectralSmoothnessLoss,
    KendallRankCorrCoeff,
    DifferentiableMCR,
)

# ----------------------------------------------------------------------------
# Configuration schema
# ----------------------------------------------------------------------------
# These dataclasses define the structure expected in the YAML config files.
# They keep dataset-specific choices, model dimensions, and optional
# regularization terms separate from the core optimization code.

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

def _existing_file(directory, filename, description):
    """Resolve a configured input and report missing files clearly."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"{description} must be a nonempty filename")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _check_grid(grid, reference, filename):
    """All spectra must use the same energy axis for matrix factorization."""
    if grid.size == 0:
        raise ValueError(f"No energy points in selected range: {filename}")
    if reference is not None and (grid.shape != reference.shape or
                                  not np.allclose(grid, reference, rtol=1e-7, atol=1e-8)):
        raise ValueError(f"Energy grid differs from first input spectrum: {filename}")


def main():

    # ------------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------------
    # The config file supplies all run-specific parameters, while the optional
    # input/output directories control file resolution and export location.
    parser = argparse.ArgumentParser(
        description="Differentiable Multivariate Curve Resolution on XAS Spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("-i", "--inpdir", type=str, default=None,
        help="Input directory with data files (default = same as config YAML)")
    parser.add_argument("-o", "--outdir", type=str, default=".",
        help="Output directory for exports (default = current directory)")
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("rt", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Configuration must contain a YAML mapping")
        config = from_dict(data_class=Config, data=data)
        if not config.filepaths:
            raise ValueError("Configuration 'filepaths' must contain at least one spectrum")
        inpdir = (Path(args.inpdir).expanduser().resolve() if args.inpdir is not None
                  else config_path.parent)
        if not inpdir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {inpdir}")
        outdir = Path(args.outdir).expanduser().resolve()
        if outdir.exists() and not outdir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {outdir}")
        # Validate all file references before starting a potentially lengthy fit.
        for name in config.filepaths:
            _existing_file(inpdir, name, "Spectrum")
        if config.model.C_guess is not None:
            _existing_file(inpdir, config.model.C_guess, "Concentration guess")
        if config.model.ST_guess is not None:
            for name in config.model.ST_guess:
                _existing_file(inpdir, name, "Spectral guess")
        if config.kendall_rank.weight > 0:
            _existing_file(inpdir, config.kendall_rank.C_ref, "Kendall reference")
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    # ------------------------------------------------------------------------
    # Load spectra
    # ------------------------------------------------------------------------
    # Each file is cropped to the requested energy window, then stacked into a
    # common array. The energy grid is retained so the recovered spectra can be
    # written back on the same abscissa.
    emin = config.e0 + config.pre_edge
    emax = config.e0 + config.post_edge
    grids = []
    specs = []
    for name in config.filepaths:
        filepath = inpdir / name
        dat = read_ascii(filepath, labels="energy flat")
        select = (dat.energy > emin) & (dat.energy < emax)
        grid = np.asarray(dat.energy[select])
        _check_grid(grid, grids[0] if grids else None, filepath)
        grids.append(grid)
        specs.append(dat.flat[select])
    grids = np.array(grids)
    specs = np.array(specs)

    # ------------------------------------------------------------------------
    # Set up guess concentrations & spectra
    # ------------------------------------------------------------------------
    # Concentration guesses are stored as NumPy arrays, while spectral guesses
    # are read from ASCII files and cropped to the same energy window.
    if config.model.C_guess is None:
        C_guess = None
    else:
        filepath = inpdir / config.model.C_guess
        C_guess = np.load(filepath)
        C_guess = torch.tensor(C_guess)

    if config.model.ST_guess is None:
        ST_guess = None
    else:
        ST_guess = []
        for filepath in config.model.ST_guess:
            filepath = inpdir / filepath
            dat = read_ascii(filepath, labels="energy flat")
            select = (dat.energy > emin) & (dat.energy < emax)
            _check_grid(np.asarray(dat.energy[select]), grids[0], filepath)
            ST_guess.append(dat.flat[select])
        ST_guess = torch.tensor(ST_guess)

    # ------------------------------------------------------------------------
    # Set up model & optimizer
    # ------------------------------------------------------------------------
    # The model stores unconstrained latent variables internally and maps them
    # to positive concentrations/spectra during evaluation.
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

    # ------------------------------------------------------------------------
    # Set up constraints
    # ------------------------------------------------------------------------
    # Each optional term is only added when its weight is positive (or, in the
    # case of spectral smoothness, when weights are supplied).
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
        filepath = inpdir / config.kendall_rank.C_ref
        C_ref = np.load(filepath)
        C_ref = torch.tensor(C_ref)
        kendall_rank = KendallRankCorrCoeff(
            weight=config.kendall_rank.weight,
            C_ref=C_ref,
        )
        constraints.append(kendall_rank)

    # ------------------------------------------------------------------------
    # Run MCR
    # ------------------------------------------------------------------------
    # The measured spectra are converted to a tensor and passed into the
    # model's training loop together with the chosen constraints and stopping
    # criteria.
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

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------
    # The optimized factors are written as NumPy arrays, and each component
    # spectrum is also exported as a two-column ASCII file.
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "C_opt.npy", C_opt)
    np.save(outdir / "ST_opt.npy", ST_opt)
    for k, _ in enumerate(ST_opt):
        write_ascii(str(outdir / f"ST_opt_{k}.dat"), grids[0], ST_opt[k])

if __name__ == "__main__":
    exit(main())
