# Differentiable Multivariate Curve Resolution (DMCR)

DMCR is a lightweight PyTorch implementation of **differentiable multivariate curve resolution (MCR)** for spectral deconvolution problems such as X-ray absorption spectroscopy (XAS).

The method factorizes a spectral sequence matrix into:

* a concentration matrix describing the abundance of each component,
* a set of component spectra describing the spectral signatures of those components.

Unlike traditional alternating-least-squares approaches, DMCR formulates MCR as a differentiable optimization problem and trains the factorization directly using gradient-based optimizers.

This repository provides:

* a reusable DMCR library,
* a command-line workflow for spectral deconvolution,
* YAML-based configuration files for reproducible analyses,
* a Cu/Ga XAS benchmark dataset,
* complete example workflows used in the accompanying publication.

---

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate dmcr
```

Install the package:

```bash
pip install -e .
```

This installs the command-line interface:

```text
deconv_spec
```

---

## Repository Structure

```text
dmcr/
├── mcr.py                    # Core DMCR model and constraints
└── deconv_spec.py            # Command-line workflow

examples/
└── CuGaXAS/
    ├── cgo_cu.yaml           # CuGaO2 Cu K-edge deconvolution example
    ├── cgo_cu_c_guess.npy    # Initial guess concentrations
    ├── cgo_cu_c_opt.npy      # Precomputed Cu concentrations
    │                         # (used by the Ga workflow)
    │
    ├── cgo_ga.yaml           # CuGaO2 Ga K-edge deconvolution example
    ├── cgo_ga_c_guess.npy    # Initial guess concentrations
    │
    ├── mix_cu.yaml           # Cu2O + Ga2O3 physical mixture Cu K-edge deconvolution example
    ├── mix_cu_c_guess.npy    # Initial guess concentrations
    │
    └── *.dat                 # Example XAS spectra

environment.yml              # Conda environment
pyproject.toml               # Package metadata and installation
README.md                    # Project documentation
LICENSE                      # License information
```

---

## Method Overview

Given a spectral sequence matrix

$$
D \in \mathbb{R}^{N \times M},
$$

DMCR seeks a factorization

$$
D \approx C S^T,
$$

where

* (C) is the concentration matrix,
* (S^T) contains the component spectra.

The implementation enforces physically meaningful constraints through differentiable parameterizations:

* concentrations are normalized with a softmax transformation,
* component spectra are constrained to remain positive,
* optional regularization terms can enforce smoothness, anchoring, or rank-order relationships.

Optimization can be performed using standard PyTorch optimizers. Currently, LBFGS is selected in `deconv_spec`.

---

## Configuration

All run-specific parameters are stored in YAML files.

Typical settings include:

* input spectrum files,
* energy window selection,
* model dimensions,
* initial guesses,
* optimization settings,
* regularization parameters.

A minimal example:

```yaml
filepaths:
  - spec0000.dat
  - spec0001.dat
  - spec0002.dat

model:
  C_guess: C_guess.npy
  ST_guess:
    - ST_guess_0.dat
    - ST_guess_1.dat
```

---

## Example Dataset

The repository includes a Cu/Ga XAS benchmark dataset demonstrating several common DMCR workflows.

### CuGaO<sub>2</sub> Cu K-edge Deconvolution

Recover concentration profiles and component spectra from a CuGaO<sub>2</sub> Cu K-edge XAS series:

```bash
deconv_spec examples/CuGaXAS/cgo_cu.yaml
```

### CuGaO<sub>2</sub> Ga K-edge Deconvolution

Perform a corresponding deconvolution on the CuGaO<sub>2</sub> Ga K-edge dataset:

```bash
deconv_spec examples/CuGaXAS/cgo_ga.yaml
```

This workflow uses a Kendall rank correlation constraint to encourage consistency with the Cu concentration evolution. A precomputed Cu concentration profile (`cgo_cu_c_opt.npy`) is distributed with the repository so that the example can be run directly.

### Cu<sub>2</sub>O + Ga<sub>2</sub>O<sub>3</sub> Physical Mixture Cu K-edge Deconvolution

Deconvolve Cu K-edge spectra for the Cu<sub>2</sub>O + Ga<sub>2</sub>O<sub>3</sub> physical mixture:

```bash
deconv_spec examples/CuGaXAS/mix_cu.yaml
```

---

## Running a Deconvolution

Execute a deconvolution using a YAML configuration file:

```bash
deconv_spec config.yaml
```

Optionally specify separate input and output directories:

```bash
deconv_spec config.yaml \
    --inpdir ./inputs \
    --outdir ./outputs
```

The default behavior is:

* input files are resolved relative to the configuration file location,
* output files are written to the current working directory.

---

## Outputs

After optimization, DMCR writes:

```text
C_opt.npy
ST_opt.npy
ST_opt_0.dat
ST_opt_1.dat
...
```

where

* `C_opt.npy` contains optimized concentrations,
* `ST_opt.npy` contains optimized component spectra,
* `ST_opt_k.dat` contains each recovered component spectrum in two-column ASCII format.

---

## Reproducing the Included Examples

From the repository root:

```bash
deconv_spec examples/CuGaXAS/cgo_cu.yaml

deconv_spec examples/CuGaXAS/cgo_ga.yaml

deconv_spec examples/CuGaXAS/mix_cu.yaml
```

All required spectra, configuration files, and initialization data are distributed with the repository.

---

## Citation

If you use this software in published work, please cite the accompanying publication.

```text
Citation information will be added upon publication.
```

---

## Acknowledgement

This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704.

---

## Disclaimer

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
