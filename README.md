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

DMCR supports Python **3.13 and 3.14** (both tested). From the repository root, install DMCR and its runtime dependencies using pip:

```bash
python -m pip install .
```

For development, use an editable installation with the optional test dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/ -v
```

The supplied `environment.yml` is a snapshot of the validated development environment, **not a prerequisite** for pip installation. It contains platform-specific and GPU-related packages and may not solve on other systems. If you wish to reproduce that environment on a compatible machine, use `conda env create -f environment.yml` and activate the resulting environment before installing DMCR. A CUDA GPU is not required by the DMCR algorithm; PyTorch installation details depend on your platform.

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

environment.yml              # Validated Conda environment snapshot
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

* $C$ is the concentration matrix (one row per measured spectrum),
* $S^T$ contains the component spectra (one row per component).

The implementation enforces physically meaningful constraints through differentiable parameterizations:

* concentrations are normalized with a softmax transformation,
* component spectra are constrained to remain positive,
* optional regularization terms can enforce smoothness, anchoring, or rank-order relationships.

The core library accepts standard PyTorch optimizers; the `deconv_spec` command uses LBFGS. Concentration rows sum to one by construction, and recovered component spectra are strictly positive. These are model assumptions and may not suit every spectral preprocessing convention.

**Interpretation:** A low reconstruction error does not establish uniqueness or chemical identity of the recovered components. MCR can have ambiguous solutions; initial guesses, chosen components, and regularization affect interpretation. The provided examples are demonstrations, not universal convergence guarantees.

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

An illustrative configuration (file paths refer to files in the same directory as the YAML unless `--inpdir` is supplied):

```yaml
filepaths:
  - spec0000.dat
  - spec0001.dat
  - spec0002.dat
e0: 8979.0
pre_edge: -20.0
post_edge: 100.0
model:
  C_guess: C_guess.npy
  ST_guess:
    - ST_guess_0.dat
    - ST_guess_1.dat
fit:
  max_epochs: 1000
  verbose: 10
```

The example assumes three spectra with a common energy grid, a concentration guess shaped `(3, 2)`, and two initial component spectra sampled on the selected energy grid. Input ASCII spectra must have energy and intensity columns readable by `larch.io.columnfile.read_ascii`.

| YAML key | Meaning |
|---|---|
| `filepaths` | Input spectra, ordered by sample/time point |
| `e0`, `pre_edge`, `post_edge` | Select energies strictly between `e0 + pre_edge` and `e0 + post_edge` |
| `model.C_guess` | Optional NumPy concentration initialization |
| `model.ST_guess` | Optional list of initial component spectra files |
| `model.n_timesteps`, `model.n_species`, `model.n_energies` | Dimensions used when corresponding guesses are omitted |
| `model.C_min`, `model.ST_min` | Floors applied before logarithmic initialization |
| `model.c_requires_grad`, `model.st_requires_grad` | Control which factors are optimized |
| `fit.max_epochs`, `fit.verbose`, `fit.gtol`, `fit.xtol` | Optimization and stopping settings |
| `anchor.weight`, `anchor.i_species`, `anchor.C0` | Penalize deviation of one concentration profile from a specified level |
| `temporal_smoothness.weight`, `temporal_smoothness.i_breaks` | Penalize changes between adjacent concentrations, except at declared sequence breaks |
| `spectral_smoothness.weights` | Component-wise penalties on spectral roughness |
| `kendall_rank.weight`, `kendall_rank.C_ref` | Kendall rank-correlation penalty using a saved reference concentration array |

Consult the three complete configurations under `examples/CuGaXAS/` for working parameter combinations. Optional constraint terms are enabled by their configured weights.

---

## Example Dataset

The repository includes a Cu/Ga XAS benchmark dataset demonstrating several common DMCR workflows.

### CuGaO<sub>2</sub> Cu K-edge Deconvolution

Recover concentration profiles and component spectra from a CuGaO<sub>2</sub> Cu K-edge XAS series:

```bash
deconv_spec examples/CuGaXAS/cgo_cu.yaml --outdir results/cgo_cu
```

### CuGaO<sub>2</sub> Ga K-edge Deconvolution

Perform a corresponding deconvolution on the CuGaO<sub>2</sub> Ga K-edge dataset:

```bash
deconv_spec examples/CuGaXAS/cgo_ga.yaml --outdir results/cgo_ga
```

This workflow uses a Kendall rank correlation constraint to encourage consistency with the Cu concentration evolution. A precomputed Cu concentration profile (`cgo_cu_c_opt.npy`) is distributed with the repository so that the example can be run directly.

### Cu<sub>2</sub>O + Ga<sub>2</sub>O<sub>3</sub> Physical Mixture Cu K-edge Deconvolution

Deconvolve Cu K-edge spectra for the Cu<sub>2</sub>O + Ga<sub>2</sub>O<sub>3</sub> physical mixture:

```bash
deconv_spec examples/CuGaXAS/mix_cu.yaml --outdir results/mix_cu
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

All required spectra, configuration files, and initialization data are distributed with the repository. Use separate output directories to avoid overwriting files: the output filenames are the same for every run. The Ga example uses the **bundled** `cgo_cu_c_opt.npy` as its Kendall reference, not the newly generated `results/cgo_cu/C_opt.npy`. Replacing that reference with a new fit requires an explicit configuration change and may change the results.

Run the automated tests with `python -m pytest tests/ -v` after installing the optional `test` dependencies. For numerical comparisons, keep the Python/PyTorch versions, initial guesses, and optimization settings consistent; exact cross-platform or GPU bitwise reproducibility is not guaranteed.

---

## Citation

If you use this software in published work, please cite the accompanying publication when it becomes available. The following entry describes a manuscript in preparation; it is **not** a published paper or a DOI. Update the citation once the publication has a permanent bibliographic record.

```text
@unpublished{
	Fornero2026,
	author={Fornero, Esteban L. and Kim, Changhae Andrew and Boscoboinik, J. Anibal and Dalotto, Florencia and Qu, Xiaohui and Chen, Xiaobo and Bjorklund, Jennifer and Stavitski, Eli and Yang, Judith and Mason, Sara E. and Ganduglia-Pirovano, M. Ver{\'o}nica and Bonivardi, Adrian L. and Lustemberg, Pablo and Stacchiola, Dario},
	title={Direct Propylene Epoxidation on Partially Oxidized Copper Centers Stabilized within a Mixed Oxide},
	note={Manuscript in preparation},
	year={2026},
}
```

---

## Version and license

The initial software version is **0.0.1**. This is a pre-1.0 research release; interfaces and configuration options may evolve. See the repository's `LICENSE` file for the approved software terms.

---

## Acknowledgement

This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704.

---

## Disclaimer

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
