import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from larch.io.columnfile import read_ascii, write_ascii
from scipy.interpolate import interp1d

from mcr import *

##################################################

dirname = '../data'

param_dicts = [
        {'expt': '4_Cu2O+Ga2O3_redox', 'edge': 'Cu-K', 'n_spec': 64, 'index': 2, },
        {'expt': '4_Cu2O+Ga2O3_redox', 'edge': 'Ga-K', 'n_spec': 66, 'index': 3, },
        {'expt': '6_Cu2O+Ga2O3_rxn',   'edge': 'Cu-K', 'n_spec': 41, 'index': 6, },
        {'expt': '6_Cu2O+Ga2O3_rxn',   'edge': 'Ga-K', 'n_spec': 41, 'index': 7, },
        ]

param_dicts[0]['subindices'] = list(range(0, 34)) + list(range(34, 44)) + list(range(44, 64))
param_dicts[1]['subindices'] = list(range(0, 34)) + list(range(35, 45)) + list(range(46, 66))
param_dicts[2]['subindices'] = list(range(0, 41))
param_dicts[3]['subindices'] = list(range(0, 41))

edge_dict = {
        'Cu-K':  8979.0, #  8978.9,
        'Ga-K': 10367.2, # 10367.1,
        }

pre_edge = {
        'pre1':    -20.0, #  -999.0,
        'post1':  +100.0, # +9999.0,
        'pre2':    -50.0,
        'post2':  +100.0,
        }

peak_shift = 0
grid_dict = {}
spec_dict = {}

for pd in param_dicts:

    ipd = pd['index']
    edge = pd['edge']

    e0 = edge_dict[edge]
    emin = e0 + pre_edge['pre1']
    emax = e0 + pre_edge['post1']

    grids = []
    specs = []
    for isp in pd['subindices']:
        filepath = f'./{dirname}/flat_{ipd}_{isp}.dat'
        dat = read_ascii(filepath, labels='energy flat')
        mask = (dat.energy > emin) & (dat.energy < emax)
        peak_shift = np.count_nonzero(~(dat.energy > emin))
        grids.append(dat.energy[mask])
        specs.append(dat.flat[mask])

    if edge not in spec_dict.keys():
        grid_dict[edge] = grids
        spec_dict[edge] = specs
    else:
        grid_dict[edge].extend(grids)
        spec_dict[edge].extend(specs)

    print(f'{pd["expt"]:20s} {pd["edge"]:4s} {pd["n_spec"]:2d} {len(spec_dict[edge]):3d}')

for edge in spec_dict.keys():
    grid_dict[edge] = np.array(grid_dict[edge])
    spec_dict[edge] = np.array(spec_dict[edge])

##################################################

VT_dict = {}

R_cutoff = 0.0001

for edge in spec_dict.keys():

    grids = grid_dict[edge]
    specs = spec_dict[edge]

    U, s, VT = np.linalg.svd(specs)
    R = U @ np.diag(s)

    n_ref = 0
    R_factor = 1.0
    while np.any(R_factor > R_cutoff):
        n_ref += 1
        svd_specs = R[:,:n_ref] @ VT[:n_ref,:]
        R_factor = np.linalg.norm(svd_specs - specs, axis=1)**2 / np.linalg.norm(specs, axis=1)**2

    VT_dict[edge] = VT[:n_ref,:]

    print(edge, n_ref, f'{R_factor.mean():.6f} +/- {R_factor.std():.6f} min {R_factor.min():.6f} max {R_factor.max():.6f}')

    emin, emax = grids.min(), grids.max()

    plt.figure(figsize=(4.8, 3.6))
    for k in range(n_ref):
        plt.plot(grids[0], np.abs(R[:,k]).mean() * VT[k,:] + 0.5 * k, label=f'PC{k+1}')
    plt.xlim(emin, emax)
    plt.xlabel(r'$E$ (eV)')
    plt.ylabel(r'Normalized $\mu(E)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_PCA_components.svg')
    plt.close()

    plt.figure(figsize=(4.8, 3.6))
    for k in range(n_ref):
        imin = np.argmin(R[:,k])
        imax = np.argmax(R[:,k])
        plt.plot(grids[0], specs[imin] + k + 0.0, label=f'PC{k+1} min (i={imin})')
        plt.plot(grids[0], specs[imax] + k + 0.5, label=f'PC{k+1} max (i={imax})')
    plt.xlim(emin, emax)
    plt.xlabel(r'$E$ (eV)')
    plt.ylabel(r'Normalized $\mu(E)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_PCA_minmax.svg')
    plt.close()

##################################################

peak_dict = {
        'Cu-K': [
            147, # 8981.4 eV
            226, # 8997.2 eV
            301, # 9015.989008 eV
            ],
        'Ga-K': [
            184, # 10376.8 eV
            236, # 10387.2 eV
            302, # 10404.662634 eV
            ],
        }

maters = [
        [  0,  64, r'Cu$_2$O + Ga$_2$O$_3$', ],
        [ 64, 105, r'Cu$_2$O + Ga$_2$O$_3$', ],
        ]

gases = [
        [  0,  24, r'H$_2$ $\to$',      ],
        [ 24,  36, r'Ar $\to$',         ],
        [ 36,  60, r'O$_2$ $\to$',      ],
        [ 60,  64, r'',                 ],
        [ 64,  74, r'H$_2$ $\to$',      ],
        [ 74,  78, r'Ar',               ],
        [ 78, 104, r'PP + O$_2$ $\to$', ],
        [104, 105, r'Terminated',       ],
        #[ 78, 139, r'PP + O$_2$ $\to$', ],
        #[139, 141, r'',                 ],
        ]

ticks = [
        2, 19, 24, 33, 36, 54, 60, 63,
        64, 74, 75, 77, 78, 82, 85, 90, 93, #115, 122, 134, 139, 140,
        ]

temps = [
        25, 380, 380, 25, 25, 400, 400, 25,
        25, 250, 250, 25, 25, 100, 100, 200, 200, #300, 300, 400, 400, 25,
        ]

mticks = [
        1.0, 21.5, 34.5, 56.0,
        63.5, 74.5, 77.5, 83.5, 91.5, #118.5, 136.5, 139.9,
        ]

mtemps = [
        25, 380, 25, 400,
        25, 250, 25, 100, 200, #300, 400, 25,
        ]

for edge in spec_dict.keys():

    grids = grid_dict[edge]
    specs = spec_dict[edge]

    peaks = peak_dict[edge] - peak_shift

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ymin, ymax = specs.min(), specs.max()
    ymin, ymax = ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for t0, t1, mater in maters:
        x = np.arange(t0, t1)
        for k, pk in enumerate(peaks):
            y = specs[t0:t1, pk]
            label = (f'{grids[t0, pk]:.1f} eV') if (t0 == 0) else (None)
            plt.plot(x, y, color=colors[k], label=label)
    plt.vlines(ticks, ymin, ymax, color='k', alpha=0.2, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in gases], ymin, ymax, color='k', alpha=1.0, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in maters], ymin, ymax, color='k', alpha=1.0, linewidth=1.5)
    for t0, t1, mater in maters:
        plt.text(t0, ymax + 0.08 * (ymax - ymin), mater)
    for t0, t1, gas in gases:
        plt.text(t0, ymax + 0.02 * (ymax - ymin), gas)
    plt.xlim(maters[0][0], maters[-1][1]-1)
    plt.ylim(ymin, ymax)
    plt.xlabel(r'Temperature Ramp (°C)')
    plt.ylabel(r'Normalized $\mu(E)$')
    ax.xaxis.set_ticks(ticks, labels=[])
    ax.xaxis.set_ticks(mticks, labels=mtemps, rotation=90, minor=True)
    ax.tick_params(axis='x', which='minor', width=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_PLT_peaks.svg')
    plt.close()

##################################################

CT = np.zeros((3, 105))
CT[1,    : 24] = 0.9 * np.arange(24)/(23)
CT[1,  24: 36] = 0.9
CT[1,  36: 60] = 0.9 * (1.0 - np.arange(24)/(23))
CT[1,  60: 64] = 0.0
CT[1,  64: 74] = 0.1 * np.arange(10)/(9)
CT[1,  74: 78] = 0.1
CT[1,  78:   ] = 0.1 * (1.0 - np.arange(27)/(26))
CT[2,    : 36] = 0.0
CT[2,  36: 60] = 0.9 * np.arange(24)/(23)
CT[2,  60: 64] = 0.9
CT[2,  64: 78] = 0.0
CT[2,  78:   ] = 0.1 * np.arange(27)/(26)
CT[0,    :   ] = 1.0 - CT[1, :] - CT[2, :]

GT = np.zeros((2, 105))
GT[1,   :64] = 0.3
GT[1, 64:  ] = 0.3
GT[0,   :  ] = 1.0 - GT[1, :]

C_guess_dict = {
        'Cu-K': CT.T,
        'Ga-K': GT.T,
        }

ST_guess_dict = {
        'Cu-K': np.array([
            spec_dict['Cu-K'][0],
            spec_dict['Cu-K'][33],
            spec_dict['Cu-K'][63],
            ]),
        'Ga-K': np.array([
            spec_dict['Ga-K'][0],
            spec_dict['Ga-K'][33],
            ]),
        }

species_dict = {
        'Cu-K': [
            r'Cu$^{+}$ (Cu$_2$O)',
            r'Cu$^{0}$ (Cu)',
            r'Cu$^{2+}$ (CuO)',
            ],
        'Ga-K': [
            r'Ga$^{3+}$ (Ga$_2$O$_3$)',
            r'Ga$^{3+}$ (Ga$_2$O$_{3\pm{w}}$)',
            ],
        }

temporal_smoothness = TemporalSmoothnessLoss(0.001, [63])
spectral_smoothness_Cu = SpectralSmoothnessLoss([0.01, 0.01, 0.01])
spectral_smoothness_Ga = SpectralSmoothnessLoss([0.01, 0.01])

constraints_dict = {
        'Cu-K': [
            temporal_smoothness,
            spectral_smoothness_Cu,
            ],
        'Ga-K': [
            temporal_smoothness,
            spectral_smoothness_Ga,
            ],
        }

no_plot_thresh = 1.0
xmax = 141

for edge in spec_dict.keys():

    grids = grid_dict[edge]
    specs = spec_dict[edge]

    C_guess = C_guess_dict[edge]
    ST_guess = ST_guess_dict[edge]

    species = species_dict[edge]
    constraints = constraints_dict[edge]

    n_timesteps = specs.shape[0]
    n_energies = specs.shape[1]
    n_species = len(species)

    D = torch.tensor(specs)

    model = DifferentiableMCR(
            n_timesteps, n_species, n_energies,
            C_guess=torch.tensor(C_guess), ST_guess=torch.tensor(ST_guess),
            C_min=0.001, ST_min=0.001,
            )

    optimizer = optim.LBFGS(model.parameters(), history_size=10000)
    scheduler = None
    model.fit(D, optimizer, scheduler, constraints=constraints,
              max_epochs=10000, verbose=10, gtol=1e-7, xtol=1e-2)

    C_opt = model.concentrations().detach().numpy()
    ST_opt = model.spectra().detach().numpy()
    D_opt = C_opt @ ST_opt

    e0 = edge_dict[edge]
    emin = e0 + pre_edge['pre2']
    emax = e0 + pre_edge['post2']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(4.8, 3.6))
    for k, sp in enumerate(species):
        plt.plot(grids[0], ST_guess[k], color=colors[k+4], label=sp)
    plt.xlim(emin, emax)
    plt.xlabel(r'$E$ (eV)')
    plt.ylabel(r'Normalized $\mu(E)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_SD_ST_guess.svg')
    plt.close()

    plt.figure(figsize=(4.8, 3.6))
    for k, sp in enumerate(species):
        plt.plot(grids[0], ST_opt[k], color=colors[k+4], label=sp)
    plt.xlim(emin, emax)
    plt.xlabel(r'$E$ (eV)')
    plt.ylabel(r'Normalized $\mu(E)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_SD_ST_opt.svg')
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ymax = 1.0
    for k, sp in enumerate(species):
        if len(species) > 2 and np.all(C_guess[:, k] > no_plot_thresh):
            ymax = 1.0 - C_guess[:, k].max()
            break
    for t0, t1, mater in maters:
        x = np.arange(t0, t1)
        for k, sp in enumerate(species):
            if len(species) > 2 and np.all(C_guess[:, k] > no_plot_thresh):
                continue
            y = C_guess[t0:t1, k]
            label = (sp) if (t0 == 0) else (None)
            plt.plot(x, y, color=colors[k+4], label=label)
    plt.vlines(ticks, 0, 1, color='k', alpha=0.2, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in gases], 0, 1, color='k', alpha=1.0, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in maters], 0, 1, color='k', alpha=1.0, linewidth=1.0)
    for t0, t1, mater in maters:
        plt.text(t0, 1.08 * ymax, mater)
    for t0, t1, gas in gases:
        plt.text(t0, 1.02 * ymax, gas)
    plt.xlim(0, xmax)
    plt.ylim(0.0, ymax)
    plt.xlabel(r'Temperature Ramp (°C)')
    plt.ylabel('Concentration (norm.)')
    ax.xaxis.set_ticks(ticks, labels=[])
    ax.xaxis.set_ticks(mticks, labels=mtemps, rotation=90, minor=True)
    ax.tick_params(axis='x', which='minor', width=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_SD_C_guess.svg')
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ymax = 1.0
    for k, sp in enumerate(species):
        if len(species) > 2 and np.all(C_opt[:, k] > no_plot_thresh):
            ymax = 1.0 - C_opt[:, k].max()
            break
    for t0, t1, mater in maters:
        x = np.arange(t0, t1)
        for k, sp in enumerate(species):
            if len(species) > 2 and np.all(C_opt[:, k] > no_plot_thresh):
                continue
            y = C_opt[t0:t1, k]
            label = (sp) if (t0 == 0) else (None)
            plt.plot(x, y, color=colors[k+4], label=label)
    plt.vlines(ticks, 0, 1, color='k', alpha=0.2, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in gases], 0, 1, color='k', alpha=1.0, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in maters], 0, 1, color='k', alpha=1.0, linewidth=1.0)
    for t0, t1, mater in maters:
        plt.text(t0, 1.08 * ymax, mater)
    for t0, t1, gas in gases:
        plt.text(t0, 1.02 * ymax, gas)
    plt.xlim(0, xmax)
    plt.ylim(0.0, ymax)
    plt.xlabel(r'Temperature Ramp (°C)')
    plt.ylabel('Concentration (norm.)')
    ax.xaxis.set_ticks(ticks, labels=[])
    ax.xaxis.set_ticks(mticks, labels=mtemps, rotation=90, minor=True)
    ax.tick_params(axis='x', which='minor', width=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_SD_C_opt.svg')
    plt.close()

    for i in range(n_timesteps):
        if i not in [ 0, 20, 40, 60, 80, 100, ]:
            continue
        print(i, C_opt[i,:])
        plt.figure(figsize=(4.8, 3.6))
        plt.plot(grids[i], specs[i], label='Data')
        plt.plot(grids[i], D_opt[i], label='Model')
        plt.xlim(emin, emax)
        plt.xlabel(r'$E$ (eV)')
        plt.ylabel(r'Normalized $\mu(E)$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{edge}_SD_D{i:03d}.svg')
        plt.close()

    for k, sp in enumerate(species):
        write_ascii(f'{edge}_SD_ST_opt_{k}.dat', grids[0], ST_opt[k])

