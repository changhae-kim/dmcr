import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from larch.io.columnfile import read_ascii, write_ascii
from scipy.interpolate import interp1d

from spec_decomp import SpectralDecomposition, MultipleSpectraDecomposition

##################################################

dirname = 'CuGaXAS/Nt3Ds1Ns357norm'

param_dicts = [
        {'expt': '2_CuGaO2_redox', 'edge': 'Cu-K', 'n_spec': 65, 'index': 0, },
        {'expt': '2_CuGaO2_redox', 'edge': 'Ga-K', 'n_spec': 65, 'index': 1, },
        {'expt': '5_CuGaO2_rxn',   'edge': 'Cu-K', 'n_spec': 88, 'index': 4, },
        {'expt': '5_CuGaO2_rxn',   'edge': 'Ga-K', 'n_spec': 89, 'index': 5, },
        ]

param_dicts[0]['subindices'] = list(range(1, 34)) + list(range(34, 61)) + list(range(62, 65))
param_dicts[1]['subindices'] = list(range(0, 33)) + list(range(34, 61)) + list(range(61, 64))
param_dicts[2]['subindices'] = list(range(0, 13)) + list(range(13, 88))
param_dicts[3]['subindices'] = list(range(0, 13)) + list(range(14, 89))

reselector = list(range(0+9, 63)) + list(range(63+10, 151))

edge_dict = {
        'Cu-K':  8979.0, #  8978.9,
        'Ga-K': 10367.2, # 10367.1,
        }

pre_edge = {
        'pre1':   -20.0,
        'post1': +100.0,
        'pre2':   -50.0,
        'post2': +100.0,
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
    specs = np.array(spec_dict[edge])
    spec_dict[edge] = np.full_like(specs, np.nan)
    spec_dict[edge][reselector] = specs[reselector]

##################################################

VT_dict = {}

R_cutoff = 0.0001

for edge in spec_dict.keys():

    grids = grid_dict[edge]
    specs = spec_dict[edge]

    U, s, VT = np.linalg.svd(specs[reselector])
    R = U @ np.diag(s)

    n_ref = 0
    R_factor = 1.0
    while np.any(R_factor > R_cutoff):
        n_ref += 1
        svd_specs = R[:,:n_ref] @ VT[:n_ref,:]
        R_factor = np.linalg.norm(svd_specs - specs[reselector], axis=1)**2 / np.linalg.norm(specs[reselector], axis=1)**2

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
        plt.plot(grids[0], specs[reselector][imin] + k + 0.0, label=f'PC{k+1} min (i={imin})')
        plt.plot(grids[0], specs[reselector][imax] + k + 0.5, label=f'PC{k+1} max (i={imax})')
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
        [  0,  63, r'CuGaO$_2$', ],
        [ 63, 151, r'CuGaO$_2$', ],
        ]

gases = [
        [  0,  20, r'H$_2$ $\to$',      ],
        [ 20,  35, r'Ar $\to$',         ],
        [ 35,  59, r'O$_2$ $\to$',      ],
        [ 59,  66, r'',                 ],
        [ 66,  81, r'H$_2$ $\to$',      ],
        [ 81,  87, r'Ar',               ],
        [ 87, 148, r'PP + O$_2$ $\to$', ],
        [148, 152, r'',                 ],
        ]

ticks = [
        0, 17, 20, 33, 35, 55, 59, 63,
        66, 77, 81, 87, 89, 95, 101, 111, 116, 127, 131, 144, 148, 151,
        ]

temps = [
        25, 380, 380, 25, 25, 400, 400, 25,
        25, 250, 250, 50, 50, 100, 100, 200, 200, 300, 300, 400, 400, 25,
        ]

mticks = [
        0.1, 18.5, 34.0, 57.0,
        64.5, 79.0, 88.0, 98.0, 113.5, 129.0, 146.0, 150.9,
        ]

mtemps = [
        25, 380, 25, 400,
        25, 250, 50, 100, 200, 300, 400, 25,
        ]

for edge in spec_dict.keys():

    grids = grid_dict[edge]
    specs = spec_dict[edge]

    peaks = peak_dict[edge] - peak_shift

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ymin, ymax = specs[reselector].min(), specs[reselector].max()
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

C0 = 0.30
##C0 = 0.40

CT = np.zeros((4, 151))
CT[0,    :   ] = 1.0 - C0
CT[2,    : 20] = 0.5 * C0 * np.arange(20)/(19)
CT[2,  20: 35] = 0.5 * C0
CT[2,  35: 59] = 0.5 * C0 * (1.0 - np.arange(24)/(23))
CT[2,  59: 66] = 0.0 * C0
CT[2,  66: 81] = 0.1 * C0 * np.arange(15)/(14)
CT[2,  81: 87] = 0.1 * C0
CT[2,  87:   ] = 0.1 * C0 * (1.0 - np.arange(64)/(63))
CT[3,    : 35] = 0.0 * C0
CT[3,  35: 59] = 0.5 * C0 * np.arange(24)/(23)
CT[3,  59: 63] = 0.5 * C0
CT[3,  63: 87] = 0.0 * C0
CT[3,  87:   ] = 0.1 * C0 * np.arange(64)/(63)
CT[1,    :   ] = 1.0 - CT[0, :] - CT[2, :] - CT[3, :]
C = np.full_like(CT.T, np.nan)
C[reselector] = CT.T[reselector]

CuT = np.zeros((3, 151))
CuT[0,:] = CT[0,:] + CT[1,:]
CuT[1,:] = CT[2,:]
CuT[2,:] = CT[3,:]
Cu = np.full_like(CuT.T, np.nan)
Cu[reselector] = CuT.T[reselector]

GaT = np.zeros((2, 151))
GaT[0,:] = CT[0,:] + CT[1,:]
GaT[1,:] = CT[2,:] + CT[3,:]
Ga = np.full_like(GaT.T, np.nan)
Ga[reselector] = GaT.T[reselector]

C_guess_dict = {
        ##'Cu-K': Cu,
        ##'Ga-K': Ga,
        'Cu-K': C,
        'Ga-K': C,
        }

ST_guess_dict = {
        'Cu-K': np.array([
            ##spec_dict['Cu-K'][reselector][0],
            ##spec_dict['Cu-K'][reselector][13],
            ##spec_dict['Cu-K'][reselector][53],
            spec_dict['Cu-K'][reselector][0],
            spec_dict['Cu-K'][reselector][0],
            spec_dict['Cu-K'][reselector][13],
            spec_dict['Cu-K'][reselector][53],
            ]),
        'Ga-K': np.array([
            ##spec_dict['Ga-K'][reselector][0],
            ##spec_dict['Ga-K'][reselector][13],
            spec_dict['Ga-K'][reselector][0],
            spec_dict['Ga-K'][reselector][0],
            spec_dict['Ga-K'][reselector][13],
            spec_dict['Ga-K'][reselector][53],
            ]),
        }

C_opt_dict = {}
ST_opt_dict = {}

species_dict = {
        'Cu-K': [
            ##r'Cu$^{+}$ (CuGaO$_2$)',
            ##r'Cu$^{\delta+}$ (CuGaO$_{2-x}$)',
            ##r'Cu$^{(1+\epsilon)+}$ (CuGaO$_{2+y}$)',
            r'Cu$^{+}$ (Bulk)',
            r'Cu$^{+}$ (Surface)',
            r'Cu$^{\delta+}$ (CuGaO$_{2-x}$)',
            r'Cu$^{(1+\epsilon)+}$ (CuGaO$_{2+y}$)',
            ],
        'Ga-K': [
            ##r'Ga$^{3+}$ (CuGaO$_2$)',
            ##r'Ga$^{3+}$ (CuGaO$_{2\pm{z}}$)',
            r'Ga$^{3+}$ (Bulk)',
            r'Ga$^{3+}$ (Surface)',
            r'Ga$^{3+}$ (CuGaO$_{2-x}$)',
            r'Ga$^{3+}$ (CuGaO$_{2+y}$)',
            ],
        }

def dominant_species(C, ST, D):
    w = 1.0
    i = 0
    c0 = 1.0 - C0
    dC = C[:,i] - c0
    loss = w * torch.mean(dC ** 2)
    return loss

def temporal_smoothness(C, ST, D):
    w = 0.001
    i = 62
    dC = C[1:,:] - C[:-1,:]
    dC[i,:] = 0.0
    loss = w * torch.mean(dC ** 2)
    return loss

def spectral_smoothness_Cu(C, ST, D):
    ##W = [0.2, 0.01, 0.01]
    ##W = [0.1, 0.01, 0.01]
    ##W = [0.05, 0.01, 0.01]
    ##W = [0.165, 0.1, 0.01, 0.01]
    ##W = [0.09, 0.05, 0.01, 0.01]
    ##W = [0.14, 0.05, 0.01, 0.01]
    W = [0.09, 0.02, 0.01, 0.01]
    ##W = [0.1, 0.01, 0.01, 0.01]
    loss = 0.0
    for w, st in zip(W, ST):
        loss += w * torch.nn.functional.mse_loss(st[1:], st[:-1])
    return loss

def spectral_smoothness_Ga(C, ST, D):
    ##W = [0.3, 0.01]
    ##W = [0.2, 0.01]
    ##W = [0.1, 0.01]
    ##W = [0.19, 0.1, 0.01, 0.01]
    W = [0.31, 0.1, 0.01, 0.01]
    loss = 0.0
    for w, st in zip(W, ST):
        loss += w * torch.nn.functional.mse_loss(st[1:], st[:-1])
    return loss

def kendall_rank(C, ST, D):
    w = 0.001
    n_timesteps, n_species = C.size()
    C_ref = torch.tensor(C_opt_dict['Cu-K'][reselector], requires_grad=False)
    pred = C[:,None,:] - C[None,:,:]
    target = torch.sign(C_ref[:,None,:] - C_ref[None,:,:])
    product = pred * target
    same_sign = (product > 0.0)
    opp_sign = (product < 0.0)
    indices = torch.arange(n_species)
    for i in range(n_species):
        selector = (indices == i)
        n_same = max(torch.numel(product[same_sign & selector]), 1)
        n_opp = max(torch.numel(product[opp_sign & selector]), 1)
        product[same_sign & selector] *= n_opp / max(n_same, n_opp)
    loss = w * (- torch.sum(product) / ((n_timesteps ** 2 - n_timesteps) * n_species))
    return loss

constraints_dict = {
        'Cu-K': [
            ##temporal_smoothness,
            ##spectral_smoothness_Cu,
            dominant_species,
            temporal_smoothness,
            spectral_smoothness_Cu,
            ],
        'Ga-K': [
            ##temporal_smoothness,
            ##spectral_smoothness_Ga,
            dominant_species,
            temporal_smoothness,
            spectral_smoothness_Ga,
            kendall_rank,
            ],
        }

no_plot_thresh = 0.5

for edge in spec_dict.keys():

    grids = grid_dict[edge]
    specs = spec_dict[edge]

    C_guess = C_guess_dict[edge]
    ST_guess = ST_guess_dict[edge]

    species = species_dict[edge]
    constraints = constraints_dict[edge]

    n_timesteps = specs[reselector].shape[0]
    n_energies = specs[reselector].shape[1]
    n_species = len(species)

    D = np.array(specs)

    model = SpectralDecomposition(
            n_timesteps, n_energies, n_species,
            C_guess=C_guess[reselector], ST_guess=ST_guess,
            C_min=0.001, ST_min=0.001,
            )

    optimizer = optim.LBFGS(model.parameters(), history_size=10000)
    scheduler = None
    model.train(D[reselector], optimizer, scheduler, constraints,
                max_epoches=10000, verbose=10, gtol=1e-7, xtol=1e-2)

    C_opt = np.full_like(C_guess, np.nan)
    C_opt[reselector] = model.concentrations().detach().numpy()
    ST_opt = model.spectra().detach().numpy()
    D_opt = np.full_like(specs, np.nan)
    D_opt[reselector] = C_opt[reselector] @ ST_opt

    C_opt_dict[edge] = C_opt
    ST_opt_dict[edge] = ST_opt

    e0 = edge_dict[edge]
    emin = e0 + pre_edge['pre2']
    emax = e0 + pre_edge['post2']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(4.8, 3.6))
    for k, sp in enumerate(species):
        if len(species) < 4:
            plt.plot(grids[0], ST_guess[k], color=colors[k+1], label=sp)
        else:
            plt.plot(grids[0], ST_guess[k], label=sp)
    plt.xlim(emin, emax)
    plt.xlabel(r'$E$ (eV)')
    plt.ylabel(r'Normalized $\mu(E)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_SD_ST_guess.svg')
    plt.close()

    plt.figure(figsize=(4.8, 3.6))
    for k, sp in enumerate(species):
        if len(species) < 4:
            plt.plot(grids[0], ST_opt[k], color=colors[k+1], label=sp)
        else:
            plt.plot(grids[0], ST_opt[k], label=sp)
    plt.xlim(emin, emax)
    plt.xlabel(r'$E$ (eV)')
    plt.ylabel(r'Normalized $\mu(E)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{edge}_SD_ST_opt.svg')
    plt.close()

    for i in range(n_timesteps):
        if i % 20 != 0:
            continue
        print(i, C_opt[reselector][i,:])
        plt.figure(figsize=(4.8, 3.6))
        plt.plot(grids[reselector][i], specs[reselector][i], label='Data')
        plt.plot(grids[reselector][i], D_opt[reselector][i], label='Model')
        plt.xlim(emin, emax)
        plt.xlabel(r'$E$ (eV)')
        plt.ylabel(r'Normalized $\mu(E)$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{edge}_SD_D{i:03d}.svg')
        plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ymax = 1.0
    for k, sp in enumerate(species):
        if len(species) > 2 and np.all(C_guess[reselector][:, k] > no_plot_thresh):
            ymax = 1.0 - C_guess[reselector][:, k].max()
            break
    for t0, t1, mater in maters:
        x = np.arange(t0, t1)
        for k, sp in enumerate(species):
            if len(species) > 2 and np.all(C_guess[reselector][:, k] > no_plot_thresh):
                continue
            y = C_guess[t0:t1, k]
            label = (sp) if (t0 == 0) else (None)
            if len(species) < 4:
                plt.plot(x, y, color=colors[k+1], label=label)
            else:
                plt.plot(x, y, color=colors[k], label=label)
    plt.vlines(ticks, 0, 1, color='k', alpha=0.2, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in gases], 0, 1, color='k', alpha=1.0, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in maters], 0, 1, color='k', alpha=1.0, linewidth=1.0)
    for t0, t1, mater in maters:
        plt.text(t0, 1.08 * ymax, mater)
    for t0, t1, gas in gases:
        plt.text(t0, 1.02 * ymax, gas)
    plt.xlim(maters[0][0], maters[-1][1]-1)
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
        if len(species) > 2 and np.all(C_opt[reselector][:, k] > no_plot_thresh):
            ymax = 1.0 - C_opt[reselector][:, k].max()
            break
    for t0, t1, mater in maters:
        x = np.arange(t0, t1)
        for k, sp in enumerate(species):
            if len(species) > 2 and np.all(C_opt[reselector][:, k] > no_plot_thresh):
                continue
            y = C_opt[t0:t1, k]
            label = (sp) if (t0 == 0) else (None)
            if len(species) < 4:
                plt.plot(x, y, color=colors[k+1], label=label)
            else:
                plt.plot(x, y, color=colors[k], label=label)
    plt.vlines(ticks, 0, 1, color='k', alpha=0.2, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in gases], 0, 1, color='k', alpha=1.0, linewidth=0.5)
    plt.vlines([t0 for t0, _, _ in maters], 0, 1, color='k', alpha=1.0, linewidth=1.0)
    for t0, t1, mater in maters:
        plt.text(t0, 1.08 * ymax, mater)
    for t0, t1, gas in gases:
        plt.text(t0, 1.02 * ymax, gas)
    plt.xlim(maters[0][0], maters[-1][1]-1)
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

    for k, sp in enumerate(species):
        write_ascii(f'{edge}_SD_ST_opt_{k}.dat', grids[0], ST_opt[k])

