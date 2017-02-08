#!/usr/bin/env python
"""
Script for plotting the unperturbed spectra of parafermion chains.

"""
import numpy as np
import parafermions as pf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


def plot_unperturbed_spectrum(N, L, colour=False, ncolours=None):
    # create operator object
    H = pf.ParafermionicChainOp(N=N, L=L, J=1.0, theta=0.0, f=0.0, phi=0.0)
    partitions, energies = H.get_bands_and_energies()

    # values of theta to use
    thetas = np.linspace(-np.pi, np.pi, 51)
    P = pf.PartitionTable(N, L-1)
    NbrBands = len(energies)

    # create array to store energies
    Es = np.zeros((len(thetas), NbrBands), dtype=np.float64)

    for i in range(len(thetas)):
        H.theta = thetas[i]
        _, Es[i,:] = H.get_bands_and_energies(refresh=True, sort=False)

    # some fun to make sure legends appear in order
    colours = sns.color_palette("hls", n_colors=(N if ncolours is None else ncolours))
    colouring = np.dot(partitions, np.arange(N).T) % N
    colours_drawn = [True] * N
    colours_drawn[0] = False
    plt.figure(figsize=(6,4))
    for i in range(NbrBands):
        if colour:
            cidx = colouring[i]
            plt.plot(thetas, Es[:,i], 'b-', lw=1.0, color=colours[cidx],
                     label=(None if colours_drawn[cidx] else 'p=%d'%(cidx,)))
            if not colours_drawn[cidx]:
                colours_drawn[cidx] = True
                if cidx < N-1: colours_drawn[cidx+1] = False
        else:
            plt.plot(thetas, Es[:,i], 'b-', lw=1.0, color='k')

    plt.xlim((thetas[0], thetas[-1]))
    ylim = plt.ylim(); plt.ylim((ylim[0]*1.2, ylim[1]*1.2))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$E$')
    sns.despine(left=True, bottom=True)
    plt.xticks(np.linspace(-np.pi,np.pi,7), [r'$-\pi$',r'$-\frac{2\pi}{3}$',
                                             r'$-\frac{\pi}{3}$', r'$0$',
                                             r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$',
                                             r'$\pi$'])

    sns.despine(bottom=True, left=True)
    if colour: plt.legend(loc='upper center', ncol=N, bbox_to_anchor=(0.5, 1.07))

N=2; L=14
plot_unperturbed_spectrum(N, L)
plt.tight_layout()
plt.savefig('FullSpectrum_N_' + str(N) + '_L_' + str(L) + '_f_0.pdf')

N=3; L=6
plot_unperturbed_spectrum(N, L, colour=True, ncolours=4)
plt.tight_layout()
plt.savefig('FullSpectrum_Colours_N_' + str(N) + '_L_' + str(L) + '_f_0.pdf')

N=4; L=4
plot_unperturbed_spectrum(N, L, colour=True, ncolours=4)
plt.tight_layout()
plt.savefig('FullSpectrum_Colours_N_' + str(N) + '_L_' + str(L) + '_f_0.pdf')
