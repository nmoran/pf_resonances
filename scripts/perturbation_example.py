#!/usr/bin/env python
"""
Script for calculating the difference between exact energies and estimates from
perturbative expansion.

"""
import numpy as np
import parafermions as pf
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
sns.set_style("white")

theta=0.3
N=3
L=10
J=1.0
bands=[1]
fs = np.linspace(0.001, 0.05, 11)
max_order=7
exact = False

data = Parallel(n_jobs=4)(delayed(pf.pt_expansion)(N=N, L=L, J=J,
                                                   theta=theta, f=f, phi=0.0,
                                                   band_idxs=bands, deltaE=30.0,
                                                   max_order=max_order,
                                                   exact=exact, qs=[0,1])
                          for f in fs)

PTEstimates = np.stack(list(map(lambda x: x[0], data)))

sns.set_style("white",{'xtick.direction': u'in', 'xtick.major.size': 5.0, 'xtick.minor.size': 0.0,
                       'ytick.direction': u'in', 'ytick.major.size': 5.0, 'ytick.minor.size': 0.0})

colours = sns.color_palette('hls', n_colors=max_order)
plt.figure()
for order in range(max_order):
    plt.plot(fs, np.max(np.abs(PTEstimates[:, 1, order,:]-PTEstimates[:, 0, order,:]),axis=1), label='n=' + str(order+1), color=colours[order])
    plt.plot(fs, fs**(order+1), color=colours[order], ls='--')
plt.yscale('log')
plt.legend(loc='lower right', ncol=3)
plt.xlim((fs[0], fs[-1]))
plt.xlabel(r'$f$')
plt.ylabel(r'max$|E_{q=1}^{(n)}-E_{q=0}^{(n)}|$')
plt.savefig('N' + str(N) + '_L' + str(L) + '_PT_Diff_theta_' + str(theta) + '_bands_' + "_".join(map(lambda x: str(x), bands)) + '.pdf')
