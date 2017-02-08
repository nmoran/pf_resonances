#! /usr/bin/env python

""" PFData.py: Utility to get various useful data for PF systems. """

__author__      = "Niall Moran"
__copyright__   = "Copyright 2016"
__license__ = "GPL"
__version__ = "0.1"
__email__ = "niall.moran@gmail.com"


import parafermions as pf
import numpy as np
import argparse
import sys
import time
import h5py


def PFData(argv):
    Parser = argparse.ArgumentParser(description="Utility to get various useful data for PF systems.")
#    Parser.add_argument('command', type=str, default='resonances', nargs=1, help='What type of data to calculate (default=resonances).')
    Parser.add_argument('--clock-degree', '-N', type=int, default=3, help='Number of positions on the clock of the clock model (default=3).')
    Parser.add_argument('--length', '-L', type=int, default=5, help='Length of the chain (default=5).')
    Parser.add_argument('--verbose', '-v', action='store_true', help='Be verbose in the output.')
    Parser.add_argument('--min-theta', type=float, default=0.0, help='Minimum theta in output range (default=0.0).')
    Parser.add_argument('--max-theta', type=float, default=0.0, help='Maximum theta in output range (default=pi/N).')

    #--------------------
    # Parse arguments
    #--------------------
    ArgVals = Parser.parse_args(argv)
    N = ArgVals.clock_degree
    L = ArgVals.length
    Verbose = ArgVals.verbose
    min_theta = ArgVals.min_theta
    if ArgVals.max_theta == 0.0:
        max_theta = np.pi/N
    else:
        max_theta = ArgVals.max_theta

    # We look at the dimension of bands and other details first
    H0 = pf.ParafermionicChainOp(N, L, 1.0, 0.0, 0.0, 0.0, 0)
    partitions, energies = H0.get_bands_and_energies()
    phases = np.pi*2.0 * np.arange(N)/np.float(N)
    resonance_points_unique = np.zeros(0)

    tol = 1e-15
    resonance_points = []
    unique_resonance_points = []
    NbrBands = partitions.shape[0]
    for i in range(NbrBands):
        for j in range(NbrBands):
            if i != j:
                p = np.sum(np.cos(phases) * (partitions[i] - partitions[j]))
                q = np.sum(np.sin(phases) * (partitions[i] - partitions[j]))
                if np.abs(q) > tol and np.abs(p) > tol:
                    theta = np.arctan(p/q)
                    if min_theta < theta < max_theta:
                        resonance_points.append((np.sum(np.abs(partitions[i] - partitions[j])), theta))
                        if len(np.where(np.abs(resonance_points_unique - theta) < tol)[0]) == 0:
                            resonance_points_unique = np.hstack([resonance_points_unique, [theta]])
                            resonance_points_unique = np.sort(resonance_points_unique)

    if Verbose:
        print('Idx\tn_c\tEnergy')
        for i, res in enumerate(resonance_points):
            print(str(i) + '\t' + str(res[0]) + '\t' + str(res[1]))
    else:
        print(' '.join(map(lambda x: str(x), resonance_points_unique)))


def __main__():
   PFData(sys.argv[1:])


if __name__ == '__main__':
    __main__()
