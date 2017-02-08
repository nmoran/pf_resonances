#! /usr/bin/env python

""" PFPT.py: Utility to do perturbative expansions for parafermion chain systems. """

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


def PFPT(argv):
    Parser = argparse.ArgumentParser(description="Utility to do perturbative expansions for parafermion chain hamiltonians.")
    Parser.add_argument('--clock-degree', '-N', type=int, default=3, help='Number of positions on the clock of the clock model (default=3).')
    Parser.add_argument('--length', '-L', type=int, default=5, help='Length of the chain (default=5).')
    Parser.add_argument('-J', type=float, default=1.0, help='J parameter (default=1.0).')
    Parser.add_argument('-f', type=float, default=0.01, help='f parameter (default=0.01).')
    Parser.add_argument('-q', type=int, default=-1, help='The q sector to use, if -1 then all q sectors will be treated in series (default=-1).')
    Parser.add_argument('--qs', type=int, nargs='*', help='List of q values to use. Only considered if no q option given.')
    Parser.add_argument('--perturb-theta', action='store_true', help='Perturb around given theta.')
    Parser.add_argument('--theta0', type=float, default=0.0, help='Theta value to perturb around (default=0.0).')
    # theta details
    Parser.add_argument('--min-theta', type=float, default=0.0, help='Minimum theta value (default=0.0).')
    Parser.add_argument('--max-theta', type=float, default=0.0, help='Maximum theta value, if zero will use pi/2N (default=0.0).')
    Parser.add_argument('--thetas', type=float, nargs='+', help='List of thetas to use.')
    Parser.add_argument('--nthetas', type=int, default=51, help='Number of theta values between min and max to use (default=50).')
    # phi details
    Parser.add_argument('--min-phi', type=float, default=0.0, help='Minimum phi value (default=0.0).')
    Parser.add_argument('--max-phi', type=float, default=0.0, help='Maximum phi value, if zero will use pi/2N (default=0.0).')
    Parser.add_argument('--phis', type=float, nargs='+', help='List of phis to use.')
    Parser.add_argument('--nphis', type=int, default=51, help='Number of phi values between min and max to use (default=50).')
    # output preferences
    Parser.add_argument('--output-prefix', type=str, default='', help='Filename to use to store output. If empty one is generated from the options.')
    Parser.add_argument('--output-id', type=str, default='', help='An additional string that is added onto the prefix to identify the run.')
    # perturbation theory specific details
    Parser.add_argument('--order', type=int, default=3, help='The order which to perform the expansion up to (default=3).')
    Parser.add_argument('--bands', type=int, nargs='*', help='The indices of unperturbed bands to start from (indices are taken at theta=0 point).' )
    Parser.add_argument('--verbose', '-v', action='store_true', help='Flag that output should be verbose.')
    Parser.add_argument('--save-matrices', action='store_true', help='Flag that effective matrices are saved.')
    Parser.add_argument('--exact', action='store_true', help='Flag that estimates should be compared to exact values (only works for small systems).')
    Parser.add_argument('--delta-E', type=float, default=20.0, help='Energy window around band to include (default=20.0).')

    #--------------------
    # Parse arguments
    #--------------------
    ArgVals = Parser.parse_args(argv)
    N = ArgVals.clock_degree
    L = ArgVals.length
    J = ArgVals.J
    f = ArgVals.f

    # set up phi values
    min_phi = ArgVals.min_phi
    max_phi = ArgVals.max_phi
    if max_phi == 0:
        max_phi = np.pi/float(2*N)
    if ArgVals.phis is not None:
        phis = ArgVals.phis
        NbrPhis = len(phis)
    else:
        NbrPhis = ArgVals.nphis
        phis = np.linspace(min_phi, max_phi, NbrPhis)

    # set up theta values
    min_theta = ArgVals.min_theta
    max_theta = ArgVals.max_theta
    if max_theta == 0:
        max_theta = np.pi/float(2*N)
    if ArgVals.thetas is not None:
        thetas = ArgVals.thetas
        NbrThetas = len(thetas)
    else:
        NbrThetas = ArgVals.nthetas
        thetas = np.linspace(min_theta, max_theta, NbrThetas)

    # set up q values
    q = ArgVals.q
    qs = ArgVals.qs
    if q == -1:
        if qs is None: qs = []
        if len(qs) == 0: qs = range(N)
    else:
        qs = [q]

    # look at other command line arguments
    Order = ArgVals.order
    Bands = ArgVals.bands
    Verbose = ArgVals.verbose
    SaveMatrices = ArgVals.save_matrices
    Method = 'BlochPT'
    Exact = ArgVals.exact
    deltaE = ArgVals.delta_E

    # set up output file name
    OutputID = ArgVals.output_id
    if ArgVals.output_prefix == '':
        OutputPrefix = ('N_' + str(N) + '_L_' + str(L) + '_J_' + str(J)
                        + '_f_' + str(f) + '_' + Method
                        + ('_' + OutputID if OutputID != '' else ''))
    else:
        OutputPrefix = ArgVals.output_prefix

    # We look at the dimension of bands and other details first
    theta0 = (thetas[0] if not ArgVals.perturb_theta else ArgVals.theta0)
    H0 = pf.ParafermionicChainOp(N, L, J, theta0, f, 0.0, 0)
    partitions, energies = H0.get_bands_and_energies()
    SubspaceDimension = 0
    for band in Bands:
        SubspaceDimension += int(H0.get_band_dimension(partitions[band]))
    if Verbose: print('Dimension of starting band(s): ' + str(SubspaceDimension) + '.')

    #-----------------------------------------------------------
    # Prepare the output file and create the necessary structure
    #-----------------------------------------------------------
    MyFile = h5py.File(OutputPrefix + '.hdf5', 'w')
    dset = MyFile.create_dataset("phis", (NbrPhis,1), dtype=np.float64)
    dset[:,0] = phis[:]
    dset = MyFile.create_dataset("thetas", (NbrThetas,1), dtype=np.float64)
    dset[:,0] = thetas[:]
    MyFile.create_group('estimates')
    MyFile['estimates'].create_group('q')
    MyFile.create_dataset('valid', (NbrPhis, NbrThetas), dtype=np.int8)
    for my_q in qs:
        MyFile.create_dataset('estimates/q/' + str(my_q), (NbrPhis, NbrThetas, Order, SubspaceDimension), dtype=np.complex128)
    if SaveMatrices:
        MyFile.create_group('matrices')
        MyFile['matrices'].create_group('q')
        for my_q in qs:
            MyFile.create_dataset('matrices/q/' + str(my_q), (NbrPhis, NbrThetas, Order, SubspaceDimension, SubspaceDimension), dtype=np.complex128)
    if Exact:
        MyFile.create_group('exact_energies')
        MyFile['exact_energies'].create_group('q')
        for my_q in qs:
            MyFile.create_dataset('exact_energies/q/' + str(my_q), (NbrPhis, NbrThetas, SubspaceDimension), dtype=np.float64)
    MyFile.close()

    #--------------------
    # Iterate over different regimes
    #--------------------
    s = time.time()
    for phi_idx in range(NbrPhis):
        phi = phis[phi_idx]
        for theta_idx in range(NbrThetas):
            s1 = time.time()
            theta = thetas[theta_idx]
            theta0 = (theta if not ArgVals.perturb_theta else ArgVals.theta0)
            H0.theta = theta0
            new_energies = H0.get_band_energies(partitions) # get the updated energies at this positions
            idxs = np.argsort(new_energies)
            band_idxs = []
            for band in Bands:
                band_idxs.append(np.where(idxs == band)[0][0])

            data = pf.pt_expansion(N, L, J, theta, f, phi, band_idxs, deltaE, Order, Exact, return_matrices=SaveMatrices, qs=qs, verbose=Verbose, theta0=theta0)
            if data is None:
                if Verbose: print('Invalid PT expansion.')
                MyFile = h5py.File(OutputPrefix + '.hdf5', 'a')
                MyFile['valid'][phi_idx, theta_idx] = 0
                MyFile.close()
            else:
                MyFile = h5py.File(OutputPrefix + '.hdf5', 'a')
                MyFile['valid'][phi_idx, theta_idx] = 1
                for q_idx, my_q in zip(range(len(qs)), qs):
                    MyFile['estimates/q/' + str(my_q)][phi_idx, theta_idx, :] = data[0][q_idx,:]
                if Exact:
                    for q_idx, my_q in zip(range(len(qs)), qs):
                        MyFile['exact_energies/q/' + str(my_q)][phi_idx, theta_idx, :] = data[1][q_idx,:]
                if SaveMatrices:
                    for q_idx, my_q in zip(range(len(qs)), qs):
                        MyFile['matrices/q/' + str(my_q)][phi_idx, theta_idx, :] = data[-1][q_idx,:]
                MyFile.close()
                e1 = time.time()
                print('Data point took ' + str(e1-s1) + ' seconds.')


    e = time.time()
    if Verbose: print('Total  time was ' + str(e-s) + ' seconds.')



def __main__():
   PFPT(sys.argv[1:])


if __name__ == '__main__':
    __main__()
