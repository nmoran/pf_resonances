#! /usr/bin/env python

""" PFFullDiag.py: Utility to do full diagonalisation on parafermion systems. """

__author__      = "Niall Moran"
__copyright__   = "Copyright 2016"
__license__ = "GPL"
__version__ = "0.1"
__email__ = "niall.moran@gmail.com"


import parafermions.ParafermionUtils as pf
import parafermions.scalapack_wrapper as scw
import mpi4py.MPI as MPI
import numpy as np
import argparse
import sys
import pickle
import time
import h5py


def PFFullDiag(argv):
    Parser = argparse.ArgumentParser(description="Utility to diagonalise parafermion hamiltonians.")
    Parser.add_argument('--clock-degree', '-N', type=int, default=3, help='Number of positions on the clock of the clock model (default=3).')
    Parser.add_argument('--length', '-L', type=int, default=5, help='Length of the chain (default=5).')
    Parser.add_argument('-J', type=float, default=1.0, help='J parameter (default=1.0).')
    Parser.add_argument('-f', type=float, default=0.01, help='f parameter (default=0.01).')
    Parser.add_argument('-q', type=int, default=-1, help='The q sector to use, if -1 then all q sectors will be treated in series (default=-1).')
    Parser.add_argument('--qs', type=int, nargs='*', help='List of q values to use. Only considered if no q option given.')
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
    Parser.add_argument('--use-pickle', action='store_true', help='Flag to indicate that output is saved in pickle format rather than the default hdf5 format.')
    # computational details
    Parser.add_argument('--prows', type=int, default=-1, help='Specify the number of rows of processors to use (default = -1).')
    Parser.add_argument('--checkpoint', action='store_true', help='Flag to indicate that results should be written after each diagonalisation to prevent data loss.' )
    Parser.add_argument('--bs', type=int, default=64, help='The block size to use (default is 64).')
    Parser.add_argument('--debug', action='store_true', help='Flag to indicate that lower level scalapack routine should be called with debugging option.' )
    Parser.add_argument('--verbose', '-v', action='store_true', help='Flag that output should be verbose.')
    Parser.add_argument('--eigenstate', '-e', action='store_true', help='Flag that eigenstates should be saved, significant space required for large systems.')

    #--------------------
    # Parse arguments
    #--------------------
    ArgVals = Parser.parse_args(argv)
    N = ArgVals.clock_degree
    L = ArgVals.length
    J = ArgVals.J
    f = ArgVals.f
    SaveEigenStates = ArgVals.eigenstate

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

    # set up output file name
    OutputID = ArgVals.output_id
    if ArgVals.output_prefix == '':
        OutputPrefix = ('N_' + str(N) + '_L_' + str(L) + '_J_' + str(J)
                        + '_f_' + str(f)
                        + ('_' + OutputID if OutputID != '' else ''))
    else:
        OutputPrefix = ArgVals.output_prefix

    # look at other command line arguments
    PRows = ArgVals.prows
    Checkpoint = ArgVals.checkpoint
    Debug = ArgVals.debug
    BlockSize = ArgVals.bs
    UseHDF5 = not ArgVals.use_pickle
    Verbose = ArgVals.verbose

    Comm = MPI.COMM_WORLD
    my_rank = Comm.Get_rank()
    size = Comm.Get_size()

    if Verbose and my_rank == 0: print('Using ' + str(size) + ' processes.')

    NbrLevels = N**(L-1)
    if my_rank == 0 and UseHDF5:
        MyFile = h5py.File(OutputPrefix + '.hdf5', 'w')
        dset = MyFile.create_dataset("phis", (NbrPhis,1), dtype=np.float64)
        dset[:,0] = phis[:]
        dset = MyFile.create_dataset("thetas", (NbrThetas,1), dtype=np.float64)
        dset[:,0] = thetas[:]
        MyFile.create_group('q')
        for my_q in qs:
            MyFile.create_dataset('q/' + str(my_q), (NbrPhis, NbrThetas, NbrLevels), dtype=np.float64)
        MyFile.close()

    if SaveEigenStates:
        MyFile = h5py.File(OutputPrefix + '_vectors.hdf5', 'w')
        dset = MyFile.create_dataset("phis", (NbrPhis,1), dtype=np.float64)
        dset[:,0] = phis[:]
        dset = MyFile.create_dataset("thetas", (NbrThetas,1), dtype=np.float64)
        dset[:,0] = thetas[:]
        MyFile.create_group('q')
        for my_q in qs:
            MyFile.create_dataset('q/' + str(my_q), (NbrPhis, NbrThetas, NbrLevels, NbrLevels), dtype=np.complex128)
        MyFile.close()

    #--------------------
    # Start diagonalisations
    #--------------------
    pickle_time = 0
    s = time.time()
    Es = dict()
    BlacsGrid = None
    for phi_idx in range(NbrPhis):
        phi = phis[phi_idx]
        Es[phi] = dict()
        for theta_idx in range(NbrThetas):
            theta = thetas[theta_idx]
            Es[phi][theta] = dict()
            for my_q in qs:
                H = pf.ParafermionicChainOp(N, L, J, theta, f, phi, my_q)
                if not SaveEigenStates:
                    BlacsGrid, Es[phi][theta][my_q] = scw.full_parallel_diag(H, bs=BlockSize, rows=PRows, debug=Debug, timing=Verbose, once_off=False, bg=BlacsGrid, eigvectors=False)
                else:
                    BlacsGrid, Es[phi][theta][my_q], EigenStateData = scw.full_parallel_diag(H, bs=BlockSize, rows=PRows, debug=Debug, timing=Verbose, once_off=False, bg=BlacsGrid, eigvectors=True)
                if my_rank == 0:
                    if UseHDF5:
                        MyFile = h5py.File(OutputPrefix + '.hdf5', 'a')
                        dset = MyFile['q/' + str(my_q)]
                        dset[phi_idx,theta_idx,:] = Es[phi][theta][my_q]
                        MyFile.close()
                    elif Checkpoint:
                        pickle.dump(Es, open(OutputPrefix + '.pickle', 'w'))
                if SaveEigenStates:
                    for proc in range(size):
                        if proc == my_rank:
                            MyFile = h5py.File(OutputPrefix + '_vectors.hdf5', 'a')
                            dset = MyFile['q/' + str(my_q)]
                            for row in range(EigenStateData[1].shape[0]):
                                dset[phi_idx,theta_idx,EigenStateData[1][row], EigenStateData[2]] = EigenStateData[0][row,:]
                            MyFile.close()
                        MPI.COMM_WORLD.Barrier()
    e = time.time()
    if Verbose and my_rank == 0: print('Diagonalisations took ' + str(e-s) + ' seconds.')

    #--------------------
    # Write final results if not already written.
    #--------------------
    if not Checkpoint and my_rank == 0 and not UseHDF5: pickle.dump(Es, open(OutputPrefix + '.pickle', 'w'))


def __main__():
    PFFullDiag(sys.argv[1:])

if __name__ == '__main__':
    __main__()
