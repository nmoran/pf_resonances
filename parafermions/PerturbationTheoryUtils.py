"""
Some utility functions for dealing with perturbative expansions numerically.
"""
import parafermions.ParafermionUtils as pf
import parafermions.ParafermionUtilsCython as pfcy
import time
import numpy as np
import scipy.sparse as sps


# define a function that gets the U operator to the nth order for given operators
def Ufunc(n, Q, V, P):
    """
    The U operator, see Messiah book, chapter 16.

    Parameters:
    ----------
    n: int
        Order of expansion.
    Q: matrix
        Projection matrix rescaled by inverse energy difference.
    V: matrix
        Perturbing part of Hamiltonian.
    P: matrix
        Projection matrix to degenerate subspace we are perturbing from.

    Returns:
    --------
    matrix
        The expansion of U to nth order.
    """
    if n == 0:
        return P
    elif n > 0:
        U = V*Ufunc(n-1, Q, V, P)
        for p in range(1,n):
            U = U - Ufunc(p,Q,V,P)*V*Ufunc(n-p-1,Q,V,P)
        return Q*U


def Afunc(n, Q, V, P, As=None):
    """
    Function A operators of Bloch's perturbation theory expansion to nth order.

    Parameters
    -----------
    n: int
        The order to calculate to.
    Q: operator
        Normalised projector to all states outside band.
    V: operator
        Perturbing part of hamiltonian.
    P: operator
        Projector to band we are perturbing from.
    As: dict
        Dictionary of precomputed expansions.

    Returns
    --------
    operator
        The effective operator to nth order.
    dict
        Dictionary of operators are each order up to this.
    """
    if As is None or type(As) is not dict:
        As = dict()
    if 0 not in As:
        A = P*V*Ufunc(0,Q,V,P)
        As[0] = A
    else: A = As[0]

    for j in range(1, n):
        if j not in As:
            A_new = P*V*Ufunc(j,Q,V,P)
            As[j] = A_new
        else:
            A_new = As[j]
        A += A_new
    return A, As


def ProjectToBand(A, E0Band, E0s):
    select = np.abs(E0s - E0Band) < 1e-12
    return A[select,:][:, select]


def Sfunc(k, P0, Q):
    if k == 0:
        return -P0
    else:
        return Q**k


def AKato(n, Q, V, P0):
    # project to the unperturbed band to make multiplications more efficient
    A = None
    bins = n+1
    Table = pf.PartitionTable(bins, n)
#     print ("Partition table" + str([bins,n]))
    for idx in range(Table[bins,n]):
        ks = pf.FindPartition(bins, n, idx, Table)
#         print ks
        Tmp = Sfunc(ks[0], P0, Q)
        for k in ks[1:]:
            Tmp = Tmp * V * Sfunc(k, P0, Q)
        if A is None:
            A = Tmp
        else:
            A = A + Tmp
    return -A


def BKato(n, Q, V, P0):
    # project to the unperturbed band to make multiplications more efficient
    B = None
    bins = n+1
    Table = pf.PartitionTable(bins, n-1)
#     print ("Partition table" + str([bins,n-1]))
    for idx in range(Table[bins, n-1]):
        ks = pf.FindPartition(bins, n-1, idx, Table)
#         print ks
        Tmp = Sfunc(ks[0], P0, Q)
        for k in ks[1:]:
            Tmp = Tmp * V * Sfunc(k, P0, Q)
        if B is None:
            B = Tmp
        else:
            B = B + Tmp
    return B


def HaKato(n, Q, V, P0):
    Ha = P0 * BKato(1, Q, V, P0) * P0
    for i in range(2,n+1):
        Ha = Ha + P0 * BKato(i, Q, V, P0) * P0
    return Ha


def KaKato(n, Q, V, P0):
    Ka = P0
    for i in range(1,n+1):
        Ka = Ka + P0 * AKato(i, Q, V, P0) * P0
    return Ka


def pt_operators(N, L, J, theta, f, phi, band_idxs, deltaE, exact=False, qs=None, verbose=False, **kwargs):
    """
    Function to calculate the operators needed for perturbation theory given input parameters.

    Parameters:
    ----------
    N: int
        The number of clock positions.
    L: int
        The length of the chain.
    J: float
        J coupling constant value(s).
    theta: float
        Chiral parameter on J term.
    f: float
        f coupling constant value(s).
    phi: float
        Chiral parameter on f term.
    band_idxs: array
        Band indexes to start from.
    deltaE: float
        The energy range in which to consider bands.
    exact: bool
        Flag to indicate that exact values should be calculated using ED (default=False).
    qs: list
        List of q values to use. If None all values 0,...,N-1 used(default=None).

    Returns
    -------
    matrix
        Projector to starting subspace.
    matrix
        Projector to complement of starting subspace within given energy range
        and scaled according to the inverse unperturbed energy difference.
    matrix
        Unperturbed hamiltonian matrix.
    list of matrices
        Perturbing hamiltonian for each q requested.
    int
        The full dimension of the starting band.
    float
        The unperturbed energy of the starting band.
    list of arrays
        List of arrays of exact eigenvalues for each requested q or None if exact is false.
    """
    if qs is None:
        qs = range(N)
    BandIdxs = band_idxs
    DeltaE = deltaE
    if 'theta0' in kwargs:
        theta0 = kwargs['theta0']
    else:
        theta0 = theta
    H0Op = pf.ParafermionicChainOp(N, L, J, theta0, 0.0, 0.0, q=0) # we get MPO object for full unperturbed Hamiltonian (note the same in each sector)
    HfOps = []
    for q in qs:
        if 'exclude_side' in kwargs:
            if kwargs['exclude_side'] == 'left':
                fs = np.ones(L)*f
                fs[0] = 0.0
                Hf = pf.ParafermionicChainOp(N, L, J, theta, fs, phi, q=q)
            elif kwargs['exclude_side'] == 'right':
                fs = np.ones(L)*f
                fs[-1] = 0.0
                Hf = pf.ParafermionicChainOp(N, L, J, theta, fs, phi, q=q)
            elif kwargs['exclude_side'] == 'neither':
                fs = np.ones(L)*f
                fs[0] = 0.0
                fs[-1] = 0.0
                Hf = pf.ParafermionicChainOp(N, L, J, theta, fs, phi, q=q)
            else:
                raise Exception('\'exlude_side\' argument should be either left or right')
        else:
            Hf = pf.ParafermionicChainOp(N, L, J, theta, f, phi, q=q)
        Hf.add(H0Op, c1=1.0, c2=-1.0, inplace=True, compress=False)
        HfOps.append(Hf)
    [Partitions, H0Energies] = H0Op.get_bands_and_energies() # get all the partitions and energies of each
    BandEnergy = H0Energies[BandIdxs[0]] # get the energy of the band we start from, this is E0
    BandPartitions = list(map(lambda x: Partitions[x], BandIdxs)) # get the
    FullBand = np.vstack(list(map(lambda x: pfcy.GetFullBandDW(BandPartitions[x]), range(len(BandIdxs)))))
    FullBandDim = len(FullBand)
    [NeighbouringBands,] = np.where(np.abs(H0Energies - BandEnergy) < DeltaE) # find other bands within deltaE in energy
    FullSubspace = np.copy(FullBand)
    for NeighbouringBand in NeighbouringBands:
        if NeighbouringBand not in BandIdxs:
            FullSubspace = np.vstack((FullSubspace, pfcy.GetFullBandDW(Partitions[NeighbouringBand])))
    FullSubspaceDim = FullSubspace.shape[0]
    if verbose: print('Full subspace dim: ' + str(FullSubspaceDim) + '.')
    x = np.arange(FullSubspaceDim)
    I = sps.diags(np.ones(FullSubspaceDim), 0)
    P0 = sps.diags(np.piecewise(x, [x < FullBandDim, x >= FullBandDim], [1.0, 0.0]), 0)
    Q0 = sps.diags(np.piecewise(x, [x < FullBandDim, x >= FullBandDim], [0.0, 1.0]), 0)
    s = time.time()
    H0 = H0Op.mats_subspace(FullSubspace)
    e = time.time()
    if verbose: print('Time taken to calculate H0 matrix: ' + str(e-s) + ' seconds.')

    s = time.time()
    Hfs = list(map(lambda x : HfOps[x].mats_subspace(FullSubspace), qs))
    e = time.time()
    if verbose: print('Time taken to calculate V matrices: ' + str(e-s) + ' seconds.')
    denominators = (BandEnergy - H0.diagonal()[FullBandDim:])
    if len(np.where(denominators == 0)[0]) > 0:
        return None
    Q = sps.diags(np.hstack([np.zeros(FullBandDim),np.ones(FullSubspaceDim-FullBandDim)/denominators]), 0)

    if exact:
        Offset = np.sum(map(lambda x: len(pfcy.GetFullBandDW(Partitions[x])), range(min(BandIdxs))))
        # for debugging purposes, calculate some of full spectrum exactly, can be time consuming
        FullEs = list(map(lambda x: pf.Diagonalise(N, L, J, theta, f, phi, q=x, k=Offset + FullBandDim), qs))
        FullEs = list(map(lambda x: FullEs[x][0][Offset:(Offset+FullBandDim)], qs))
    else:
        FullEs = None

    return [P0, Q, H0, Hfs, FullBandDim, BandEnergy, FullEs]


def pt_expansion(N, L, J, theta, f, phi, band_idxs, deltaE, max_order, exact=False, return_matrices=False, qs=None, verbose=False, **kwargs):
    """
    Function to calculate the perturbative expansion with the given parameters.

    Parameters:
    ----------
    N: int
        The number of clock positions.
    L: int
        The length of the chain.
    J: float
        J coupling constant value(s).
    theta: float
        Chiral parameter on J terms.
    f: float
        f coupling constant value(s).
    phi: float
        Chiral parameter on f terms.
    band_idxs: array
        Band indexes to start from.
    deltaE: float
        The energy range in which to consider bands.
    max_order: int
        The order up to which to calculate expansions.
    exact: bool
        Flag to indicate that exact values should be calculated using ED (default=False).
    return_matrices: bool
        Flag to indicate that the effective matrices should be returned (default=False).
    qs: list
        List of q values to use. If None all values 0,...,N-1 used(default=None).
    verbose: bool
        Flag to indicate that debugging and timing information should be printed.

    Returns
    -------
    tensor
        Rank 3 tensor containing the PT estimates at each order and for each q. Indices correspond to
        q, order and state respectively.
    tensor
        Rank 2 tensor exact energy values for each q and state (only if exact is True).
    tensor
        Rank 4 tensor containing the effective matrices at each order and for each q (only if return_matrices is True).
    """

    if qs is None:
        qs = range(N)

    BandIdxs = band_idxs # index of band we are perturbing about, indexed form 0 and order by ascending energy. If more than one must be degenerate.
    DeltaE = deltaE # energy range within which to include bands

    # first calculate the operators we will need
    s = time.time()
    data = pt_operators(N, L, J, theta, f, phi, BandIdxs, DeltaE, exact, qs=qs, verbose=verbose, **kwargs)
    if data is None:
        return None
    P0, Q, H0, Hfs, FullBandDim, BandEnergy, FullEs = data
    e = time.time()
    if verbose: print ('Getting ops took ' + str(e-s))

    Heffs = np.zeros((len(qs), max_order, FullBandDim, FullBandDim), np.complex128)
    PTEstimates = np.zeros(((len(qs), max_order, FullBandDim)), dtype=np.complex128)
    if exact: ExactEnergies = np.zeros(((len(qs), FullBandDim)))

    s = time.time()
    for q_idx, q  in zip(range(len(qs)), qs):
        As = dict()
        for order in range(1, max_order+1):
            A, As = Afunc(order, Q, Hfs[q], P0, As)
            Heffs[q_idx, order-1, :, :] = A[:FullBandDim, :FullBandDim].todense()
    e = time.time()
    if verbose: print ('Calculating expansions took ' + str(e-s))

    for q_idx, q  in zip(range(len(qs)), qs):
        if exact:
            energies_exact = FullEs[q_idx]
            ExactEnergies[q_idx,:]= energies_exact
        for order in range(max_order):
            es,_ = np.linalg.eig(Heffs[q_idx,order,:,:])
            es += BandEnergy
            es = np.sort(es)
            PTEstimates[q_idx, order, :] = es

    return_data = [PTEstimates]
    if exact:
        return_data.append(ExactEnergies)
    if return_matrices:
        return_data.append(Heffs)

    return return_data
