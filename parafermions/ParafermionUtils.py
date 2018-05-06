import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg.eigen.arpack as arp
import scipy.special as special

from parafermions.MPO import MPO
import parafermions.ParafermionUtilsCython as pfcy
import time

class ParafermionicChainOp(MPO):
    """
    Class which contains method for treating parafermion chains.
    """

    def __init__(self, N, L, J, theta, f, phi, q=-1, dtype=np.dtype('complex128'), sigmaJ=0.0, sigmaf=0.0, Fendley=False):
        """
        Constructor of parafermonic chain operator class with the given parameters.

        Parameters
        -----------
        N: int
            Dimension at each site.
        L: int
            Length of chain.
        J: float (or float array)
            Coefficient on pair term of if array given, for each term.
        theta: float
            Phase on J term.
        f: float (or float array)
            Coefficient on field term, or if array given, for each term.
        phi: float
            Phase on f term.
        q: int
            The q-sector to restrict to.
        dtype: dtype
            Datatype to use internally, default=np.dtype('complex128').
        sigmaJ: float
            The standard deviation of the disorder strength on J (default=0.01).
        sigmaf: float
            The standard deviation of the disorder strength on f (default=0.01).
        Fendley: flag
            Indicates that the original Fendley variant of the models is used (default=False).
        """
        self.q = q
        self.RealL = L # when using domain wall picture, effective length is one shorter

        if q != -1:
            self.L = L-1 # if q is set, we use hamiltonian in difference picture which is one site shorter.
        else:
            self.L = L   # otherwise full hamiltonian
        self.N = N
        self.J = J
        self.theta = theta
        self.f = f
        self.phi = phi
        self.omega = np.exp(2.0*np.pi*1j/np.float(N))

        # operators
        self.sigma = np.diag(self.omega**np.arange(N))
        self.tau = np.diag(np.ones(N-1, dtype=dtype), k=1); self.tau[-1,0] = 1.0
        self.I = np.eye(N)

        # disorder
        self.sigmaJ = np.abs(sigmaJ)
        self.sigmaf = np.abs(sigmaf)

        # setting dimension and datatype
        self.dim = N**self.L
        self.dtype = dtype


        self.J_phase = np.exp(self.theta*1j)
        self.f_phase = np.exp(self.phi*1j)

        # the MPO dimension changes if we are using the original Fendley prescription in a particular q-sector.
        if not Fendley or q == -1:
            self.chi = 4
        else:
            self.chi = N + 1 # need additional entries for the (sigma_j sigma_{j+1})^m terms.

        # If array of couplings given check length. If not array make into fixed array.
        if type(J) is np.ndarray:
            if len(J) != L-1:
                print('Length of J coefficient array must be ' + str(L-1) + '.')
                sys.exit(-1)
        else:
            J = np.ones(L-1) * J

        if sigmaJ > 0.0:
            self.Js = np.random.lognormal(np.log(J), np.abs(J*sigmaJ), L-1)
        else:
            self.Js = J

        # If array of couplings given check length. If not array make into fixed array.
        if type(f) is np.ndarray:
            if len(f) != L:
                print('Lenght of f coefficient array must be L.')
                sys.exit(-1)
        else:
            f = np.ones(L) * f

        if sigmaf > 0.0 and f > 0.0:
            self.fs = np.random.lognormal(np.log(f), np.abs(f*sigmaf), L)
        else:
            self.fs = f

        if q == -1: # if not using domain wall picture
            Ws = dict() # create a dictionary to store tensors
            for i in range(self.L):
                W = np.zeros((self.chi,self.chi,N,N), dtype=self.dtype)
                W[0,0:3] = [self.I, self.sigma.conj().T, self.sigma]
                if not Fendley:
                    W[0,-1] = -self.fs[i]*(self.f_phase*self.tau  + (self.f_phase*self.tau).conj().T)
                else:
                    for m in range(1, N): W[0,-1] -= self.fs[i]*(np.linalg.matrix_power(self.tau, m))
                if i > 0:
                    W[1,-1] = -self.Js[i-1]*self.J_phase*self.sigma
                    W[2,-1] = -self.Js[i-1]*(self.J_phase*self.sigma).conj().T
                W[-1,-1] = self.I
                Ws[i] = W
        else:
            Ws = dict() # create a dictionary to store tensors
            for i in range(self.L):
                W = np.zeros((self.chi,self.chi,N,N), dtype=self.dtype)
                if not Fendley:
                    W[0,0:3] = [self.I, self.tau.conj().T, self.tau]
                else:
                    W[0,0] = self.I
                    for m in range(1, N):
                        W[0,m] = np.linalg.matrix_power(self.tau.conj().T, m)
                W[0,-1] = -self.Js[i]*self.J_phase*self.sigma - (self.Js[i] * self.J_phase * self.sigma).conj().T
                if i == 0:
                    if not Fendley:
                        W[0,-1] = W[0,-1] - self.f_phase * self.fs[i] * self.tau - (self.f_phase * self.fs[i] * self.tau).conj().T
                    else:
                        for m in range(1, N): W[0,-1] -= self.fs[i]*(np.linalg.matrix_power(self.tau, m))
                if i == (self.L-1):
                    if not Fendley:
                        W[0,-1] = W[0,-1] - (self.omega**(q) * self.f_phase * self.fs[i+1] * self.tau
                                             + (self.omega**(q) * self.f_phase * self.fs[i+1] * self.tau).conj().T)
                    else:
                        for m in range(1, N):
                            W[0,-1] -= self.fs[i]*(np.linalg.matrix_power(self.omega**(q)*self.tau, m))
                if not Fendley:
                    W[1,-1] = -self.fs[i]*self.f_phase*self.tau
                    W[2,-1] = -self.fs[i]*(self.f_phase*self.tau).conj().T
                else:
                    for m in range(1, N):
                        W[m,-1] = -self.fs[i]*np.linalg.matrix_power(self.tau, m)
                W[-1,-1] = self.I
                Ws[i] = W


        self.shape = (N**self.L, N**self.L) # for mat vec routine
        self.Lp = np.zeros(self.chi, dtype=dtype); self.Lp[0] = 1.0
        self.Rp = np.zeros(self.chi, dtype=dtype); self.Rp[-1] = 1.0
        self.Ws = Ws


    def number_bands(self):
        """
        Find the number of bands for this chain.

        Returns
        ---------
        int
            The number of distinct bands.
        """
        if not hasattr(self, 'NbrBands'):
            if not hasattr(self, 'PartitionTable'):
                self.PartitionTable = PartitionTable(self.N, self.RealL-1)

            self.NbrBands = self.PartitionTable[-1,-1]
        return self.NbrBands


    def get_band_energies(self, partitions):
        """
        Given partitions representing bands, calculate the energy with respect to the unperturbed system.

        Parameters
        -----------
        partitions: matrix
            Matrix of partitions, where each row is a parititon.

        Returns
        --------
        float array
            Corresponding energies.
        """
        phases = np.arange(self.N, dtype=np.float64) * np.pi*2.0 / np.float(self.N)
        vals = -2.0 * self.J * np.cos(phases + self.theta)
        return np.dot(partitions, vals.T)


    def get_bands_and_energies(self, refresh=False, sort=True):
        """
        Get the paritions and corresponding energies with respect to the pair term for clean system with
        f=0

        Parameters
        -----------
        refresh: bool
            Force recalculation of energies (default=False).
        sort: bool
            Return sorted by energy (default=True).

        Returns
        --------
        int32 matrix
            Matrix where rows correspond to partitions.
        float64 array
            Corresponding energies.
        """
        if not hasattr(self, 'partitions'):
            self.partitions = np.zeros((self.number_bands(), self.N), dtype=np.int32)
            for i in range(self.number_bands()):
                self.partitions[i,:] = FindPartition(self.N, self.RealL-1, i, self.PartitionTable)

        if not hasattr(self, 'Es') or refresh:
            self.Es = self.get_band_energies(self.partitions)

        if sort:
            idxs = np.argsort(self.Es)
            self.Es = self.Es[idxs]
            self.partitions[:,:] = self.partitions[idxs,:]

        return [self.partitions, self.Es]


    def get_band_dimension(self, partition):
        """
        Give a partition for a particular band, find the corresponding dimension given
        by the product of binomial coefficients.

        Parameters
        -----------
        partition: array
            Array with number of domain walls of each type present in the band.

        Returns
        --------
        int
            The dimension of the band.
        """
        L = np.sum(partition)
        dim = 1
        for i in range(len(partition)):
            dim *= special.comb(L, partition[i])
            L -= partition[i]

        return dim


class QOp(MPO):
    """
    Class to represent the generalised parity operator for a chain (Subclass of
    ParafermionicChainOp).
    """

    def __init__(self, N, L, dtype=np.dtype('complex128')):
        """
        Constructor to setup instance.

        Parameters
        -----------
        N: int
            Dimension at each site.
        L: int
            Length of chain.
        dtype: dtype
            Datatype to use internally, default=np.dtype('complex128').
        """
        self.L = L; self.N = N;
        self.chi = 1
        self.dtype = dtype
        self.dim = N**L
        self.tau = np.diag(np.ones(N-1), k=1); self.tau[-1,0] = 1.0
        self.I = np.eye(N)

        Ws = dict() # create a dictionary to store tensors
        for i in range(L):
            W = np.zeros((self.chi,self.chi,N,N), dtype=self.dtype)
            W[0,0] = self.tau
            Ws[i] = W

        self.shape = (N**L, N**L)
        self.Ws = Ws
        self.Lp = np.ones(1)
        self.Rp = np.ones(1)


def orthogonalise(v):
    """
    Uses eigenvalue decomponsition to orthogonalise column vectors of given matrix

    Parameters
    -----------
    v: complex matrix
        Matrix where columns are states to be orthogonalised.

    Returns
    ---------
    complex matrix
        Matrix with orthogonalised states.
    """
    dim = v.shape[1]
    mat = np.zeros((dim, dim), dtype=v.dtype)
    for i in range(dim):
        for j in range(dim):
            mat[i,j] = np.vdot(v[:,j], v[:,i])
    d,e = np.linalg.eig(mat)

    w = np.zeros(v.shape, dtype=v.dtype)
    for i in range(dim):
        for j in range(dim):
            w[:,i] += np.conj(e[j,i]) * v[:,j]
        w[:,i] /= np.sqrt(np.vdot(w[:,i], w[:,i]))
    return w


def orthogonalise_degenerate_spaces(e, v, tol=1e-10):
    """
    Orthogonalise degenerate spaces.

    Parameters
    ----------
    e: float array
        Array of energy values
    v: complex matrix
        Matrix with columns corresponding to eigenstates.
    tol: float
        Tolerance by which one decides on whether two levels are degenerate (default=1e-10).

    Returns
    --------
    complex matrix
        Matrix with columns corresponding to orthogonalised eigenstates.
    """
    i = 1
    start = 0
    while i < len(e):
        if np.abs(e[i] - e[i-1]) > 1e-10:
            if (i - start) > 1:
                v[:,start:i] = orthogonalise(v[:,start:i])
            start = i
        i += 1
    if (i - start) > 1:
        v[:,start:i] = orthogonalise(v[:,start:i])
    return v


def diagonalise(v, O):
    """
    Diagonalise space w.r.t. operator O

    Parameters
    -----------
    v: complex matrix
        Matrix with columns are states.
    O: operator
        Operator to use to diagonalise given states.

    Returns
    ---------
    complex matrix
        Matrix containing diagonalised states.
    """
    dim = v.shape[1]
    mat = np.zeros((dim, dim), dtype=np.complex128)
    if isinstance(O, MPO):
        for i in range(dim):
            for j in range(dim):
                mat[i,j] = O.expectation(v[:,j], v[:,i])
    else:
        for i in range(dim):
            for j in range(dim):
                mat[i,j] = np.dot(np.dot(np.conj(v[:,j]), O), v[:,i])
    d,e = np.linalg.eig(mat)

    w = np.zeros(v.shape, dtype=v.dtype)
    for i in range(dim):
        for j in range(dim):
            w[:,i] += np.conj(e[j,i]) * v[:,j]
        w[:,i] /= np.sqrt(np.vdot(w[:,i], w[:,i]))
    return w


def diagonalise_degenerate_spaces(e, v, O, tol=1e-10):
    """
    diagonalise degenerate spaces w.r.t. operator O.

    Parameters
    ----------
    e: float array
        Array of energy values
    v: complex matrix
        Matrix with columns corresponding to eigenstates.
    O: operator object
        The operator to diagonalise space w.r.t.
    tol: float
        Tolerance by which one decides on whether two levels are degenerate (default=1e-10).

    Returns
    --------
    complex matrix
        Matrix with columns corresponding to diagonalised eigenstates.
    """
    i = 1
    start = 0
    while i < len(e):
        if np.abs(e[i] - e[i-1]) > 1e-10:
            if (i - start) > 1:
                v[:,start:i] = diagonalise(v[:,start:i], O)
            start = i
        i += 1
    if (i - start) > 1:
        v[:,start:i] = diagonalise(v[:,start:i], O)
    return v


def Diagonalise(N, L, J, theta, f, phi, k, q=-1, sigmaJ=0.0, sigmaf=0.0, Fendley=False):
    """
    High level function to diagonalise a chain model with the specified parameters.

    Parameters
    -----------
    N: int
       Dimension of system at each site.
    L: int
       Length of chain
    J: float
       Coefficient on pair term.
    theta: float
       Phase on pair term.
    f: float
       Coefficient on field term.
    phi: float
       Phase on onsite term.
    k: int
       The number of states to request from arnoldi solver.
    q: int
       The q-sector to diagonalise within ranges from 0 to N-1 (default = -1).
    sigmaJ: float
       The mean of the disorder strength on J (default=0.0).
    sigmaf: float
       The mean of the disorder strength on f (default=0.0).
    Fendley: flag
       Use the Fendley variant of the model (defulat=False).

    Returns
    --------
    float array
        Array of energy eigenvalues.
    complex array
        Array of eigenvalues of parity operator (not returned if q != -1).
    complex matrix
        Matrix with columns corresponding to eigenvectors.
    """
    H = ParafermionicChainOp(N, L, J, theta, f, phi, q=q, sigmaJ=sigmaJ, sigmaf=sigmaf, Fendley=Fendley)

    EffL = (L if (q==-1) else L-1)
    if k > ((N**EffL)-2) and k < 3000: # if asking for too many eigenvalues and matrix small enough.
        Hmat = H.fullmat()
        wH, vH = np.linalg.eig(Hmat)
    else:
        wH, vH = arp.eigsh(H, k=k, which='SA')
    idxs = np.argsort(wH)
    wH = wH[idxs]
    vH = vH[:,idxs]
    vH = orthogonalise_degenerate_spaces(wH, vH)

    if q != -1:
        return [wH, vH]
    else:
        Q = QOp(N,L)

        vH = diagonalise_degenerate_spaces(wH, vH, Q)

        wQ = np.zeros(len(wH), dtype=np.complex128)

        if isinstance(Q, MPO):
            for i in range(len(wH)):
                wQ[i] = Q.expectation(vH[:,i])
        else:
            for i in range(len(wH)):
                wQ[i] = np.dot(np.dot(np.conj(v[:,i]), Q), v[:,i])

        return [wH, wQ, vH]


def PartitionTable(BMax, MMax):
    """
    Build partition table iteratively using recursion relation P(M,B) = P(B, M-1) + P(B-1, M)

    Parameters
    ----------
    BMax: int
        Maximum number of boxes.
    MMax: int
        Number of items.

    Returns
    ---------
    int matrix
        Numbers of partitions.
    """
    P = np.zeros((BMax+1, MMax+1), dtype=np.int64) # indices run from 0 to MMax and 0 to BMax
    # first column is 1's and first row is 0's (in that order)
    P[:,0] = 1
    P[0,:] = 0
    if BMax+1 > 1:
        P[1,:] = 1

    for i in range(2, BMax+1):
        for j in range(1, MMax+1):
            P[i,j] = P[i,j-1] + P[i-1,j]

    return P


def FindPartition(B, M, idx, P=None, C=0):
    """
    Given an index along with number of particles and boxes, find the partition.

    Parameters:
    ------------
    B: int
        Number of boxes
    M: int
        Number of particles
    idx: int
        Index of partition to retrieve.
    P: partition table
        If none will generate new one.
    C: int
        This is the number of particles carried over

    Returns
    --------
    int array
        partition corresponding to given index.
    """
    if P is None:
        P = PartitionTable(B, M)

    if idx >= P[B, M]:
        return None

    if idx < P[B-1, M]:
        Partition = np.zeros(B, dtype=np.int32)
        Partition[0] = C
        Partition[1:] = FindPartition(B-1, M, idx, P)
    else:
        idx = idx - P[B-1, M]
        if M == 1:
            Partition = np.zeros(B, dtype=np.int32)
            Partition[idx] = 1 + C
        elif B == 1:
            Partition = np.zeros(B, dtype=np.int32)
            Partition[0] = M
        else:
            Partition = FindPartition(B, M-1, idx, P, C+1)

    return Partition


def FockStr(N, L, config):
    """
    Convert numeric fock config to string format.

    Parameters:
    -----------
    N: int
        Dimension of each site.
    L: int
        Length of the chain.
    config: int
        Integer representation of the fock coefficient in base N

    Returns:
    --------
    str:
        Fock config in string format.
    """
    return '0'*(L-len(np.base_repr(config, N))) + np.base_repr(config, N)


