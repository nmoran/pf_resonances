
"""MPS.py: Class definition for Matrix Product State object."""

import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import sys
import time
import scipy.sparse as sps
import scipy as sp

def dag(x):
    return np.transpose(np.conj(x))


class MatrixWrapper(object):
    """
    Class the wraps a given matrix object and provided an interface that can
    contains a matvec function which can be used with standard solvers.
    """
    def __init__(self, A, dtype=float):
        self.A = A
        self.dtype = dtype
        self.shape = A.shape

    def matvec(self,x):
        x=np.tensordot(self.A, x, axes=([1],[0]))
        if(self.dtype==float):
            return np.real(x)
        else:
            return(x)


class MatrixWrapperTwoSiteOp(object):
    """
    Class that provides the multiplication of the two site MPS tensors
    with the surrounding tensors. Takes the two site tensors resized into
    a vector and resizes them again at the end.
    """
    def __init__(self, Lp, Rp, W1, W2, dtype=float):
        """
        Initialise the tensors

        Parameters
        -----------
        Lp: tensor
            This is  the tensor representing full contraction to the left.
        Rp: tensor
            This is  the tensor representing full contraction to the right.
        W1: tensor
            This is  the tensor of MPO on left.
        W2: tensor
            This is  the tensor of MPO on left.
        dtype: dtype
            The datatype to use (default is float).
        """
        self.Lp = Lp
        self.Rp = Rp
        self.W1 = W1
        self.W2 = W2
        self.dim = (self.W1.shape[2]* self.Lp.shape[0]* self.W2.shape[2]* self.Rp.shape[0])
        self.shape = (self.dim, self.dim)
        self.dtype = dtype


    def matvec(self,x):
        """
        The method that performs the contraction.

        Parameters
        ----------
        x: vector
            Vector representing the two contracted tensors with dim (d*chi*d*chi)

        Returns
        vector
            The result of contracting x with surrounding tensors.
        """
        x=x.reshape((self.W1.shape[2], self.Lp.shape[0], self.W2.shape[2], self.Rp.shape[0]))
        x=np.tensordot(self.Lp, x, axes=([0], [1]))
        x=np.tensordot(x, self.W1, axes=([0,2], [0,2]))
        x=np.tensordot(x, self.W2, axes=([3,1], [0,2]))
        x=np.tensordot(x, self.Rp, axes=([1,3], [0,1])).transpose((1,0,2,3))
        x=x.reshape((self.dim))
        if(self.dtype==float):
            return np.real(x)
        else:
            return(x)



class MPS(object):
    """
    Class to represent matrix product state objects.
    """

    def __init__(self, N, chi, d, conf=None, existing=None, dtype=np.float64, tolerance=1e-12):
        """
        Initialise MPS object with given dimensions.

        Parameters
        -----------
        N : int
            Number of sites
        chi : int
            Maximum virtual bond dimension
        d : int
            Physical dimension
        conf: str
            Starting configuration. if not none then mps will have bond dimension 1.
            tensors will be filled with random elements up to bond dimension d.
        existing: mps obj
            An existing mps object to copy if not none.
        dtype: numpy datatype
            Data type to use for elements. default is np.float64.
        tolerance: float
            The tolerance to use when truncating singular values.
        """
        self.N = N; self.d = d; self.chi = chi
        self.dtype = dtype
        self.tolerance = tolerance
        self.Rps = None
        self.Lps = None

        if conf is None:
            conf = np.zeros(N, dtype=np.int32)

        if existing is None:
            self.M = dict() # store tensors at each site
            for i in range(0, N):
                self.M[i] = np.zeros((d, 1, 1), dtype=dtype)
                self.M[i][conf[i], 0, 0] = 1.0
            self.s = dict() # store singular values on each bond.
            for i in range(0, N-1):
                self.s[i] = np.ones((1, 1), dtype=dtype)
        elif isinstance(existing, MPS):
            E = existing
            self.N = E.N
            self.d = E.d
            self.dtype = E.dtype
            self.M = dict()
            self.s = dict()
            for i in E.M.keys():
                self.M[i] = np.copy(E.M[i])
            for i in E.s.keys():
                self.s[i] = np.copy(E.s[i])


    def innerProduct(self, B=None):
        """
        Function to calculate the inner product with the given tensor. if none then inner product with self is given
        which corresponds to norm. naive method that does not take into account the canonical normalisation of mps's.

        Parameters
        --------------
        B : MPS object
            MPS to take inner product with.
        """
        M1 = self.M
        N = self.N
        if B is None:
            M2 = self.M
        else:
            M2 = B.M

        T = np.tensordot(M1[0], M2[0].conj(), axes=([0,1],[0,1]))
        for i in range(1, N):
            T = np.tensordot(T, M1[i], axes=(0, 1))
            T = np.tensordot(T, M2[i].conj(), axes=([0, 1], [1,0]))
        T = np.reshape(T, (1))
        return T[0]


    def normalise(self):
        """
        function to normalise the mps.
        """
        M = self.M
        N = self.N

        norm = np.sqrt(self.innerProduct(self))**(1.0/np.float(N))
        for i in range(N):
             M[i] = M[i] / norm


    def expectationValue(self, Op, normalise=True):
        """
        Calculate the expectation value of the given operator. we contract the network with the mpo sandwiched between the MPS and
        it's conjugate transpose. we go from left to right.

        Op: MPO object
            MPO of the operator to calculate the expectation value of.
        normalise: bool
            Flag to indicate that the expectation value should be normalised.
        """
        M = self.M
        W = Op.Ws
        N = self.N

        Lp = Op.Lp.reshape(1, Op.Lp.shape[0], 1)
        for i in range(N):
            Lp = np.tensordot(Lp, M[i], axes=([0], [1]))
            Lp = np.tensordot(Lp, W[i], axes=([0, 2], [0, 2]))
            Lp = np.tensordot(Lp, M[i].conj(), axes=([0, 3], [1, 0]))

        Rp = Op.Rp.reshape(1, Op.Rp.shape[0], 1)
        Lp = np.tensordot(Lp, Rp, axes=([0, 1, 2], [0, 1, 2]))
        if normalise:
            Lp /= self.innerProduct()
        return Lp


    def get_two_site_matrix(self, Op, site):
        """
        Given contracted tensors to left and right and MPO tensors on each
        site, return the two site matrix that will be optmised.

        Parameters
        ----------
        Op: MPO
            Operator that to calculate matrix with respect to.
        site: int
            The index of the left of the two sites being updated.

        Returns
        --------
        matrix
            contracted matrix.
        """
        A = np.tensordot(self.Lps[site-1], Op.Ws[site], axes=([1],[0]))  # contract left virtual bond of left mpo tensor and left side.
        A = np.tensordot(A, Op.Ws[site+1], axes=([2],[0]))  # contract with left bond of right mpo.
        A = np.tensordot(A, self.Rps[site+2], axes=([4], [1])) # contract right virtual bond of right mpo tensor with right side.
        A = A.transpose((2, 0, 4, 6, 3, 1, 5, 7)) # transpose so indices are ordered with upper: physl, virtl, physr, virtr and lower: physl, virtl, physr, virtr
        A = A.reshape((self.d**2 * self.Lps[site-1].shape[0] * self.Rps[site+2].shape[0], self.d**2 * self.Lps[site-1].shape[0] * self.Rps[site+2].shape[0]))

        return A


    def two_site_update(self, site, direction, Op, method='lanczosten', **kwargs):
        """
        For two site optimisation approach update tensors when on specified site from specified direction.

        Parameters
        ----------
        site: int
            The index of the site (the left site of two site update in range 0 to N-2).
        direction: char
            Character indicating the direction of sweep out of 'r', 'l'.
        Op: MPO
            MPO operator that we are performing update with resepect to.
        method: str
            Which method to use for optimisation step. Options are:
            'full': Full matrix diagonalisation, very slow but reliable.
            'arpackmat': Arpack iterative but still using full matrix.
            'arpackten': Arpack iterative using tensor contraction for matvec.
            'lanczosten': Custom Lanczos tensor contraction for matvec with default ncv=20 (default).
        """
        M = self.M
        W = Op.Ws
        Lp = self.Lps[site-1]
        Rp = self.Rps[site+2]
        s = self.s

        if method == 'lanczosten':
            Mat = MatrixWrapperTwoSiteOp(Lp, Rp, W[site], W[site+1], dtype=self.M[0].dtype)
            # prepare initial vector
            v0 = np.tensordot(M[site], M[site+1], axes=([2], [1])).reshape((self.d**2 * Lp.shape[0] * Rp.shape[0]))
            [d, gs] = Lanczos(Mat.matvec, ((20 if Mat.shape[0] > 20 else Mat.shape[0]) if 'ncv' not in kwargs else kwargs['ncv']), v0, dtype=self.dtype, return_eigenvectors=True)
            gs = gs.conj()
        elif method == 'arpackten':
            Mat = MatrixWrapperTwoSiteOp(Lp, Rp, W[site], W[site+1], dtype=self.M[0].dtype)
            # prepare initial vector
            v0 = np.tensordot(M[site], M[site+1], axes=([2], [1])).reshape((self.d**2 * Lp.shape[0] * Rp.shape[0]))
            try:
                [d, gs] = arp.eigsh(Mat, k=1, which='SA', return_eigenvectors=True, v0=v0,
                                    maxiter=(271 if 'maxiter' not in kwargs else kwargs['maxiter']),
                                    ncv=((3 if Mat.shape[0] > 3 else Mat.shape[0]) if 'ncv' not in kwargs else kwargs['ncv']),
                                    tol=(0 if 'tol' not in kwargs else kwargs['tol'])
                                    )
                gs = gs[:,0].conj()
            except arp.ArpackNoConvergence as NoConvergence:
                print("broken:")
                print(str(NoConvergence))
                sys.exit(1)
        elif method == 'arpackmat':
            A = self.get_two_site_matrix(Op, site)
            Mat = MatrixWrapper(A, dtype=self.M[0].dtype)
            # prepare initial vector
            v0 = np.tensordot(M[site], M[site+1], axes=([2], [1])).reshape((self.d**2 * Lp.shape[0] * Rp.shape[0]))
            try:
                [d, gs] = arp.eigsh(Mat, k=1, which='SA', return_eigenvectors=True, v0=v0)
                gs = gs[:,0]
            except arp.ArpackNoConvergence as NoConvergence:
                print("broken:")
                print(str(NoConvergence))
                sys.exit(1)
        elif method == 'full':
            A = self.get_two_site_matrix(Op, site)
            d, v  = np.linalg.eig(A)
            gs = v[:, np.argsort(d)[0]]
        else:
            print('Method must be "full", "arpackmat", "arpackten" or "lanczosten".')
            sys.exit(1)

        B = gs.reshape((self.d * Lp.shape[0], self.d * Rp.shape[0]))
        B = B.conj()

        [U, S, V] = np.linalg.svd(B, full_matrices=False)
        V = V.T

        K = min(len(S[S > self.tolerance]), self.chi)
        S = S[:K]/np.sqrt(np.sum(S[:K]**2))
        s[site] = np.diag(S)
        U = np.reshape(U[:,:K], (self.d, Lp.shape[0], K))
        V = np.reshape(V[:,:K], (self.d, Rp.shape[0], K)).transpose((0, 2, 1))

        M[site] = U
        if site > 0:
            M[site] = np.tensordot(M[site], np.diag(np.diag(s[site-1])**(-1)), axes=([1],[1])) # divide in previous singular values
            M[site] = M[site].transpose((0, 2, 1))
        if site < self.N-1:
            M[site] = np.tensordot(M[site], s[site], axes=([2],[0])) # multiply new singular values in
        M[site+1] = V

        if direction == 'R':
            Lp = np.tensordot(Lp, M[site], axes=([0], [1]))
            Lp = np.tensordot(Lp, W[site], axes=([0,2],[0, 2]))
            Lp = np.tensordot(Lp, M[site].conj(), axes=([0,3], [1,0]))
            self.Lps[site] = np.copy(Lp)
        elif direction == 'L':
            Rp = np.tensordot(Rp, M[site+1], axes=(0, 2))
            Rp = np.tensordot(Rp, W[site+1], axes=([0, 2], [1, 2]))
            Rp = np.tensordot(Rp, M[site+1].conj(), axes=([0, 3], [2, 0]))
            self.Rps[site+1] = np.copy(Rp)


    def minimise(self, Op, method='lanczosten', new=True, **kwargs):
        """
        Function that performs sweeps and optimises the tensor to minimise the expectation
        value of of the mps for the given operator.

        Parameters
        ------------
        Op: MPO objct
            MPO to minimise mps with relation to.
        Rps: list of tensors
            List of contractions of runs from previous iteraction to save time
        method: str
            Which method to use for optimisation step. Options are:
            'full': Full matrix diagonalisation, very slow but reliable.
            'arpackmat': Arpack iterative but still using full matrix.
            'arpackten': Arpack iterative using tensor contraction for matvec (default).
        new: bool
            Flag to indicate this is a new minimisation and to recalculate everything (default=True).
        """
        M = self.M; W = Op.Ws; N = self.N

        # contract from the right and save each contraction
        # if we have this saved from previously, then do not do it again.
        if self.Rps is None or new:
            Rp = np.reshape(Op.Rp, (1, W[N-1].shape[1], 1))
            Rps = dict()
            Rps[N] = Rp
            self.Rps = Rps

            for i in range(N-1, 0, -1):
                A = M[i]
                Rp = np.tensordot(Rp, A, axes=([0, 2]))
                Rp = np.tensordot(Rp, W[i], axes=([0, 2], [1, 2]))
                Rp = np.tensordot(Rp, A.conj(), axes=([0, 3], [2, 0]))
                Rps[i] = np.copy(Rp)
        else:
            Rps = self.Rps

        if self.Lps is None or new:
            Lp = np.reshape(Op.Lp, (1, W[N-1].shape[1], 1))
            Lps = dict()
            Lps[-1] = Lp
            self.Lps = Lps
        else:
            Lps = self.Lps

        # now we are ready to sweep right, optimising two rungs at a time.
        for i in range(N-1): # loop over bonds
            self.two_site_update(i, 'R', Op, method, **kwargs)

        # now we sweep back to the left updating the rps as we go.
        for i in range(N-2, -1, -1): # loop over bonds
            self.two_site_update(i, 'L', Op, method, **kwargs)


    def get_entanglement_entropy(self):
        """
        Get entanglment entropy at each link.

        Returns
        -------
        vector
            vector of entanglement entropy values on each link.
        """
        ees = np.zeros(len(self.s))
        for i in range(len(self.s)):
            s = np.diag(self.s[i])
            ees[i] = np.sum(-(s**2)*np.log(s**2))

        return ees


    def get_entanglement_normal(self):
        """
        Get entanglment entropy at each link.

        Returns
        -------
        vector
            Vector of entanglement entropy values on each link.
        """
        ens = np.zeros(len(self.s))
        for i in range(len(self.s)):
            s = np.diag(self.s[i])
            ens[i] = np.sum(s**2)

        return ens


    def get_entanglement_spectra(self):
        """
        Get entanglment spectral at each link.

        Returns
        -------
        list
            List of entanglement spectra values at each link.
        """
        ess = []
        for i in range(len(self.s)):
            s = np.sort(np.diag(self.s[i]))
            ess.append(-2*s*np.log(s[s!=0]))

        return ess


    def add(self, B, c=1.0, inplace=True):
        """
        Add another MPS to this MPS.

        Parameters
        -----------
        B: mps object
            The object to add.
        c: scalar
            Amount to multiply by before adding.
        inplace: bool
            Flag to indicate whether addition should be done inplace or not (default=True).

        Returns
        -------
        MPS
            Result of addition of the two MPSs.
        """
        if self.d != B.d:
            print("Cannot add MPSs, physical dimensions not equal! ( " + str(self.d) + " vs " + str(B.d) + ")")
            return

        if self.N != B.N:
            print("Cannot add MPSs, lengths not equal! ( " + str(self.N) + " vs " + str(B.N) + ")")
            return

        N = self.N

        if inplace:
            A = self
        else:
            A = self.copy()

        factor = np.abs(c)**(1.0/np.float(N)) # split the magnitude out to avoid any particular tensors getting too big or too small
        angle = np.angle(c)
        phase = np.exp(angle*1j)

        M1 = self.M[0]; M2 = B.M[0]
        C = np.zeros((A.d, 1, M1.shape[2] + M2.shape[2]), dtype=A.dtype)
        for j in range(A.d):
            C[j, 0, :M1.shape[2]] = M1[j,0,:]
            C[j, 0,  M1.shape[2]:] = phase * factor*M2[j,0,:]
        A.M[0] = C

        for i in range(1, A.N-1):
            M1 = A.M[i]; M2 = B.M[i]
            C = np.zeros((A.d, M1.shape[1] + M2.shape[1], M1.shape[2] + M2.shape[2]), dtype=A.dtype)
            for j in range(A.d):
                C[j, :M1.shape[1], :M1.shape[2]] = M1[j,:,:]
                C[j, M1.shape[1]:, M1.shape[2]:] = factor*M2[j,:,:]
            A.M[i] = C
            s1 = A.s[i]
            s2 = B.s[i]
            s = np.diag(np.concatenate((np.diag(s1), np.diag(s2))))
            A.s[i] = s

        M1 = A.M[N-1]; M2 = B.M[N-1]
        C = np.zeros((A.d, M1.shape[1] + M2.shape[1], 1), dtype=A.dtype)
        for j in range(A.d):
            C[j, :M1.shape[1], 0] = M1[j,:,0]
            C[j, M1.shape[1]:,0] = factor*M2[j,:,0]
        A.M[N-1] = C

        return A


    def multiply(self, Op, inplace=True, compress=False):
        """
        Method to multiply the MPS by the given operator.

        Parameters
        ------------
        Op: MPO operator
            The operator to multiply by.
        inplace: boolean
            Flag to indicate whether to perform this operation inplace or
            create a new MPS and return this (default=True).
        compress: boolean
            Flag to indicate that resulting MPS should be compressed (default=False).

        Returns
        --------
        MPS object
            Result of multiplying the MPS by an MPO.
        """
        N = self.N
        if inplace:
            new_state = self
        else:
            new_state = self.copy()

        M = new_state.M
        Ws = dict()

        Lp = np.reshape(Op.Lp, (Op.Lp.shape[0], 1))
        Ws[0] = np.tensordot(Lp, Op.Ws[0], axes=([0], [0]))
        for i in range(1, N-1):
            Ws[i] = Op.Ws[i]
        Rp = np.reshape(Op.Rp, (Op.Rp.shape[0], 1))
        Ws[N-1] = np.tensordot(Rp, Op.Ws[N-1], axes=([0], [1])).transpose((1, 0, 2, 3))

        for i in range(N):
            M[i] = np.tensordot(M[i], Ws[i], axes=([0], [2])).transpose((4, 0, 2, 1, 3)).reshape((self.d, M[i].shape[1]*Ws[i].shape[0], M[i].shape[2]*Ws[i].shape[1]))

        if compress:
            new_state.compress(inplace=True)

        return new_state


    def compress(self, tol=None, normalise=True, inplace=True, max_chi=None):
        """
        Attempt to compress the MPS by performing SVDs at each bond and truncating
        to singular values greater than the given tolerance.

        Parameters
        -----------
        tol: float
            The tolerance above which to keep singular values, if none takes MPS tolerance (default=None).

        Returns
        ---------
        MPS object
            The compressed MPS object.
        """
        if tol is None:
            tol = self.tolerance

        if inplace:
            new_state = self
        else:
            new_state = self.copy()

        M = new_state.M
        s = new_state.s

        for site in range(self.N-1):
            W = np.tensordot(M[site], M[site+1], axes=([2], [1])).reshape((self.d*M[site].shape[1], self.d*M[site+1].shape[2]))
            U, S, V = np.linalg.svd(W, full_matrices=False)
            V = V.T

            SNorm = S/np.sqrt(np.sum(S**2))
            if max_chi:
                K = min(len(SNorm[SNorm > tol]), max_chi)
            else:
                K = len(SNorm[SNorm > tol])
            S = S[:K]
            if normalise: S = S/np.sqrt(np.sum(S**2))
            s[site] = np.diag(S)
            U = np.reshape(U[:,:K], (self.d, M[site].shape[1], K))
            V = np.reshape(V[:,:K], (self.d, M[site+1].shape[2], K)).transpose((0, 2, 1))

            M[site] = U
            M[site+1] = np.tensordot(V, s[site], axes=([1], [0])).transpose((0, 2, 1))


        for site in range(self.N-2, -1, -1):
            W = np.tensordot(M[site], M[site+1], axes=([2], [1])).reshape((self.d*M[site].shape[1], self.d*M[site+1].shape[2]))
            U, S, V = np.linalg.svd(W, full_matrices=False)
            V = V.T

            SNorm = S/np.sqrt(np.sum(S**2))
            if max_chi:
                K = min(len(SNorm[SNorm > tol]), max_chi)
            else:
                K = len(SNorm[SNorm > tol])
            S = S[:K]
            if normalise: S = S/np.sqrt(np.sum(S**2))
            s[site] = np.diag(S)
            U = np.reshape(U[:,:K], (self.d, M[site].shape[1], K))
            V = np.reshape(V[:,:K], (self.d, M[site+1].shape[2], K)).transpose((0, 2, 1))

            M[site] = np.tensordot(U, s[site], axes=([2], [0]))
            M[site+1] = V

        return new_state


    def copy(self):
        """
        Method to copy the current MPS object.

        Returns
        --------
        MPS object
            A copy of the current MPS object.
        """
        return MPS(self.N, self.chi, self.d, existing=self, dtype=self.dtype)


    def residual(self, Op):
        """
        Method to calculate the residual of the given MPS. Should be 0 if the state is
        an eigenstate of the given operator.

        Parameters
        ----------
        Op: MPO object
            Operator to get residual with respect to.

        Returns
        ----------
        float
            The residual of the MPS with the given MPO.
        """
        psi2 = self.multiply(Op, inplace=False)
        psi2.normalise()
        return np.abs(1.0 - np.abs(self.innerProduct(psi2)))


    def run_dmrg(self, Op, sweeps=5, verbose=False, method='arpackten', D=None, **kwargs):
        """
        Run a DRMG minimisation to get the ground state MPS.

        Parameters
        ----------
        Op: MPO
            The MPO object to perform DMRG on.
        sweeps: int
            Number of sweeps.
        verbose: bool
            Flag to indicate that verbose information should be printed.
        method: str
            Which method to use for optimisation step. Options are:
            'full': Full matrix diagonalisation, very slow but reliable.
            'arpackmat': Arpack iterative but still using full matrix.
            'arpackten': Arpack iterative using tensor contraction for matvec (default).
            'lanczosten': Lanczos iterative using tensor contraction for matvec.
        D: int
            The maximum virtual bond dimension, None to use existing (default is None).
        """
        if D is not None:
            self.chi = D
        if verbose: print('Starting energy: ' + str(self.expectationValue(Op)))
        for sweep in range(sweeps):
            s = time.time()
            self.minimise(Op, method, **kwargs)
            e = time.time()
            if verbose: print('Sweep ' + str(sweep) + ' energy: ' + str(self.expectationValue(Op))
                              + ', residual: ' + str(self.residual(Op))
                              + ', max bond: ' + str(np.max(list(map(lambda x: self.M[x].shape[1], self.M.keys()))))
                              + ', time: ' + str(e-s) + 's.')


    def print_dims(self):
        """
        Print dimensions of MPS tensors.
        """
        print(list(map(lambda x: self.M[x].shape, self.M.keys())))


def Lanczos(A,k,v,dtype=np.float,return_eigenvectors=True, low_mem=False):
    """
    Lanczos implementation from wikipedia pseudocode

    Parameters
    ----------
    A: function ref
        Function which multiplies matrix
    k: int
        Number of requiested eigenvalues
    v: vector
        The starting vector
    eigenvector: bool
        Flag to indicate whether to calculate eigenvectors or not (default=True).
    low_mem: bool
        Flag to indicate that keeping the memory requirements low is a priority
        and eigenvectors will be calculated in a second pass rather than storing
        intermediate vectors (default=False).

    Returns
    --------
    float
        Lowest eigenvalue
    vector
        Lowest eigenvector
    """
    n = len(v)
    if k > n: k = n
    # find elements of tri-diagonal matrix
    v0 = np.copy(v)/np.linalg.norm(v)
    alpha = np.zeros(k, dtype=dtype)  #diagonal
    beta = np.zeros(k, dtype=dtype)   #offdiagonal
    save_vectors = not low_mem and return_eigenvectors
    if save_vectors:
        vs = np.zeros((n,k), dtype=dtype)
        vs[:,0] = v0
    for j in range(k-1):
        omega = A(v0)
        alpha[j] = np.dot(omega.conj().T, v0)
        omega = omega -alpha[j]*v0 - (0.0 if j == 0 else beta[j]*v1)
        beta[j+1] = np.linalg.norm(omega)
        v1 = (omega if beta[j+1] == 0 else omega/beta[j+1]) # avoid divide by zero
        if save_vectors: vs[:,j+1] = v1
        v0, v1 = v1, v0
    omega = A(v0)
    alpha[k-1] = np.dot(omega.conj().T,v0)

    # Get lowest eigenvalue of tri-diagonal matrix
    T = np.vstack((beta, alpha))
    w, u = sp.linalg.eig_banded(T, select='i', select_range=(0,0))

    if not return_eigenvectors: # Don't calculate the eigenvector, just return.
        return w
    elif return_eigenvectors and low_mem: # Calculate eigenvectors by stepping back through everything.
        v0 = np.copy(v)/np.linalg.norm(v)
        f = np.zeros(n, dtype=dtype)
        for j in range(k-1):
            f += u[j] * v0
            omega = A(v0)
            omega = omega -alpha[j]*v0 - (0.0 if j == 0 else beta[j]*v1)
            v1 = (omega if beta[j+1] == 0 else omega/beta[j+1]) # avoid divide by zero
            v0, v1 = v1, v0
        f += u[k-1] * v0
        return [w, f]
    else:  # Calculate eigenvectors with saved intermediate vectors.
        f = np.zeros(n, dtype=dtype)
        for j in range(k):
            f += u[j] * vs[:,j]
        return [w, f]

