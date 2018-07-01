import numpy as np

from parafermions.MPO import MPO

class PeschelEmerySpinHalf(MPO):
    """
    Class which contains method for treating Peschel Emery line of parameters
    """

    def __init__(self, L, l, dtype=np.dtype('complex128')):
        """
        Constructor of spin half Peschel Emergy chain operator class with the given parameters.

        Parameters
        -----------
        L: int
            Length of chain.
        l: float
            The sinlge parameter of the Peschel Emery spin half line
        dtype: dtype
            Datatype to use internally, default=np.dtype('complex128').
        """
        self.L = L   # otherwise full hamiltonian
        self.l = l
        self.N = 2

        # setting dimension and datatype
        self.dim = self.N**self.L
        self.dtype = dtype
        self.chi = 4

        self.sigmax = np.zeros((2, 2), dtype=self.dtype)
        self.sigmax[0, 1] = 1; self.sigmax[1, 0] = 1

        self.sigmaz = np.zeros((2, 2), dtype=self.dtype)
        self.sigmaz[0, 0] = 1; self.sigmaz[1, 1] = -1

        self.I = np.eye(2, dtype=self.dtype)
        self.Ul = (np.cosh(self.l) - 1)/2.0
        self.hl = np.sinh(l)

        c_i = lambda x: 2 if x > 0 and x < self.L-1 else 1

        Ws = dict() # create a dictionary to store tensors
        for i in range(self.L):
            W = np.zeros((self.chi,self.chi,self.N,self.N), dtype=self.dtype)
            W[0,0:4] = [self.I, self.sigmax, self.sigmaz, c_i(i)/2*(self.hl*self.sigmaz + self.I *(self.Ul + 1))]
            W[1,-1] = -self.sigmax
            W[2,-1] = self.Ul * self.sigmaz
            W[-1,-1] = self.I
            Ws[i] = W

        self.shape = (self.N**self.L, self.N**self.L) # for mat vec routine
        self.Lp = np.zeros(self.chi, dtype=dtype); self.Lp[0] = 1.0
        self.Rp = np.zeros(self.chi, dtype=dtype); self.Rp[-1] = 1.0
        self.Ws = Ws


class ParitySpinHalf(MPO):
    """
    Parity operator for spin half chains
    """

    def __init__(self, L, dtype=np.dtype('complex128')):
        """
        Parity operator for spin half chains

        Parameters
        -----------
        L: int
            Length of chain.
        dtype: dtype
            Datatype to use internally, default=np.dtype('complex128').
        """
        self.L = L   # otherwise full hamiltonian
        self.N = 2

        # setting dimension and datatype
        self.dim = self.N**self.L
        self.dtype = dtype
        self.chi = 1

        self.sigmaz = np.zeros((2, 2), dtype=self.dtype)
        self.sigmaz[0, 0] = 1; self.sigmaz[1, 1] = -1

        self.I = np.eye(2, dtype=self.dtype)

        Ws = dict() # create a dictionary to store tensors
        for i in range(self.L):
            W = np.zeros((self.chi,self.chi,self.N,self.N), dtype=self.dtype)
            W[0,0] = self.sigmaz
            Ws[i] = W

        self.shape = (self.N**self.L, self.N**self.L) # for mat vec routine
        self.Lp = np.zeros(self.chi, dtype=dtype); self.Lp[0] = 1.0
        self.Rp = np.zeros(self.chi, dtype=dtype); self.Rp[-1] = 1.0
        self.Ws = Ws
