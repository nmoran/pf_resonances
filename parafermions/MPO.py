import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg.eigen.arpack as arp
import sys
from parafermions.MPS import MPS
import time

class MPO(object):
    """
    Base MPO class for a chain MPO.
    """

    def __init__(self, N, L, dtype=np.dtype('complex128'), existing=None,):
        """
        Constructor of which constructs trivial MPO with virtual bond dimension 1.

        Parameters
        -----------
        N: int
            Physical bond dimension.
        L: int
            Length of chain.
        dtype: datatype
            The datatype to use for MPO elements (default=complex128).
        existing: MPO object
            If an existing MPO object is passed in then this is copied (default=None).
        """
        if existing is None:
            self.L = L
            self.N = N
            self.chi = 1
            self.I = np.eye(N)
            self.dtype = dtype
            self.dim = N**L

            Ws = dict() # create a dictionary to store tensors
            for i in range(L):
                W = np.zeros((self.chi,self.chi,N,N), dtype=self.dtype)
                W[0,0,:,:] = self.I
                Ws[i] = W

            self.shape = (N**L, N**L) # for mat vec routine
            self.Lp = np.zeros(self.chi, dtype=dtype); self.Lp[0] = 1.0
            self.Rp = np.zeros(self.chi, dtype=dtype); self.Rp[-1] = 1.0
            self.Ws = Ws
        else:
            E = existing
            self.L = E.L
            self.N = E.N
            self.chi = E.chi # this is more the maximum chi
            self.I = np.eye(E.N)
            self.dtype = E.dtype
            self.dim = E.N**E.L

            Ws = dict() # create a dictionary to store tensors
            for i in range(E.L):
                Ws[i] = np.copy(E.Ws[i])

            self.shape = (E.N**E.L, E.N**E.L) # for mat vec routine
            self.Lp = np.copy(E.Lp)
            self.Rp = np.copy(E.Rp)
            self.Ws = Ws




    def getdim(self):
        """
        Return full physical dimension of operator.
        """
        return self.dim


    def fullmat(self):
        """
        Get the full matrix. Only for testing with small systems.
        """
        r = np.tensordot(self.Ws[self.L-1], self.Rp, axes=([1], [0]))

        for j in range(self.L-2, -1, -1):
            r = np.tensordot(self.Ws[j], r, axes=([1], [0]))

        y = np.tensordot(self.Lp, r, axes=([0],[0]))
        y = np.transpose(y, list(range(0, 2*self.L, 2)) + list(range(1, 2*self.L, 2)))
        return np.reshape(y, (self.N**self.L, self.N**self.L))


    def fullmats(self):
        """
        Get the full matrix in sparse format.
        """
        nnz = (self.L * self.chi) * self.dim
        nzis = np.zeros(nnz, dtype=np.int)
        nzjs = np.zeros(nnz, dtype=np.int)
        nzs = np.zeros(nnz, dtype=self.dtype)

        fill = 0

        i = 0
        maxnzs = 0
        while i < self.dim:
            phys_indices = self.decompose_index(i)
            phys_indices = phys_indices[::-1]

            # now finally calculate the element by multiplying matrices

            # simple way
#            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
#            for j in range(self.L-2, -1, -1):
#                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))
#            rowvec = np.tensordot(r, self.Lp, axes=([0], [0]))

            # more optimal
            l = np.tensordot(self.Lp, self.Ws[0][:, :,phys_indices[0],:], axes=([0], [0]))
            for site in range(1, int(self.L/2)):
                l = np.tensordot(l, self.Ws[site][:,:,phys_indices[site],:], axes=([site-1], [0]))

            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
            for j in range(self.L-2, int(self.L/2)-1, -1):
                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))

            rowvec = np.tensordot(l, r, axes=([int(self.L/2) - 1], [0]))

            # we can save more time by reusing p)revious partial contractions

            rowvec = np.reshape(rowvec, (self.dim))
            row_nzjs = np.nonzero(rowvec)[0]
            nnzjs = len(row_nzjs)
            if nnzjs > maxnzs:
                maxnzs = nnzjs
            if nnzjs > 0:
                if (fill + nnzjs) >= nnz:
                    print('Oh no, more non zeros than anticipated.')
                    return None
                s = slice(fill, fill + nnzjs, 1)
                nzjs[s] = row_nzjs
                nzis[s] = i
                nzs[s] = rowvec[nzjs[s]]
                fill += nnzjs
            i += 1

        return sps.coo_matrix((nzs[:fill], (nzis[:fill], nzjs[:fill])), shape=(self.dim, self.dim),dtype=self.dtype).tocsr()


    def mats_subspace(self, subspace):
        """
        Get the matrix of a particular subspace in sparse format.

        subspace: 2d int array
            Array of basis states of subspace.
        """
        subspace_dim = subspace.shape[0]
        subspace_indices = np.zeros(subspace_dim, dtype=np.int64)
        for i in range(subspace_dim):
            subspace_indices[i] = self.recompose_index(subspace[i,:])
        # there is a potential issue when using multiple bands that the subspace indices are not sorted.
        # we sort here and keep track of mapping.
        subspace_indices_order = np.argsort(subspace_indices)
        subspace_indices = subspace_indices[subspace_indices_order]

        #print 'subspace indices : ' + str(subspace_indices) + ' at ' + str(subspace_indices_order)

        #nnz = (self.L * self.chi) * subspace_dim
        #nzis = np.zeros(nnz, dtype=np.int)
        #nzjs = np.zeros(nnz, dtype=np.int)
        #nzs = np.zeros(nnz, dtype=self.dtype)

        nzis = []
        nzjs = []
        nzs = []

        fill = 0

        i = 0
        maxnzs = 0
        while i < subspace_dim:
            phys_indices = subspace[i,::-1]
            # more optimal
            l = np.tensordot(self.Lp, self.Ws[0][:, :,phys_indices[0],:], axes=([0], [0]))
            for site in range(1, self.L//2):
                l = np.tensordot(l, self.Ws[site][:,:,phys_indices[site],:], axes=([site-1], [0]))

            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
            for j in range(self.L-2, self.L//2-1, -1):
                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))

            # we try to create sparse matrices for l and r and then perform tensor operation as sparse matrix multiplication.
            ld = len(l.shape)
            l = np.transpose(l, axes=list(range(ld-2))+[ld-1, ld-2]) # permute last two indices
            l = np.reshape(l, (np.prod(l.shape[:-1]), l.shape[-1]))
            ls = sps.csr_matrix(l)
            r = np.reshape(r, (r.shape[0], np.prod(r.shape[1:])))
            rs = sps.csr_matrix(r)
            rowvec = ls * rs

            rows, cols = rowvec.nonzero()
            full_col_idxs = rows*rowvec.shape[1] + cols

            # this is only working of all columns are in the subspace
            #pos_idxs = np.searchsorted(subspace_indices, full_col_idxs) # find positions in subspace indices where full col idx belong
            #pos_idxs = pos_idxs[np.where(pos_idxs < len(subspace_indices))]
            #matching_idxs = np.where(subspace_indices[pos_idxs] == full_col_idxs) # find indices which actually match
            #js = subspace_indices_order[pos_idxs[matching_idxs]]

            matching_idxs = []
            matching_cols = []
            for k in range(len(full_col_idxs)):
                pos_idx = np.searchsorted(subspace_indices, full_col_idxs[k])
                if pos_idx < len(subspace_indices) and subspace_indices[pos_idx] == full_col_idxs[k]:
                    matching_idxs.append(k)
                    matching_cols.append(pos_idx)
            if len(matching_idxs) > 0:
                js = subspace_indices_order[np.asarray(matching_cols)]
                matching_idxs = np.asarray(matching_idxs)

                nzis.append(np.ones(len(js))*i)
                nzjs.append(js)
                nzs.append(rowvec.data[matching_idxs])


#            # this is the bottle neck currently as this is very large and full of zeros.
#            rowvec = np.tensordot(l, r, axes=([self.L/2 - 1], [0]))
#
#            # we can save more time by reusing previous partial contractions
#            rowvec = np.reshape(rowvec, (self.dim))
#            row_nzjs = np.nonzero(rowvec)[0]
#            nnzjs = len(row_nzjs)
#            nzvals = rowvec[row_nzjs]
#            #print 'row ' + str(i) + ' has nonzeros ' + str(nzvals) + ' at cols ' + str(row_nzjs)
#            sj = 0
#            rj = 0
#            for j in range(nnzjs):
#                while sj < subspace_dim and row_nzjs[j] > subspace_indices[sj]:
#                    sj += 1
#                if sj < subspace_dim and row_nzjs[j] == subspace_indices[sj]:
#                    #print 'found'
#                    row_nzjs[rj] = subspace_indices_order[sj]
#                    nzvals[rj] = nzvals[j]
#                    rj += 1
#            nnzjs = rj
#            #print 'row ' + str(i) + ' has nonzeros ' + str(nzvals) + ' at cols ' + str(row_nzjs)
#            if nnzjs > maxnzs:
#                maxnzs = nnzjs
#            if nnzjs > 0:
#                if (fill + nnzjs) >= nnz:
#                    print('Oh no, more non zeros than anticipated.')
#                    return None
#                s = slice(fill, fill + nnzjs, 1)
#                nzjs[s] = row_nzjs[:nnzjs]
#                nzis[s] = i
#                nzs[s] = nzvals[:nnzjs]
#                fill += nnzjs
            i += 1

        #print(str(map(lambda x: x.shape, nzs)))
        #print(str(map(lambda x: x.shape, nzis)))
        #print(str(map(lambda x: x.shape, nzjs)))
        nzs = np.hstack(nzs)
        nzis = np.hstack(nzis)
        nzjs = np.hstack(nzjs)
        #print('Sizes: ' + str(nzs.shape) + ', ' + str(nzis.shape) + ', ' + str(nzjs.shape))
        return sps.coo_matrix((nzs, (nzis, nzjs)), shape=(subspace_dim, subspace_dim), dtype=self.dtype).tocsr()


    def fullmat_petsc_sparse(self):
        """
        Get the full matrix as a sparse petsc matrix. If run
        in parallel (using MPI), the matrix will be distributed equally among
        processes. The petsc4py package and a working petsc installation is required for this.
        """
        try:
            import petsc4py.PETSc as PETSc
        except ImportError as importError:
            print("Problem importing petsc, make sure petsc4py is installed and working.")
            return None

        #First we create the matrix and zero all the elements.
        mat = PETSc.Mat()
        mat.create()
        mat.setSizes([self.dim, self.dim])
        mat.setUp()
        rs, re = mat.getOwnershipRange()
        mat.destroy()

        mat = PETSc.Mat()
        mat.create()
        mat.setSizes([self.dim, self.dim])

        Py_dnnz = np.ones(re - rs, dtype=np.int32)  # Diagonal non zeros
        Py_odnnz = np.zeros(re - rs, dtype=np.int32) # off diagonal non zeros

        i = rs
        while i < re:
            phys_indices = self.decompose_index(i)
            phys_indices = phys_indices[::-1]

            # more optimal
            l = np.tensordot(self.Lp, self.Ws[0][:, :,phys_indices[0],:], axes=([0], [0]))
            for site in range(1, int(self.L/2)):
                l = np.tensordot(l, self.Ws[site][:,:,phys_indices[site],:], axes=([site-1], [0]))
            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
            for j in range(self.L-2, int(self.L/2)-1, -1):
                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))

            rowvec = np.tensordot(l, r, axes=([int(self.L/2) - 1], [0]))

            # we can save more time by reusing previous partial contractions
            rowvec = np.reshape(rowvec, (self.dim))
            row_nzjs = np.nonzero(rowvec)[0]
            Py_dnnz[i - rs] += len(np.where((row_nzjs >= rs) & (row_nzjs < re)))
            Py_odnnz[i - rs] += len(np.where((row_nzjs < rs) & (row_nzjs >= re)))
            nnzjs = len(row_nzjs)
            i += 1
        mat.setPreallocationNNZ((Py_dnnz,Py_odnnz))
        mat.setUp()

        i = rs
        while i < re:
            phys_indices = self.decompose_index(i)
            phys_indices = phys_indices[::-1]

            # more optimal
            l = np.tensordot(self.Lp, self.Ws[0][:, :,phys_indices[0],:], axes=([0], [0]))
            for site in range(1, int(self.L/2)):
                l = np.tensordot(l, self.Ws[site][:,:,phys_indices[site],:], axes=([site-1], [0]))

            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
            for j in range(self.L-2, int(self.L/2)-1, -1):
                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))

            rowvec = np.tensordot(l, r, axes=([int(self.L/2) - 1], [0]))

            # TODO: we can save more time by reusing previous partial contractions
            rowvec = np.reshape(rowvec, (self.dim))
            row_nzjs = np.nonzero(rowvec)[0]
            nnzjs = len(row_nzjs)
            row_nzs = rowvec[row_nzjs]
            mat.setValuesIJV(np.arange(nnzjs + 1, dtype=np.int32), np.asarray(row_nzjs, dtype=np.int32), row_nzs, rowmap=i*np.ones(nnzjs, dtype=np.int32))
            i += 1
        mat.assemble(assembly=PETSc.Mat.AssemblyType.FINAL)

        return mat


    def fullmat_petsc_dense(self):
        """
        Get the full matrix as a dense petsc matrix. If run
        in parallel (using MPI), the matrix will be distributed equally among
        processes. The petsc4py package and working petsc installation is required for this.
        """
        try:
            import petsc4py.PETSc as PETSc
        except ImportError as importError:
            print("Problem importing petsc, make sure petsc4py is installed and working.")
            return None

        #First we create the matrix and zero all the elements.
        mat = PETSc.Mat()
        mat.createDense(self.dim)
        mat.zeroEntries()
        mat.setUp()
        rs, re = mat.getOwnershipRange()

        i = rs
        while i < re:
            phys_indices = self.decompose_index(i)
            phys_indices = phys_indices[::-1]

            # now finally calculate the element by multiplying matrices

            # simple way
#            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
#            for j in range(self.L-2, -1, -1):
#                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))
#            rowvec = np.tensordot(r, self.Lp, axes=([0], [0]))

            # more optimal
            l = np.tensordot(self.Lp, self.Ws[0][:, :,phys_indices[0],:], axes=([0], [0]))
            for site in range(1, int(self.L/2)):
                l = np.tensordot(l, self.Ws[site][:,:,phys_indices[site],:], axes=([site-1], [0]))

            r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
            for j in range(self.L-2, int(self.L/2)-1, -1):
                r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))

            rowvec = np.tensordot(l, r, axes=([int(self.L/2) - 1], [0]))

            # we can save more time by reusing previous partial contractions
            rowvec = np.reshape(rowvec, (self.dim))
            row_nzjs = np.nonzero(rowvec)[0]
            nnzjs = len(row_nzjs)
            row_nzs = rowvec[row_nzjs]
            mat.setValuesIJV(np.arange(nnzjs + 1, dtype=np.int32), np.asarray(row_nzjs, dtype=np.int32), row_nzs, rowmap=i*np.ones(nnzjs, dtype=np.int32))
            i += 1
        mat.assemble()

        return mat


    def sparse_vector_detangle(self, v, n):
        fulld = v.shape[0]

        def split_index(base, l, idx):
            s = np.base_repr(idx, base)
            if len(s) < 2*l:
                s = '0'*(2*l-len(s)) + s
            a = np.int(s[::2], base)
            b = np.int(s[1::2], base)
            return [a,b]

        flat_indices = v.indices
        nnz = len(v.indices)
        rows = np.zeros(nnz)
        cols = np.zeros(nnz)

        for j in range(nnz):
            [rows[j],cols[j]] = split_index(self.N, self.L, flat_indices[j])

        return sps.csr_matrix((v.data, (rows, cols)), shape=(n,n))


    def fullmats2(self):
        """
        Get the full matrix in sparse format.
        """
        chi = self.chi
        L = self.L
        N = self.N
        l = np.tensordot(self.Lp, self.Ws[0], axes=([0], [0]))
        l = np.reshape(l, (l.shape[0], N * N))
        l = sps.csr_matrix(l).T

        for i in range(1, int(L/2)):
            W = np.transpose(self.Ws[i], (0, 2, 3, 1))
            W = np.reshape(W, (W.shape[0], N * N * W.shape[3]))
            W = sps.csr_matrix(W)
            l = l * W
            l = reshape(l, (N**((i+1)*2), self.Ws[i].shape[1]))

        r = np.tensordot(self.Rp, self.Ws[L-1], axes=([0], [1]))
        r = np.reshape(r, (r.shape[0], N * N))
        r = sps.csr_matrix(r)

        for i in range(L-2, int(L/2) - 1, -1):
            W = np.transpose(self.Ws[i], (0, 2, 3, 1))
            W = np.reshape(W, (W.shape[0] * N * N, W.shape[3]))
            W = sps.csr_matrix(W)
            r = W * r
            r = reshape(r, (self.Ws[i].shape[0], N**((L-i)*2)))

        l = l * r
        l = reshape(l, (1, self.dim**2))

        return [l, self.sparse_vector_detangle(l, self.dim)]


    def matvec(self, x):
        """
        Multiply the vector x by the operator.

        Parameters
        -----------
        complex array
            Vector to apply operator to.

        Returns
        ---------
        complex array
            Result of matrix vector multiplication.
        """
        x = np.reshape(x, [self.N]*self.L + [1])
        y = np.reshape(self.Rp, (1, self.Rp.shape[0]))
        x = np.tensordot(y, x, axes=([0], [self.L]))

        for j in range(self.L-1, -1, -1):
            x = np.tensordot(self.Ws[j], x, axes=([1, 2], [0, self.L]))

        x = np.tensordot(x, self.Lp, axes=([0], [0]))
        x = np.reshape(x, (self.N**self.L))

        return x


    def expectation(self, x, y=None):
        """
        Calculate the expectation value of the given vector for the operator, (note: assumes normalised states).

        Parameters
        -----------
        x: complex array
            State to calculate expectation value in.
        y: complex array
            If given is other side of expectation value expression.

        Returns
        --------
        complex
            expectation value
        """
        z = self.matvec(x)
        if y is None:
            return np.vdot(z,x)
        else:
            return np.vdot(z,y)


    def getrow(self, i):
        """
        Get all elements in the specified row of the full matrix.

        Parameters
        ----------
        i: int
            Row index.

        Returns
        ---------
        complex/double array
            Array of the matrix elements on the given row.
        """
        if i < 0 or i >= self.dim:
            print('Row index ' + str(i) + ' is outside matrix dimensions.')
            sys.exit(1)

        phys_indices = self.decompose_index(i)
        phys_indices = phys_indices[::-1]

        # more optimal
        l = np.tensordot(self.Lp, self.Ws[0][:, :,phys_indices[0],:], axes=([0], [0]))
        for site in range(1, int(self.L/2)):
            l = np.tensordot(l, self.Ws[site][:,:,phys_indices[site],:], axes=([site-1], [0]))

        r = np.tensordot(self.Rp, self.Ws[self.L-1][:, :,phys_indices[self.L-1],:], axes=([0], [1]))
        for j in range(self.L-2, int(self.L/2)-1, -1):
            r = np.tensordot(self.Ws[j][:,:,phys_indices[j],:], r, axes=([1], [0]))

        rowvec = np.tensordot(l, r, axes=([int(self.L/2) - 1], [0]))

        rowvec = np.reshape(rowvec, (self.dim))

        return rowvec


    def draw(self, precision=1e-15):
        """
        Show graphical representation of structure of full matrix. Only applicable to small systems.

        Parameters
        -----------
        precision: float
            The magnitude below which entries are considered zero (default=1e-15).
        """
        import matplotlib.pyplot as plt
        get_ipython().magic(u'matplotlib inline')

        plt.figure()
        plt.spy(self.fullmat())


    def matrix_element(self, i, j):
        """
        Get the matrix element for row i column j.

        Parameters
        -----------
        i: int
            Row index
        j: int
            Column index

        Returns
        --------
        complex
            matrix element
        """
        i_str = np.base_repr(i, self.N)
        j_str = np.base_repr(j, self.N)

        # must pad out to be L characters long
        i_str = '0' * (self.L-len(i_str)) + i_str
        j_str = '0' * (self.L-len(j_str)) + j_str

        # reverse to make consistent with earlier numbering
        i_str = i_str[::-1]
        j_str = j_str[::-1]

        # now finally calculate the element by multiplying matrices
        r = np.dot(self.Lp, self.Ws[0][:, :,i_str[0], j_str[0]])

        for site in range(1, self.L):
            r = np.dot(r, self.Ws[site][:, :, i_str[site], j_str[site]])

        return np.dot(r, self.Rp)


    def decompose_index(self, idx):
        """
        Method to decompose an index into the clock value at each site.

        Parameters
        ------------
        idx: int
            The index.

        Returns
        --------
        array
            Clock value at each site.
        """
        idx_str = np.base_repr(idx, self.N)
        # must pad out to be L characters long
        idx_str = '0' * (self.L-len(idx_str)) + idx_str
        # reverse to make consistent with earlier numbering
        idx_str = idx_str[::-1]
        return list(map(lambda x: int(x), idx_str))


    def recompose_index(self, array):
        """
        Method to recompose the given array into a numerical indices.
        """
        idx = 0
        for i in range(len(array)):
            idx += array[i] * self.N**i
        return idx


    def matrix_element_bloch(self, i, j, iphases, jphases):
        """
        Get matrix element between given bloch elements.

        Parameters
        -------------
        i: int array
            Array of left indices.
        j: int array
            Array of right indices.
        iphases: complex array
            Array of phases for left indices.
        jphases: complex array
            Array of phases for right indices.

        Returns
        ---------
        complex
            The matrix element.
        """
        val = 0.0

        for p in range(len(i)):
            for q in range(len(j)):
                val += iphases[p].conj() * self.matrix_element(i[p], j[q]) * jphases[q]

        return val


    def Diagonalise(self, k=-1):
        """
        Diagonalise the operator.

        Parameters
        -----------
        k: int
            Number of eigenvalues to try to get (default=-1 (will try to get them all)).

        Returns
        --------
        wH: array
            Array of eigenvalues.
        vH: array
            Matrix of eigenvectors.
        """
        if k == -1:
            k = self.N**self.L # get all eigenvalues

        if k > ((self.N**self.L)-2) and k < 3000: # if asking for too many eigenvalues and matrix small enough.
            Hmat = self.fullmat()
            wH, vH = np.linalg.eig(Hmat)
        else:
            wH, vH = arp.eigsh(self, k=k, which='SA')
        idxs = np.argsort(wH)
        wH = wH[idxs]
        vH = vH[:,idxs]
        vH = orthogonalise_degenerate_spaces(wH, vH)

        return [wH, vH]


    def __mult__(self, other):
        """
        Multiply the MPO by a state object.

        Parameters
        -----------
        other: state
            State object. Only implemented for vectors thus far.
        """
        if isinstance(other, np.ndarray):
            return self.matvec(x)
        else:
            return NotImplemented


    def run_dmrg(self, sweeps=5, D=10, verbose=False, method='arpackten', **kwargs):
        """
        Run a DRMG minimisation to get the ground state MPS.

        Parameters
        ----------
        sweeps: int
            Number of sweeps.
        D: int
            Maximum virtual bond dimension.
        verbose: bool
            Flag to indicate that verbose information should be printed.
        method: str
            Which method to use for optimisation step. Options are:
            'full': Full matrix diagonalisation, very slow but reliable.
            'arpackmat': Arpack iterative but still using full matrix.
            'arpackten': Arpack iterative using tensor contraction for matvec (default).
            'lanczosten': Lanczos iterative using tensor contraction for matvec.
        """
        psi = MPS(self.L, D, self.N, dtype=self.dtype)
        psi.run_dmrg(self, sweeps, verbose, method, **kwargs)
        return psi


    def compress(self, tol=None, inplace=True):
        """
        Attempt to compress the MPO by converting to an MPS and compressing this.

        Parameters
        -----------
        tol: float
            The tolerance above which to keep singular values, if none takes MPS tolerance (default=None).
        inplace: bool
            Flag to indicate that the compression should be performed inplace (default=True).

        Returns
        --------
        MPO object
            The compressed MPO.

        """
        if inplace:
            A = self
        else:
            A = self.copy()

        MPSRep = self.convert_to_mps()
        MPSRep.compress(tol, normalise=False)
        MPO.convert_from_mps(MPSRep, A)

        return A


    def multiply(self, B=None, inplace=True, compress=True):
        """
        Multply the MPO by another MPO.

        Parameters
        ----------
        B: MPO
            The MPO object to multply by.
        compress: bool
            Flag to indicate that resulting MPO should be compressed.

        Returns
        --------
        MPO object
            The multiplied object
        """
        if inplace:
            A = self
        else:
            A = self.copy()

        if B is None:
            B = self

        if A.L != B.L:
            print('MPOs must have the same length!')
            return None

        As = list(map(lambda x: A.Ws[x].shape, range(self.L)))
        Bs = list(map(lambda x: B.Ws[x].shape, range(self.L)))

        for i in range(self.L):
            A.Ws[i] = (np.tensordot(A.Ws[i], B.Ws[i], axes=([3], [2]))
            .transpose((0, 3, 1, 4, 2, 5))
            .reshape((As[i][0]*Bs[i][0], As[i][1]*Bs[i][1], As[i][2], Bs[i][3])))

        ALp = np.expand_dims(A.Lp, axis=0)
        BLp = np.expand_dims(A.Lp, axis=0)
        A.Lp = np.tensordot(ALp, BLp, axes=([0], [0])).reshape(ALp.shape[1]*BLp.shape[1])

        ARp = np.expand_dims(A.Rp, axis=0)
        BRp = np.expand_dims(A.Rp, axis=0)
        A.Rp = np.tensordot(ARp, BRp, axes=([0], [0])).reshape(ARp.shape[1]*BRp.shape[1])

        if compress: A.compress(inplace=True)

        return A


    def copy(self):
        """
        Create a copy of this MPO object.

        Returns
        ---------
        MPO object
            A copy of the current MPO object.
        """
        return MPO(self.N, self.L, self.dtype, self)


    def convert_to_mps(self):
        """
        Convert the MPO object to an MPS by combining physical dimensions.

        Returns
        --------
        MPS object:
            New MPS object.
        """
        MPSRep = MPS(self.L, self.chi, self.N**2)
        Ws = {}
        Lp = np.reshape(self.Lp, (self.Lp.shape[0], 1))
        Ws[0] = np.tensordot(Lp, self.Ws[0], axes=([0], [0]))
        for i in range(1, self.L-1):
            Ws[i] = self.Ws[i]
        Rp = np.reshape(self.Rp, (self.Rp.shape[0], 1))
        Ws[self.L-1] = np.tensordot(Rp, self.Ws[self.L-1], axes=([0], [1])).transpose((1, 0, 2, 3))

        for site in range(self.L):
            MPSRep.M[site] = np.transpose(Ws[site], (2, 3, 0, 1)).reshape((self.N**2, Ws[site].shape[0], Ws[site].shape[1]))

        return MPSRep

    @staticmethod
    def convert_from_mps(psi, MPO_Obj=None):
        """
        Convert the MPO object to an MPS by combining physical dimensions. Assumes physical
        dimension of psi is a square.

        Parameters
        -----------
        psi: MPS object
            MPS object to convert from.
        MPO: MPO object
            If a suitable MPO object is present already, if None create one (default=None).

        Returns
        --------
        MPO object:
            MPO version of mps.
        """
        d = np.sqrt(psi.d)
        if np.abs(d % 1.0) > 1e-15:
            print('Physical dimension must be a square!')
            return None
        d = int(d)

        if MPO_Obj is None:
            A = MPO(d, psi.N, dtype=psi.dtype)
        else:
            A = MPO_Obj

        for site in range(psi.N):
            A.Ws[site] = psi.M[site].reshape((d, d, psi.M[site].shape[1], psi.M[site].shape[2])).transpose((2, 3, 0, 1))
        A.Lp = np.ones(1)
        A.Rp = np.ones(1)

        return A


    def add(self, B, c1=1.0, c2=1.0, inplace=True, compress=True):
        """
        Method to calculate add c1*A + c2*B, where A is the current MPO, B is another MPO and c1 and c2 are scale factors.

        Parameters
        ------------
        B: MPO
            MPO object to add.
        c1: scalar
            The factor to scale current MPO with before addition (default=1.0).
        c2: scalar
            The factor to scale B with before addition (default=1.0).
        inplace: bool
            Flag to indicate that the addition should be done in place.
        compress: bool
            Flag to indicate that resultant MPO should be compressed.

        Returns
        --------
        The MPO object that results from the addition.
        """
        if inplace:
            A = self
        else:
            A = self.copy()

        if A.dtype == B.dtype:
            dtype = A.dtype
        elif A.dtype == np.dtype('complex128') or B.dtype == np.dtype('complex128'):
            dtype = np.dtype('complex128') # if either are complex the use complex
            A.dtype = dtype

        if self.N != B.N:
            print("Cannot add MPSs, physical dimensions not equal! ( " + str(self.N) + " vs " + str(B.N) + ")")
            return

        if self.L != B.L:
            print("Cannot add MPSs, lengths not equal! ( " + str(self.L) + " vs " + str(B.L) + ")")
            return

        N = self.N # physical dimension (FIXME: should change to d )
        L = self.L # chain length (FIXME: should change to N )

        factor1 = np.abs(c1)**(1.0/np.float(L)) # split the magnitude out to avoid any particular tensors getting too big or too small
        phase1 = np.exp(np.angle(c1)*1j)
        factor2 = np.abs(c2)**(1.0/np.float(L)) # split the magnitude out to avoid any particular tensors getting too big or too small
        phase2 = np.exp(np.angle(c2)*1j)

        # first stitch the Lps together
        Lp = np.zeros(A.Lp.shape[0] + B.Lp.shape[0], dtype=dtype)
        Lp[:A.Lp.shape[0]] = A.Lp[:]
        Lp[A.Lp.shape[0]:] = B.Lp[:]
        A.Lp = Lp

        for i in range(A.L):
            W1 = A.Ws[i]; W2 = B.Ws[i]
            W = np.zeros((W1.shape[0] + W2.shape[0], W1.shape[1] + W2.shape[1], W1.shape[2], W1.shape[3]), dtype=dtype)
            W[:W1.shape[0], :W1.shape[1], :, :] = (phase1 if i == 0 else 1.0) * factor1 * W1[:]
            W[W1.shape[0]:, W1.shape[1]:, :, :] = (phase2 if i == 0 else 1.0) * factor2 * W2[:]
            A.Ws[i] = W

        # first stitch the Rps together
        Rp = np.zeros(A.Rp.shape[0] + B.Rp.shape[0], dtype=dtype)
        Rp[:A.Rp.shape[0]] = A.Rp[:]
        Rp[A.Rp.shape[0]:] = B.Rp[:]
        A.Rp = Rp

        if compress:
            A.compress()

        return A


    def dagger(self, inplace = True):
        """
        Method to get the conjugate of the MPO.

        Parameters
        -----------
        inplace: bool
            Flag to indicate that the operation should be performed inplace (default=True).

        Returns
        -------
        MPO
            The conjugated operator.
        """
        if inplace:
            A = self
        else:
            A = self.copy()

        for site in range(A.L):
            for i in range(A.Ws[site].shape[0]):
                for j in range(A.Ws[site].shape[1]):
                    A.Ws[site][i,j,:,:] = A.Ws[site][i,j,:,:].T.conj()

        return A


    def tensorsite(self, B, c1=1.0, c2=1.0, inplace=False):
        """
        Take the tensor product of this MPO with another MPO B on each site such that length stays the same but physical dimension increases.

        Parameters
        -----------
        B: MPO
            The MPO object to tensor the current MPO with (should have the same number of sites and same datatype).
        c1: scalar
            Scalar to scale current mpo by before tensoring (default=1.0).
        c2: scalar
            Scalar to scale given mpo by before tensoring (default=1.0).
        inplace: bool
            Flag to indicate that the operation should be performed inplace.

        Returns
        --------
        MPO
            The result of the tensor product operation.
        """
        if inplace:
            A = self
        else:
            A = self.copy()

        if A.dtype == B.dtype:
            dtype = A.dtype
        else:
            print("Cannot add MPSs, if datatypes differ ( " + str(self.dtype) + " vs " + str(B.dtype) + ")")
            return

        if self.L != B.L:
            print("Cannot add MPSs, lengths not equal! ( " + str(self.L) + " vs " + str(B.L) + ")")
            return

        # the physical dimension of the new MPO will be the product of the two constituent tensors
        N = A.N  * B.N # physical dimension (FIXME: should change to d )
        A.N = N
        L = self.L # chain length (FIXME: should change to N )

        factor1 = np.abs(c1)**(1.0/np.float(L)) # split the magnitude out to avoid any particular tensors getting too big or too small
        phase1 = np.exp(np.angle(c1)*1j)
        factor2 = np.abs(c2)**(1.0/np.float(L)) # split the magnitude out to avoid any particular tensors getting too big or too small
        phase2 = np.exp(np.angle(c2)*1j)


        Lp = np.tensordot(np.expand_dims(A.Lp, 0), np.expand_dims(B.Lp, 0), axes=([0],[0]))
        A.Lp = np.reshape(Lp, (Lp.shape[0]*Lp.shape[1]))

        for site in range(L):
            W = np.tensordot(np.expand_dims((phase1 if site == 0 else 1.0)*factor1*A.Ws[site], 0),
                             np.expand_dims((phase2 if site == 0 else 1.0)*factor2*B.Ws[site], 0),
                             axes=([0],[0]))
            a_dim = np.asarray(A.Ws[site].shape)
            b_dim = np.asarray(B.Ws[site].shape)
            A.Ws[site] = W.transpose((0, 4, 1, 5, 2, 6, 3, 7)).reshape((a_dim*b_dim))

        Rp = np.tensordot(np.expand_dims(A.Rp, 0), np.expand_dims(B.Rp, 0), axes=([0],[0]))
        A.Rp = np.reshape(Rp, (Rp.shape[0]*Rp.shape[1]))

        return A


    def print_dims(self):
        """
        Print dimensions of MPO tensors.
        """
        print('Left: ' + str(self.Lp.shape))
        print('Right: ' + str(self.Rp.shape))
        print(list(map(lambda x: self.Ws[x].shape, self.Ws.keys())))


    def inner_product(self, B=None, normalise=True):
        """
        Get the inner product between this MPO and a given MPO. If None given
        will get inner product with itself.

        Parameters
        -----------
        B: MPO
            MPO to get inner product with (default=None).
        normalise: bool
            Flag to indicate that innerproduct should be normalised.
        """
        A = self
        if B is None:
            B = A

        A_mps = A.convert_to_mps()
        B_mps = B.convert_to_mps()

        inner_product = A_mps.innerProduct(B_mps)

        if normalise:
            inner_product /= np.sqrt(A_mps.innerProduct())
            inner_product /= np.sqrt(B_mps.innerProduct())

        return inner_product


    def conj(self, inplace=False):
        """
        Get the conjugate of the currrent MPO.

        Parameters
        ----------
        inplace: bool
            Flag to indicate that operation should be done inplace (default=False).

        Returns
        --------
        MPO object
            Conjugated MPO.
        """
        if inplace:
            A = self
        else:
            A = self.copy()

        for site in range(A.L):
            for i in range(A.Ws[site].shape[0]):
                for j in range(A.Ws[site].shape[1]):
                    A.Ws[site][i,j,:,:] = A.Ws[site][i,j,:,:].T

        return A


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
    mat = np.zeros((dim, dim), dtype=O.dtype)
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

def reshape(a, shape):
    """Reshape the sparse matrix `a`.

    Returns a csr_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    rows = np.array(c.row, dtype=np.int64)
    cols = np.array(c.col, dtype=np.int64)

    flat_indices = np.array(ncols * rows + cols, dtype=np.int64)
    new_row, new_col = divmod(flat_indices, shape[1])

    b = sps.csr_matrix((c.data, (new_row, new_col)), shape=shape)
    return b
