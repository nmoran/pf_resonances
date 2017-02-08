# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False
# cython: cdivision=True

from parafermions.MPO import *
import numpy as np
cimport numpy as np
import sys
import time
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

ctypedef struct complex16:
    double dr, di

ctypedef long long int sc_int

cdef extern void Cblacs_get(sc_int context, sc_int what, sc_int *val);

cdef extern void Cblacs_gridinit(sc_int* context, char* order,
                                 sc_int nproc_rows, sc_int nproc_cols);
cdef extern void Cblacs_pcoord(sc_int context, sc_int p,
                               sc_int* my_proc_row, sc_int* my_proc_col);
cdef extern void Cblacs_exit(sc_int doneflag);

cdef extern void Cblacs_freebuff(sc_int ConTxt, sc_int Wait);

cdef extern void descinit_(sc_int* descrip, sc_int* m, sc_int* n,
                           sc_int* row_block_size, sc_int* col_block_size,
                           sc_int* first_proc_row, sc_int* first_proc_col,
                           sc_int* blacs_grid, sc_int* leading_dim,
                           sc_int* error_info);

cdef extern sc_int numroc_(sc_int* order, sc_int* block_size,
                        sc_int* my_process_row_or_col, sc_int* first_process_row_or_col,
                        sc_int* nproc_rows_or_cols);

cdef extern void pzheevd_(char *jobz, char *uplo, sc_int *n, complex16 *a, sc_int *ia, sc_int *ja, sc_int *desca, double *w, complex16 *z, sc_int *iz, sc_int *jz, sc_int *descz, complex16 *work, sc_int *lwork, double *rwork, sc_int *lrwork, sc_int *iwork, sc_int *liwork, sc_int *info);

def full_parallel_diag(mpo, rows=-1, debug=False, bs=1, timing=False, once_off=True, bg=None, eigvectors=False):
    """
    Function to calculate all eigenvalues in parallel using the scalapack pzheevd routine.

    Parameters:
    ------------
    mpo: MPO object
        The MPO object to calculate all eigenvalues for.
    rows: int
        The number of rows to have in the processor grid. -1 indicates that one should choose sqrt(nproc) (default is -1).
    debug: flag
        Flag to indicate whether debugging output should be written to files by each process (default is False).
    bs: int
        The block size to use. For large matrices a larger block size is more efficient (default is 64, but reduced if matrix is too small).
    timing: flag
        Flag to indicate that the timing for allocation of matrix elements and solving should be recorded and output.
    once_off: flag
        Flag to indicate that this is a once off calculation and the blacs grid should be destroyed afterwards (default True).
    bg: int
        This is the blacs_grid identifier if there is already a blacs grid setup. If None, one is initialised (default is None).
    eigvectors: bool
        Flag to indicate that the eigenvectors should also be returned.

    Returns:
    ------------
    bg: int
        The blacs_grid number that was used.
    eigenvalues: array of np.float64
        Array containing the eigenvalues.
    eigenvectors: The local submatrix of eigenvectors along with the row and column indices, only return if eigenvectors flag is true.

    """
    if not isinstance(mpo, MPO):
        print("Object passed in must be an instance of the MPO class.")
        sys.exit(-1)

    cdef int my_rank, size
    cdef sc_int row_block_size, col_block_size
    cdef sc_int nproc_rows, nproc_cols
    cdef sc_int my_process_row, my_process_col
    cdef sc_int blacs_grid
    cdef sc_int m
    cdef sc_int n
    cdef sc_int first_proc_row = 0, first_proc_col = 0
    cdef sc_int descrip[9]
    cdef sc_int error_info
    cdef sc_int nlocal_rows, nlocal_cols
    cdef int i,j
    cdef sc_int leading_dim
    cdef sc_int ai = 1, aj = 1, zi = 1, zj = 1
    cdef double *rwork
    cdef complex16 *work
    cdef sc_int *iwork
    cdef sc_int lwork, lrwork, liwork
    cdef complex16 *a
    cdef complex16 *z
    cdef double *w
    cdef sc_int col_idx

    import mpi4py.MPI as MPI

    Comm = MPI.COMM_WORLD
    my_rank = Comm.Get_rank()
    size = Comm.Get_size()

    if rows == -1:
        rows = int(np.sqrt(size))

    if (size % rows) != 0:
        if my_rank == 0: print('Number of processes not suitable for grid layout.')
        sys.exit(-1)

    n = mpo.dim
    m = mpo.dim
    nproc_rows = rows
    nproc_cols = int(size/rows)

    if (4 * bs) >  (n / nproc_rows) or (4 * bs) > (n/nproc_cols):
        bs = min(n/(nproc_rows*4), n/(nproc_cols))
        bs = max(1, bs)
        if my_rank == 0: print('Block size is too large. Reducing to ' + str(bs))
    row_block_size = bs
    col_block_size = bs

    if bg is None:
        Cblacs_get(0, 0, &blacs_grid)
        order = "R".encode("ascii")
        Cblacs_gridinit(&blacs_grid, order, nproc_rows, nproc_cols)
    else:
        blacs_grid = bg

    Cblacs_pcoord(blacs_grid, my_rank, &my_process_row, &my_process_col)

    nlocal_rows = numroc_(&m, &row_block_size, &my_process_row, &first_proc_row, &nproc_rows)
    nlocal_cols = numroc_(&n, &col_block_size, &my_process_col, &first_proc_col, &nproc_cols)
    leading_dim = numroc_(&m, &row_block_size, &my_process_row, &first_proc_row, &nproc_rows)

    descinit_(descrip, &m, &n, &row_block_size, &col_block_size, &first_proc_row, &first_proc_col, &blacs_grid, &leading_dim, &error_info)

    if error_info != 0:
        if my_rank == 0: print("Error setting up descriptor: " + str(error_info) + ".")
        sys.exit(-1)

    if debug:
        f = open('desc_p_' + str(my_rank) + '.dat', 'w')
        f.write('Error info: ' + str(error_info) + '\n')
        f.write('Leading dim: ' + str(leading_dim) + '\n')
        f.write('starting: row x col: ' + str(first_proc_row) + ' x ' + str(first_proc_col) + '\n')
        i = 0
        while i < 9:
            f.write(str(descrip[i]) + "\n")
            i += 1
        f.close()
    if debug: print(str(my_rank) + ": Initialising descriptor result: " + str(error_info) + ".")
    a = <complex16 *>PyMem_Malloc(nlocal_rows * nlocal_cols * sizeof(complex16))
    z = <complex16 *>PyMem_Malloc(nlocal_rows * nlocal_cols * sizeof(complex16))
    w = <double *>PyMem_Malloc(n*sizeof(double))

    row_indices = np.zeros(nlocal_rows, dtype=np.int64)
    col_indices = np.zeros(nlocal_cols, dtype=np.int64)
    i = 0
    # calculate the indices of rows that will be stored on this process
    while i < nlocal_rows:
        row_indices[i] = int(i/bs) * bs * nproc_rows + my_process_row * bs + (i % bs)
        i+=1

    s = time.time()
    i = 0
    while i < nlocal_cols:
        col_idx = int(i/bs) * bs * nproc_cols + my_process_col * bs + (i % bs) # global column index
        col_indices[i] = col_idx
        col = mpo.getrow(col_idx) # get column values (can use getrow method since the matrix is Hermitian).
        col_vals = col[row_indices]
        j = 0
        while j < nlocal_rows:
            a[i * nlocal_rows + j].dr = np.real(col_vals[j])
            a[i * nlocal_rows + j].di = np.imag(col_vals[j])
            j+=1
        i+=1
    e = time.time()

    if timing: print(str(my_rank) + ": Matrix elements assigned in " + str(e-s) + " seconds.")

    if debug:
        f = open('mat_p_' + str(my_rank) + '.dat', 'w')
        i = 0
        while i < nlocal_rows:
            j = 0
            while j < nlocal_cols:
                #f.write('(' + str(a[i*nlocal_cols + j].dr) + ',' + str(a[i*nlocal_cols + j].di) + ')\t')
                #f.write(str(a[i*nlocal_cols + j].dr) + '\t')
                f.write(str(a[j*nlocal_rows + i].dr) + '\t')
                j += 1
            f.write('\n')
            i += 1
        f.close()

    rwork = <double*>PyMem_Malloc(2 * sizeof(double))
    work = <complex16*>PyMem_Malloc(2 * sizeof(complex16))
    iwork = <sc_int*>PyMem_Malloc(2 * sizeof(sc_int))

    jobz = 'V'.encode("ascii") # 'V' is for eigenvectors and eigenvalues, 'N' for eigenvalues only not implemented yet in scalapack
    uplo = 'U'.encode("ascii")
    lwork = -1; lrwork = -1; liwork = -1
    pzheevd_(jobz, uplo, &n, a, &ai, &aj, descrip, w, z, &zi, &zj, descrip, work, &lwork, rwork, &lrwork, iwork, &liwork, &error_info)

    if debug: print(str(my_rank) + ": Found required workspace sizes.")

    lwork = int(work[0].dr)
    lrwork = int(rwork[0])
    liwork = iwork[0]

    PyMem_Free(work); PyMem_Free(iwork); PyMem_Free(rwork)

    rwork = <double*>PyMem_Malloc(lrwork * sizeof(double))
    work = <complex16*>PyMem_Malloc(lwork * sizeof(complex16))
    iwork = <sc_int*>PyMem_Malloc(liwork * sizeof(sc_int))

    s = time.time()
    pzheevd_(jobz, uplo, &n, a, &ai, &aj, descrip, w, z, &zi, &zj, descrip, work, &lwork, rwork, &lrwork, iwork, &liwork, &error_info)
    e = time.time()

    if timing:
        if my_rank == 0: print("Diagonalisation took " + str(e-s) + " seconds.")

    if error_info != 0:
        if my_rank == 0: print("Error from diagonalisation: " + str(error_info) + ".")
        sys.exit(-1)

    eigvals = np.zeros(n)
    i = 0
    while i < n:
        eigvals[i] = w[i]
        i += 1

    # if we should return eigenvectors
    if eigvectors:
        eigvectors_sub = np.zeros((nlocal_cols, nlocal_rows), dtype=np.complex128)
        i = 0
        while i < nlocal_cols:
            j = 0
            while j < nlocal_rows:
                eigvectors_sub[i, j] = z[i*nlocal_rows + j].dr + z[i*nlocal_rows + j].di*1j
                j+=1
            i+=1

    PyMem_Free(a); PyMem_Free(w); PyMem_Free(z)
    PyMem_Free(work); PyMem_Free(iwork); PyMem_Free(rwork)
    if once_off:
        Cblacs_exit(1);

    if eigvectors:
        return blacs_grid, eigvals, [eigvectors_sub.T, row_indices, col_indices]
    else:
        return blacs_grid, eigvals
