import ParafermionUtils as pf
import scalapack_wrapper as scw
import mpi4py.MPI as MPI
import numpy as np

bg = None

for L in range(3,7):
    print("L: " + str(L))
    H = pf.ParafermionicChainOp(3, L, 1.0, 0.0, 0.1, 0)

    bg, w = scw.full_parallel_diag(H, bs=8, debug=False, timing=True, once_off=False, bg=bg)

    [w_exact, v_exact] = H.Diagonalise()

    Comm = MPI.COMM_WORLD
    my_rank = Comm.Get_rank()

    Comm.Barrier()

    if my_rank == 0:
        print('Scalapack: ' + str(w))
        print('Exact: ' + str(w_exact))

    #if my_rank == 0:
    #    my_mat = H.fullmat()
    #    print(my_mat)
    #    my_mat_pad = np.zeros((6,6), dtype=np.complex128)
    #    my_mat_pad[:4, :4] = my_mat
    #    w,v = np.linalg.eig(my_mat_pad)
    #    print(w)
