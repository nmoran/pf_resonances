import numpy as np

def  NextPermutation(A, n):
    """
    Given an array of values, return the next permutation up until the array is descending in value.
    Algorithm works as follows:
    1. Start from the last item n. If this is less than the previous item n-1, proceed to n-1 and repeat this step.
    2. If item n is greater than item n-1. Swap n and n-1 and return new array.
    """
    p = n-1
    while p >= 0 and A[p+1] <= A[p]:
        p -= 1

    if p < 0:
        A[0] = -1
    else:
        q = n
        while q > p and A[q] <= A[p]:
            q -= 1

        tmp = A[q]
        A[q] = A[p]
        A[p] = tmp

        p = p+1
        while p < n:
            tmp = A[n]
            A[n] = A[p]
            A[p] = tmp
            p += 1
            n -= 1

    return A


def GetFullBand(partition, q=None):
    """
    Given a partition, expand to get all configurations in the band.
    If q is specified, we return configurations which are eigenstates of the Q operator
    with eigenvalue q.

    Parameters
    ------------
    partition: int array
        Array of integers giving the number of each type of boundary wall.
    q: int
        The Q sector to use. Should be between 0 and N-1. If None, gives all all sectors.

    Returns
    --------
    integer matrix:
        Each element corresponds to a fock coefficent. Rows give each basis element where column entries
        are the fock configurations that make up that superposition.
    complex matrix:
        Each element is the phase the corresponding fock configuration comes with.
    """
    #get representative transitions in ascending order
    B = np.int64(len(partition))
    M = np.sum(partition, dtype=np.int64)
#     print ('(B,M): ' + str((B,M)))

    t = np.zeros(M, dtype=np.int32)
    config_list = list()
    phase_list = list()

    for i in range(B):
        if i < (B-1):
            t[np.sum(partition[:i]):np.sum(partition[:i+1])] = i
        else:
            t[np.sum(partition[:i]):np.sum(partition)] = i

    config = np.zeros(M+1, dtype=np.int32)

    powers = np.arange(M, -1, -1, dtype=np.int64)
    powers = B**powers

    if q is None:
        config_row = np.zeros(1, dtype=np.int64)
        phase_row = np.ones(1, dtype=np.complex128)
        norm = 1.0
    else:
        config_row = np.zeros(B, dtype=np.int64)
        phase_row = np.zeros(B, dtype=np.complex128)
        phases = np.exp(np.complex(0.0, 1.0) *  2.0*np.pi * np.arange(B)/np.float(B))
        norm = 1.0/np.sqrt(np.float(B))

    while t[0] != -1:
        if q is None:
            for s in range(B):
                config[0] = s
                for i in range(M):
                    config[i+1] = (t[-(i+1)] + config[i]) % B
                config_row = np.dot(config, powers)
                config_list.append(config_row.copy())
                phase_list.append(phase_row.copy())
        else:
            for s in range(B):
                config[0] = s
                for i in range(M):
                    config[i+1] = (t[-(i+1)] + config[i]) % B
                config_row[s] = np.dot(config, powers)
            phase_row = norm * phases**q
            config_list.append(config_row.copy())
            phase_list.append(phase_row.copy())

        t = NextPermutation(t, M-1)

    return [np.vstack(config_list), np.vstack(phase_list)]
