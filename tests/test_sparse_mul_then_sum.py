import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return sum(A.multiply(B), 1)
    

import numba
from numpy import empty, zeros, matmul, empty_like
from scipy.sparse import csr_matrix
@numba.njit(parallel=True)
def _f0(A_indptr, A_indices, A_data, A_shape, B_indptr, B_indices, B_data, B_shape):
    __ret = empty(A_shape[0])
    for i in numba.prange(0, A_shape[0], 1):
        buf1 = zeros(A_shape[1])
        for __pB_i in range(B_indptr[i], B_indptr[i + 1], 1):
            j = B_indices[__pB_i]
            buf1[j] = B_data[__pB_i] * 1 # target_indices: ['i', 'j']
        buf2 = zeros(A_shape[1])
        for __pA_i in range(A_indptr[i], A_indptr[i + 1], 1):
            j = A_indices[__pA_i]
            buf2[j] = A_data[__pA_i] * buf1[j] # target_indices: ['i', 'j']
        __scalar_0 = 0
        for j in range(A_shape[1]):
            __scalar_0 += buf2[j]
        #__scalar_0 = buf2.sum()
        __ret[i] = __scalar_0
    return __ret

def f0(A, B):
    return _f0(A.indptr, A.indices, A.data, A.shape, B.indptr, B.indices, B.data, B.shape)


def test():
    for N in [1000, 4000, 8000]:
        A = sp.random(N, N, density=0.02, format='csr')        
        B = sp.random(N, N, density=0.02, format='csr')
        assert np.allclose(f0(A, B), np.squeeze(A.multiply(B).sum(axis=1)))

        t0 = timeit.timeit(lambda: A.multiply(B).sum(axis=1), number=10) / 10
        t1 = timeit.timeit(lambda: f0(A, B), number=10) / 10
        print(t0, t1, f'{(t0/t1):.3f}')
    
if __name__ == '__main__':
    test()
