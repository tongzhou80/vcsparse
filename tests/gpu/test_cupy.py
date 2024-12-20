
import cupy as np
import cupyx.scipy.sparse as sp
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j'), B: Tensor('i,j', 'csr')):
    return A + B

import appy
from numpy import empty, zeros, matmul, empty_like
from scipy.sparse import csr_matrix

appy.config.tensorlib = 'cupy'

@appy.jit
def _f0(A, B_indptr, B_indices, B_data, B_shape):
    __ret = np.empty((A.shape[0], A.shape[1]))
    #pragma parallel for
    for i in range(0, __ret.shape[0], 1):
        for j in range(0, __ret.shape[1], 1):
            __ret[i, j] = A[i, j]
        for __pB_i in range(B_indptr[i], B_indptr[i + 1], 1):
            j = B_indices[__pB_i]
            __ret[i, j] = __ret[i, j] + B_data[__pB_i] # target_indices: ['i', 'j']
    return __ret

def f0(A, B):
    return _f0(A, B.indptr, B.indices, B.data, B.shape)
@compile(dump_code=True, full_opt=True)
def f1(B: Tensor('i,j', 'csr')):
    return 1 + B

def test_dense_add_sparse():
    for N in [200]:
        A = np.random.randn(N, N)
        B = sp.random(N, N, density=0.01, format='csr')
        print(A+B)
        print(type(B))
        print(B.indptr)
        print(f0(A, B))
        #assert np.allclose(f0(A, B), (A + B))
        

    
if __name__ == '__main__':
    test_dense_add_sparse()