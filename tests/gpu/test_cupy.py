
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
    __ret = cupy.empty((A.shape[0], A.shape[1]))
    M, N = __ret.shape
    #pragma parallel for
    for i in range(0, M, 1):
        #pragma simd
        for j in range(0, N, 1):
            __ret[i, j] = A[i, j]
        ##pragma simd
        for __pB_i in range(B_indptr[i], B_indptr[i + 1], 1):
            j1 = B_indices[__pB_i]
            __ret[i, j1] = __ret[i, j1] + B_data[__pB_i] # target_indices: ['i', 'j']
    return __ret

def f0(A, B):
    return _f0(A, B.indptr, B.indices, B.data, B.shape)

@compile(dump_code=True, full_opt=True)
def f1(B: Tensor('i,j', 'csr')):
    return 1 + B

def test_dense_add_sparse():
    for N in [1000, 4000, 8000]:
        A = np.random.randn(N, N)
        B = sp.random(N, N, density=0.01, format='csr')
        #print(A+B)        
        #print(f0(A, B))
        assert np.allclose(f0(A, B), (A + B))

        import triton
        t0 = triton.testing.do_bench(lambda: A + B)
        t1 = triton.testing.do_bench(lambda: f0(A, B))
        print(t0, t1, t0/t1)
        

    
if __name__ == '__main__':
    test_dense_add_sparse()
