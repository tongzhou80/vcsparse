import cupy as np
import cupyx.scipy.sparse as sp
import triton
from vcsparse import *

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,k'), B: Tensor('k,j', 'csr')):
    return A @ B

import appy
from cupy import empty, zeros, matmul, empty_like
from cupyx.scipy.sparse import csr_matrix
appy.config.tensorlib = 'cupy'
@appy.jit
def _f0(A, B_indptr, B_indices, B_data, B_shape):
    __ret = empty((A.shape[0], B_shape[1]))
    B_shape_1 = B_shape[1]
    A_shape_0 = A.shape[0]
    A_shape_1 = A.shape[1]
    #pragma parallel for
    for i in range(0, A_shape_0, 1):
        #pragma simd
        for j in range(0, B_shape_1, 1):
            __ret[i, j] = 0

        for k in range(0, A_shape_1, 1):
            ##pragma simd
            for __pB_k in range(B_indptr[k], B_indptr[k + 1], 1):
                j = B_indices[__pB_k]
                __ret[i, j] = __ret[i, j] + A[i, k] * B_data[__pB_k] # target_indices: ['i', 'j']
    return __ret

def f0(A, B):
    return _f0(A, B.indptr, B.indices, B.data, B.shape)

def test_dense_matmul_sparse():
    for N in [1000, 4000]:
        A = np.random.randn(N, N)
        B = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A, B), A @ B)
        
        t0 = triton.testing.do_bench(lambda: (A @ B))
        t1 = triton.testing.do_bench(lambda: f0(A, B))
        print(t0, t1, f'{(t0/t1):.3f}')


if __name__ == '__main__':
    test_dense_matmul_sparse()