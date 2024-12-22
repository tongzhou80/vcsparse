import cupy as np
import cupyx.scipy.sparse as sp
import triton
from vcsparse import *

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    # sparse add sparse
    return A + B

@compile(dump_code=True, full_opt=True, backend='appy')
def f1(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr'), C: Tensor('i,j', 'csr')):
    # add three sparse
    return A + B + C

import appy
from cupy import empty, zeros, matmul, empty_like
from cupyx.scipy.sparse import csr_matrix
appy.config.tensorlib = 'cupy'
@appy.jit
def _f1(A_indptr, A_indices, A_data, A_shape, B_indptr, B_indices, B_data, B_shape, C_indptr, C_indices, C_data, C_shape):
    __dB = empty((A_shape[0], A_shape[1]))
    __v2 = empty((A_shape[0], A_shape[1]))
    __ret = empty((A_shape[0], A_shape[1]))
    __dB_shape_1 = __dB.shape[1]
    __dB_shape_0 = __dB.shape[0]
    __v2_shape_1 = __v2.shape[1]
    __v2_shape_0 = __v2.shape[0]
    __ret_shape_1 = __ret.shape[1]
    __ret_shape_0 = __ret.shape[0]
    #pragma parallel for
    for i in range(0, __dB_shape_0, 1):
        #pragma simd
        for j in range(0, __dB_shape_1, 1):
            __dB[i, j] = 0
        #pragma simd(128)
        for __pB_i in range(B_indptr[i], B_indptr[i + 1], 1):
            j1 = B_indices[__pB_i]
            __dB[i, j1] = __dB[i, j1] + B_data[__pB_i] * 1 # target_indices: ['i', 'j']
        #pragma simd
        for j in range(0, __v2_shape_1, 1):
            __v2[i, j] = __dB[i, j]
        #pragma simd(128)
        for __pA_i in range(A_indptr[i], A_indptr[i + 1], 1):
            j2 = A_indices[__pA_i]
            __v2[i, j2] = __v2[i, j2] + A_data[__pA_i] # target_indices: ['i', 'j']
        #pragma simd
        for j in range(0, __ret_shape_1, 1):
            __ret[i, j] = __v2[i, j]
        #pragma simd(128)
        for __pC_i in range(C_indptr[i], C_indptr[i + 1], 1):
            j3 = C_indices[__pC_i]
            __ret[i, j3] = __ret[i, j3] + C_data[__pC_i] # target_indices: ['i', 'j']
    return __ret

def f1(A, B, C):
    return _f1(A.indptr, A.indices, A.data, A.shape, B.indptr, B.indices, B.data, B.shape, C.indptr, C.indices, C.data, C.shape)

def test_sparse_add_sparse():
    for N in [1000, 2000, 4000, 8000]:
        A = sp.random(N, N, density=0.01, format='csr')
        B = sp.random(N, N, density=0.01, format='csr')
        C = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A, B), (A + B).toarray())
        assert np.allclose(f1(A, B, C), (A + B + C).toarray())

        t0 = triton.testing.do_bench(lambda: (A + B + C))
        t1 = triton.testing.do_bench(lambda: f1(A, B, C))
        print(t0, t1, f'{(t0/t1):.3f}')


if __name__ == '__main__':
    test_sparse_add_sparse()
    
    