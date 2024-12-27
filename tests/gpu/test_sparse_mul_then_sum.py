import cupy as np
import cupyx.scipy.sparse as sp
import triton
from vcsparse import *

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return sum(A.multiply(B), 1)
    
import appy
from cupy import empty, zeros, matmul, empty_like
from cupyx.scipy.sparse import csr_matrix
appy.config.tensorlib = 'cupy'
@appy.jit
def _f0(A_indptr, A_indices, A_data, A_shape, B_indptr, B_indices, B_data, B_shape):
    __dB = empty((A_shape[0], A_shape[1]))
    __v1 = empty((A_shape[0], A_shape[1]))
    __ret = empty(A_shape[0])
    __dB_shape_1 = __dB.shape[1]
    __dB_shape_0 = __dB.shape[0]
    __v1_shape_1 = __v1.shape[1]
    __v1_shape_0 = __v1.shape[0]
    #pragma parallel for
    for i in range(0, __dB_shape_0, 1):
        #pragma simd
        for j in range(0, __dB_shape_1, 1):
            __dB[i, j] = 0
        #pragma simd(128)
        for __pB_i in range(B_indptr[i], B_indptr[i + 1], 1):
            j2 = B_indices[__pB_i]
            __dB[i, j2] = __dB[i, j2] + B_data[__pB_i] * 1 # target_indices: ['i', 'j']
        #pragma simd
        for j3 in range(0, __v1_shape_1, 1):
            __v1[i, j3] = 0
        #pragma simd(128)
        for __pA_i in range(A_indptr[i], A_indptr[i + 1], 1):
            j4 = A_indices[__pA_i]
            __v1[i, j4] = __v1[i, j4] + A_data[__pA_i] * __dB[i, j4] # target_indices: ['i', 'j']
        __ret[i] = 0.0
        #pragma simd
        for j5 in range(0, __v1_shape_1, 1):
           __ret[i] += __v1[i, j5] # target_indices: ['i']
        #__ret[i] = __scalar_0
    return __ret

def f0(A, B):
    return _f0(A.indptr, A.indices, A.data, A.shape, B.indptr, B.indices, B.data, B.shape)

def test():
    for N in [1000, 4000, 8000]:
        # Denser inputs lead to better perf
        A = sp.random(N, N, density=0.1, format='csr')        
        B = sp.random(N, N, density=0.1, format='csr')
        assert np.allclose(f0(A, B), np.squeeze(A.multiply(B).sum(axis=1)))

        t0 = triton.testing.do_bench(lambda: A.multiply(B).sum(axis=1))
        t1 = triton.testing.do_bench(lambda: f0(A, B))
        print(t0, t1, f'{(t0/t1):.3f}')
    
if __name__ == '__main__':
    test()
