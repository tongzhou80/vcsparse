import cupy as np
import cupyx.scipy.sparse as sp
import triton
from vcsparse import *

np.cuda.runtime.setDevice(1)

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,k', 'csr'), B: Tensor('k,j')):
    return A @ B

def test_sparse_matmul_dense():
    for density in [0.001, 0.003, 0.01]:
        for N in [1000, 4000, 8000]:
            A = sp.random(N, N, density=density, format='csr')
            B = np.random.randn(N, N)
            
            #print(type(A.multiply(B)), type(A * alpha))
            #print(np.allclose(A.multiply(B).toarray(), (A * B).toarray()))
            assert np.allclose(f0(A, B), A @ B)
            
            t0 = triton.testing.do_bench(lambda: A @ B)
            t1 = triton.testing.do_bench(lambda: f0(A, B))
            print(t0, t1, f'{(t0/t1):.3f}')
    

if __name__ == '__main__':
    test_sparse_matmul_dense() 