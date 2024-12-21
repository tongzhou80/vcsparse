import cupy as np
import cupyx.scipy.sparse as sp
import triton
from vcsparse import *

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,k'), B: Tensor('k,j', 'csr')):
    return A @ B

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