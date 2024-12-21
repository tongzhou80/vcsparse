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

def test_sparse_add_sparse():
    for N in [1000, 4000]:
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
    
    