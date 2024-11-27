import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,k'), B: Tensor('k,j', 'csr')):
    return A @ B

def test_dense_matmul_sparse():
    for N in [1000, 4000]:
        A = np.random.randn(N, N)
        B = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A, B), A @ B)
        
        t0 = timeit.timeit(lambda: (A @ B), number=10) / 10
        t1 = timeit.timeit(lambda: f0(A, B), number=10) / 10
        print(t0, t1, f'{(t0/t1):.3f}')


if __name__ == '__main__':
    test_dense_matmul_sparse()