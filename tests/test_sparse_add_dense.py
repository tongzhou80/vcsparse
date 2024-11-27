import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j')):
    return A + B

@compile(dump_code=True, full_opt=True)
def f1(A: Tensor('i,j', 'csr')):
    return A + 1

def test_sparse_add_dense():
    for N in [1000, 4000]:
        A = sp.random(N, N, density=0.01, format='csr')
        B = np.random.randn(N, N)
        assert np.allclose(f0(A, B), (A + B))
        assert np.allclose(f1(A), (A.toarray() + 1))


if __name__ == '__main__':
    test_sparse_add_dense()