import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True, use_sparse_output=True)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j')):
    return A * B

@compile(dump_code=True, full_opt=True, use_sparse_output=True)
def f1(A: Tensor('i,j', 'csr'), alpha):
    return A * alpha

def test_sparse_mul_dense():
    for N in [1000, 4000]:
        A = sp.random(N, N, density=0.01, format='csr')
        B = np.random.randn(N, N)
        alpha = 0.1    
        assert np.allclose(f0(A, B).toarray(), (A.multiply(B)).toarray())
        assert np.allclose(f1(A, alpha).toarray(), (A.multiply(alpha)).toarray())


if __name__ == '__main__':
    test_sparse_mul_dense()