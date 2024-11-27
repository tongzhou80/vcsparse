import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return A * B

def test_sparse_mul_sparse():
    for N in [1000, 4000]:
        A = sp.random(N, N, density=0.01, format='csr')
        B = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A, B), (A.multiply(B)).toarray())
    
if __name__ == '__main__':
    test_sparse_mul_sparse()