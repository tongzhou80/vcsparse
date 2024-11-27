import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,k'), B: Tensor('k,j'), C: Tensor('i,j', 'csr')):
    return C.multiply(A @ B)

@compile(dump_code=True, full_opt=True)
def f1(A: Tensor('i,k'), B: Tensor('k,j'), C: Tensor('i,j', 'csr')):
    return C.multiply(A.T @ B)

def test():
    for N in [1000]:
        A = np.random.randn(N, N)
        B = np.random.randn(N, N)
        C = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A, B, C), C.multiply(A @ B).toarray())
        assert np.allclose(f1(A, B, C), C.multiply(A.T @ B).toarray())

    
if __name__ == '__main__':
    test()