import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,k'), B: Tensor('k,j')):
    return A.T @ B

def test():
    for N in [1000]:
        A = np.random.randn(N, N)
        B = np.random.randn(N, N)
        assert np.allclose(f0(A, B), (A.T @ B))

    
if __name__ == '__main__':
    test()