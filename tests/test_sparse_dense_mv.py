import scipy.sparse as sp
import numpy as np
import timeit
from react import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('j')):
    return A @ B

def test():
    for N in [1000, 4000]:
        A = sp.random(N, N, density=0.01, format='csr')
        B = np.random.randn(N)
        assert np.allclose(f0(A, B), A @ B)
    
if __name__ == '__main__':
    test()