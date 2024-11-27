import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True,)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('j'), C: Tensor('j')):
    return A @ B, A @ C

def test():
    for N in [1000, 4000, 8000]:
        A = sp.random(N, N, density=0.01, format='csr')
        B = np.random.randn(N)
        C = np.random.randn(N)
        assert np.allclose(f0(A, B, C)[0], A @ B)
        t0 = timeit.timeit(lambda: (A @ B, A @ C), number=10) / 10
        t1 = timeit.timeit(lambda: f0(A, B, C), number=10) / 10
        print(t0, t1, f'{(t0/t1):.3f}')
    
if __name__ == '__main__':
    test()