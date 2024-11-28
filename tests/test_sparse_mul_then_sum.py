import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return sum(A.multiply(B), 1)
    

def test():
    for N in [1000, 4000, 8000]:
        A = sp.random(N, N, density=0.01, format='csr')        
        B = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A, B), np.squeeze(A.multiply(B).sum(axis=1)))

        t0 = timeit.timeit(lambda: A.multiply(B).sum(axis=1), number=10) / 10
        t1 = timeit.timeit(lambda: f0(A, B), number=10) / 10
        print(t0, t1, f'{(t0/t1):.3f}')
    
if __name__ == '__main__':
    test()