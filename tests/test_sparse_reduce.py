import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True, use_sparse_output=True)
def f0(A: Tensor('i,j', 'csr')):
    b = sum(A, 1)
    return A / b[:, None]

def f0_py(A):
    b = A.sum(1)
    return A / b

def test():
    for N in [1000, 4000, 8000]:
        A = sp.random(N, N, density=0.01, format='csr')
        assert np.allclose(f0(A).toarray(), f0_py(A).toarray())
        
        t0 = timeit.timeit(lambda: f0_py(A), number=10) / 10
        t1 = timeit.timeit(lambda: f0(A), number=10) / 10
        print(t0, t1, f'{(t0/t1):.3f}')

    
if __name__ == '__main__':
    test()