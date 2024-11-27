import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True)
def f0(A: Tensor('i,j')):
    b = sum(A, 1)
    return A / b[:, None]

def f0_py(A):
    b = np.sum(A, 1)
    return A / b[:, None]

def test():
    for N in [1000, 4000]:
        A = np.random.randn(N, N)
        assert np.allclose(f0(A), f0_py(A))
        
        t0 = timeit.timeit(lambda: f0_py(A), number=10) / 10
        t1 = timeit.timeit(lambda: f0(A), number=10) / 10
        print(t0, t1, f'{(t0/t1):.3f}')

    
if __name__ == '__main__':
    test()