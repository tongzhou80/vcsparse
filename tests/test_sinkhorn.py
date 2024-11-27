import scipy.sparse as sp
import numpy as np
import timeit
from vcsparse import *


@compile(dump_code=True, full_opt=True, use_sparse_output=True)
def f0(K: Tensor('k,i'), KT: Tensor('i,k'), x: Tensor('k,j'), c: Tensor('i,j', 'csr'), r: Tensor('k,i'), max_iter = 1):
    it = 0
    while it < max_iter:
        u = 1.0 / x
        v = c.multiply(1 / (KT @ u))
        x = (1 / r) * K @ v
        it += 1
    return x

def f0_py(K, KT, x, c, r, max_iter=1):
    it = 0
    while it < max_iter:
        u = 1.0 / x
        v = c.multiply(1 / (KT @ u))
        x = (1 / r) * K @ v
        it += 1
    return x

def test():
    NI = 1000
    NJ = 1200
    NK = 1400

    K = np.random.randn(NK, NI)
    x = np.random.randn(NK, NJ)
    c = sp.rand(NI, NJ, density=0.01, format='csr')
    r = np.random.randn(NK, NI)
    niters = 1
    x_0 = f0(K, K.T, x.copy(), c, r, max_iter=niters)
    x_1 = f0_py(K, K.T, x.copy(), c, r, max_iter=niters)
   
    assert np.allclose(x_0, x_1)
    t0 = timeit.timeit(lambda: f0_py(K, K.T, x.copy(), c, r, max_iter=niters), number=10) / 10
    t1 = timeit.timeit(lambda: f0(K, K.T, x.copy(), c, r, max_iter=niters), number=10) / 10
    print(t0, t1, f'{(t0/t1):.3f}')

    
if __name__ == '__main__':
    test()