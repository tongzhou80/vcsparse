import scipy.sparse as sp
import numpy as np
import timeit
from react import *

def f0(A: Tensor('i,k'), B: Tensor('k,j', 'csr')):
    return A @ B

f0 = compile(f0, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)

for N in [1000, 4000]:
    A = np.random.randn(N, N)
    B = sp.random(N, N, density=0.01, format='csr')
    assert np.allclose(f0(A, B), A @ B)
    
    t0 = timeit.timeit(lambda: (A @ B), number=10) / 10
    t1 = timeit.timeit(lambda: f0(A, B), number=10) / 10
    print(t0, t1, f'{(t0/t1):.3f}')
    
    