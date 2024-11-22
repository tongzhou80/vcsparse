import scipy.sparse as sp
import numpy as np
import timeit
from react import *

def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j')):
    return A + B

def f1(A: Tensor('i,j', 'csr')):
    return A + 1

f0 = compile(f0, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)
f1 = compile(f1, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)

for N in [1000, 4000]:
    A = sp.random(N, N, density=0.01, format='csr')
    B = np.random.randn(N, N)
    assert np.allclose(f0(A, B), (A + B))
    assert np.allclose(f1(A), (A.toarray() + 1))
    
    