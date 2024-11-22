import scipy.sparse as sp
import numpy as np
import timeit
from react import *

def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    # sparse add sparse
    return A + B

def f1(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr'), C: Tensor('i,j', 'csr')):
    # add three sparse
    return A + B + C

f0 = compile(f0, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)
f1 = compile(f1, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)

for N in [1000, 4000]:
    A = sp.random(N, N, density=0.01, format='csr')
    B = sp.random(N, N, density=0.01, format='csr')
    C = sp.random(N, N, density=0.01, format='csr')
    assert np.allclose(f0(A, B), (A + B).toarray())
    assert np.allclose(f1(A, B, C), (A + B + C).toarray())
    
    