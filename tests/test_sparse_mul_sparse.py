import scipy.sparse as sp
import numpy as np
import timeit
from react import *

def f0(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return A * B

f0 = compile(f0, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)

for N in [1000, 4000]:
    A = sp.random(N, N, density=0.01, format='csr')
    B = sp.random(N, N, density=0.01, format='csr')
    
    #print(type(A.multiply(B)), type(A * alpha))
    #print(np.allclose(A.multiply(B).toarray(), (A * B).toarray()))
    assert np.allclose(f0(A, B), (A.multiply(B)).toarray())
    
    