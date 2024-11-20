import numpy as np
import timeit
from react import *

def kernel_react(A: Tensor('i,j'), B: Tensor('i,j')):
    return A + B + 1

kernel_react = compile(kernel_react, dump_code=True, trie_fuse=True, gen_numba_code=True, parallelize=True, memory_opt=True)

def kernel_py(A, B):
    return A + B + 1

n = 6000
A = np.random.random((n, n))
B = np.random.random((n, n))

print(np.allclose(kernel_py(A, B), kernel_react(A, B)))

t0 = timeit.timeit(lambda: kernel_py(A, B), number=10) / 10
t1 = timeit.timeit(lambda: kernel_react(A, B), number=10) / 10
print(t0, t1, f'{(t0/t1):.3f}')