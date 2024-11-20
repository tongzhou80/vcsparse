import scipy.sparse as sp
import numpy as np
import timeit
from react import *

#@compile(dump_code=True)
def kernel_react(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return A + B

kernel_react = compile(kernel_react, dump_code=True, trie_fuse=1, gen_numba_code=True, parallelize=True)
# import numba
# from numpy import empty, zeros
# def kernel_react(A, B):
#     _B = empty((A.shape[0], A.shape[1]))
#     _A = empty((A.shape[0], A.shape[1]))
#     __ret = empty((A.shape[0], A.shape[1]))
#     for i in range(0, B.shape[0], 1):
#         for _j in range(B.indptr[i], B.indptr[i + 1], 1):
#             j = B.indices[_j]
#             _B[i, j] = B.data[_j] # indices: ['i', 'j']
#             j = A.indices[_j]
#             _A[i, j] = A.data[_j] # indices: ['i', 'j']
#         for j in range(0, A.shape[1], 1):
#             __ret[i, j] = A[i, j] + B[i, j] # indices: ['i', 'j']
#     return __ret

def kernel_py(A, B):
    return (A + B).toarray()

n = 3000
A = sp.random(n, n, density=0.01, format='csr')
B = sp.random(n, n, density=0.01, format='csr')

print(np.allclose(kernel_py(A, B), kernel_react(A, B)))

t0 = timeit.timeit(lambda: kernel_py(A, B), number=10) / 10
t1 = timeit.timeit(lambda: kernel_react(A, B), number=10) / 10
print(t0, t1, f'{(t0/t1):.3f}')
