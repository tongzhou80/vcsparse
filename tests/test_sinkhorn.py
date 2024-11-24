import scipy.sparse as sp
import numpy as np
import timeit
from react import *
from numpy import empty

@compile(dump_code=True, full_opt=True)
def f0(K: Tensor('k,i'), KT: Tensor('i,k'), x: Tensor('k,j'), c: Tensor('i,j', 'csr'), r: Tensor('k,i'), max_iter = 10):
    u = 1.0 / x
    v = c.multiply(1 / (KT @ u))
    x = (1 / r) * K @ v
    return x

def f0_py(K, KT, x, c, r, max_iter=10):
    u = 1.0 / x
    v = c.multiply(1 / (KT @ u))
    x = (1 / r) * K @ v
    return x

# def f0(K, KT, x, c, r, max_iter=10):
#     __v1 = empty((K.shape[1], x.shape[1]))
#     __v2 = empty((K.shape[1], x.shape[1]))
#     v = empty((K.shape[1], x.shape[1]))
#     __v4 = empty((K.shape[0], K.shape[1]))
#     __v5 = empty((K.shape[0], K.shape[1]))
#     x = empty((K.shape[0], x.shape[1]))
#     for i in range(0, KT.shape[0], 1):
#         for j in range(0, x.shape[1], 1):
#             __v1[i, j] = 0
#     for i in range(0, KT.shape[0], 1):
#         for k in range(0, KT.shape[1], 1):
#             for j in range(0, x.shape[1], 1):
#                 __v1[i, j] = __v1[i, j] + KT[i, k] * x[k, j] # target_indices: ['i', 'j'], value_indices: ['i', 'j']
#                 print(KT[i, k], x[k, j])
#     for i in range(0, __v1.shape[0], 1):
#         for j in range(0, __v1.shape[1], 1):
#             if __v1[i, j] == 0:
#                 print(i,j)
#                 exit(0)
#             __v2[i, j] = 1 / __v1[i, j] # target_indices: ['i', 'j'], value_indices: ['i', 'j']
#     for i in range(0, v.shape[0], 1):
#         for j in range(0, v.shape[1], 1):
#             v[i, j] = 0
#     for i in range(0, v.shape[0], 1):
#         for __pc_i in range(c.indptr[i], c.indptr[i + 1], 1):
#             j = c.indices[__pc_i]
#             v[i, j] = v[i, j] + c.data[__pc_i] * __v2[i, j] # target_indices: ['i', 'j'], value_indices: ['i', 'j']
#     for k in range(0, r.shape[0], 1):
#         for i in range(0, r.shape[1], 1):
#             __v4[k, i] = 1 / r[k, i] # target_indices: ['k', 'i'], value_indices: ['k', 'i']
#     for k in range(0, __v4.shape[0], 1):
#         for i in range(0, __v4.shape[1], 1):
#             __v5[k, i] = __v4[k, i] * K[k, i] # target_indices: ['k', 'i'], value_indices: ['k', 'i']
#     for k in range(0, __v5.shape[0], 1):
#         for j in range(0, v.shape[1], 1):
#             x[k, j] = 0
#     for k in range(0, __v5.shape[0], 1):
#         for i in range(0, __v5.shape[1], 1):
#             for j in range(0, v.shape[1], 1):
#                 x[k, j] = x[k, j] + __v5[k, i] * v[i, j] # target_indices: ['k', 'j'], value_indices: ['k', 'j']
#     return x


def test():
    NI = 100
    NJ = 120
    NK = 140

    K = np.random.randn(NK, NI)
    x = np.random.randn(NK, NJ)
    c = sp.rand(NI, NJ, density=1, format='csr')
    r = np.random.randn(NK, NI)
    #f0(K, K.T, x, c, r, max_iter=10)
    assert np.allclose(f0(K, K.T, x, c, r, max_iter=10), f0_py(K, K.T, x, c, r, max_iter=10), atol=1e-2)

    
if __name__ == '__main__':
    test()