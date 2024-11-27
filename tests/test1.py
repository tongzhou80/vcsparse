import inspect
from vcsparse import *

def f0(alpha, x: Tensor('i'), y: Tensor('i')):
    return alpha * x + y

def f1(A: Tensor('i,j'), b: Tensor('j')):
    return relu(A + b[None, :])

def f2(alpha, A: Tensor('i,j')):
    return where(A < 0, alpha * A, A)

def f3(A: Tensor('i,j')):
    b = sum(A, 1)
    return A / b[:, None]

def f4(A: Tensor('i,k'), B: Tensor('k,j')):
    return matmul(A, B)

def f5(A: Tensor('i,k', 'csr')):
    b = sum(A, 1)
    return A / b[:, None]

def f6(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return A * B

for f in [f0, f2, f3, f4, f5, f6]:
    newcode = compile_from_src(inspect.getsource(f), 
                            trie_fuse=1, parallelize=1, gen_numba_code=True, 
                            memory_opt=1)
    print(newcode)