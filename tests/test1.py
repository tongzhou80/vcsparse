import inspect
from react import *

def f0(alpha, x: Index('i'), y: Index('i')):
    return alpha * x + y

def f1(A: Index('i,j'), b: Index('j')):
    return relu(A + b[None, :])

def f2(alpha, A: Index('i,j')):
    return where(A < 0, alpha * A, A)

def f3(A: Index('i,j')):
    b = sum(A, 1)
    return A / b[:, None]


for f in [f0, f1, f2, f3]:
    newcode = compile_from_src(inspect.getsource(f), trie_fuse=1)
    print(newcode)