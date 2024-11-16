import inspect
from react import *

def f0(alpha, x: Index('i'), y: Index('i')):
    return alpha * x + y

def f1(A: Index('i,j'), b: Index('j')):
    return relu(A + b)

def f2(alpha, A: Index('i,j')):
    return where(A < 0, alpha * A, A)


for f in [f0, f1, f2]:
    newcode = compile_from_src(inspect.getsource(f), trie_fuse=0)
    print(newcode)