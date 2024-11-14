import inspect
from react import *

def f0(a: Index('i,j')):
    s = sum(a, axis=1)
    b = a / s[:, None]
    return b

def f1(b: Index('i,j'), c: Index('i,j'), d: Index('i,k'), e: Index('k,j')):
    a = (b + c) * (d @ e)
    return a


for f in [f0, f1]:
    tree = compile_from_src(inspect.getsource(f))
    print(ast_to_code(tree))