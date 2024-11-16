import inspect
from react import *

def f0(alpha, x: Index('i'), y: Index('i')):
    return alpha * x + y

def f1(A: Index('i,j'), b: Index('j')):
    return relu(A + b)


for f in [f0, f1]:
    newcode = compile_from_src(inspect.getsource(f))
    print(newcode)