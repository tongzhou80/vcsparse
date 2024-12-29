import cupy as np
import cupyx.scipy.sparse as sp
import timeit
from vcsparse import *

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,k'), B: Tensor('k,j')):
    return A @ B

def test():
    for N in [1000]:
        A = np.random.randn(N, N)
        B = np.random.randn(N, N)
        assert np.allclose(f0(A, B), A @ B)
        
        import triton
        t0 = triton.testing.do_bench(lambda: (A @ B))
        t1 = triton.testing.do_bench(lambda: f0(A, B))
        print(t0, t1, f'{(t0/t1):.3f}')


if __name__ == '__main__':
    test()