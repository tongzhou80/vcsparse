import torch

@compile(dump_code=True, full_opt=True, backend='appy')
def f0(A: Tensor('i,j'), B: Tensor('i,j', 'csr')):
    return A + B

def test_dense_add_sparse():
    for density in [0.01, 0.04]:
        for N in [1000, 4000]:
            # Gen a dense matrix A of size N x N, and a sparse matrix B given density
            A = torch.randn(N, N)
            B = torch.sparse.rand(N, N, density=density)
            print(A+B)
            print(f0(A, B))

    
if __name__ == '__main__':
    test_dense_add_sparse()