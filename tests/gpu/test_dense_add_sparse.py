import torch

torch.set_default_device('cuda')

def sparse_randn(shape, density, device='cuda'):
    """
    Creates a random sparse CSR matrix with a given shape and density on the specified device.
    
    Args:
        shape (tuple): Shape of the sparse matrix (rows, cols).
        density (float): Fraction of non-zero elements (0 < density <= 1).
        device (str): Target device ('cuda' for GPU or 'cpu').
    
    Returns:
        torch.Tensor: Sparse CSR matrix on the specified device.
    """
    rows, cols = shape
    total_elements = rows * cols
    num_nonzeros = int(total_elements * density)

    # Randomly generate row and column indices
    row_indices = torch.randint(0, rows, (num_nonzeros,), device=device)
    col_indices = torch.randint(0, cols, (num_nonzeros,), device=device)
    
    # Sort row indices to ensure CSR format correctness
    sorted_indices = torch.argsort(row_indices)
    row_indices = row_indices[sorted_indices]
    col_indices = col_indices[sorted_indices]

    # Generate random values for the non-zero elements
    values = torch.randn(num_nonzeros, device=device)

    # Build the crow_indices for CSR format
    crow_indices = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    crow_indices.scatter_add_(0, row_indices + 1, torch.ones(num_nonzeros, dtype=torch.int32, device=device))
    crow_indices = torch.cumsum(crow_indices, dim=0)

    # Create the sparse CSR tensor
    sparse_csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(rows, cols), device=device)

    return sparse_csr_tensor


def test_dense_add_sparse():
    for density in [0.01, 0.04]:
        for N in [1000, 4000]:
            # Gen a dense matrix A of size N x N, and a sparse matrix B given density
            A = torch.randn(N, N)
            B = sparse_randn((N, N), density)
            print(B)
            print(A+B)

    
if __name__ == '__main__':
    test_dense_add_sparse()
