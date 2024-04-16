import torch

def b_matmul(A, B, block_size=10000):
    
    m, k = A.shape
    _, n = B.shape
    C = torch.zeros((m, n), device=A.device, dtype=A.dtype)

    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for l in range(0, k, block_size):

                A_block = A[i:i+block_size, l:l+block_size]
                B_block = B[l:l+block_size, j:j+block_size]

                C[i:i+block_size, j:j+block_size] += torch.matmul(A_block, B_block)

    return C
