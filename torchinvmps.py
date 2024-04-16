import torch

def inv_mps(M, m=1024):

    # start_time = time.time()
    
    n = M.shape[0]
    M = M.to('cpu').to(dtype=torch.float32)

    if n <= m:
        return torch.linalg.inv(M.to('mps')).to('cpu')
    else:
        M_inv = torch.zeros_like(M)
        A_inv = inv_mps(M[:m,:m])
        c = M[m:,:m]
        b = M[:m,m:]
        _A = M[m:,m:] - c @ A_inv @ b
        _A_inv = inv_mps(_A)
        C = - (_A_inv)@c@A_inv
        B = - A_inv@b@_A_inv

        M_inv[:m,:m] = A_inv
        M_inv[m:,:m] = C
        M_inv[:m,m:] = B
        M_inv[m:,m:] = _A_inv
        
        return M_inv
    
    
