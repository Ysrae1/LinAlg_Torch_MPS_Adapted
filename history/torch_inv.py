import torch
from torchinvmps import inv_mps
from b_matmal import b_matmul
import time

n = 30000 #timestep

sample_size = 2500

M_ex = torch.randn(n, n, dtype=torch.float32)

b_ex = torch.randn(n, sample_size, dtype=torch.float32)

inv_time = time.time()

M_ex_inv = inv_mps(M_ex)
    
inv_time = time.time() - inv_time
print(f"MPS 完成了 M（{n}x{n} 矩阵）求逆，花了 {inv_time} 秒，释放原矩阵以继续后续计算。")

del M_ex

# X = M_ex_inv.to('mps') @ b_ex.to('mps')

mul_time = time.time()

n1 = 1024 #step_limit
m_, k_ = M_ex_inv.shape
_, n_ = b_ex.shape
X = torch.zeros((m_, n_), device = M_ex_inv.device, dtype = M_ex_inv.dtype).to('mps')

for i in range(0, m_, n1):
    for j in range(0, n_, n1):
        for l in range(0, k_, n1):

            A_block = M_ex_inv[i:i+n1, l:l+n1]
            B_block = b_ex[l:l+n1, j:j+n1]

            X[i:i+n1, j:j+n1] += torch.matmul(A_block.to('mps'), B_block.to('mps'))

X = X.to('cpu')
mul_time = time.time() - mul_time


#print(f"MPS 进行了 M（{n}x{n} 矩阵）的求逆, 并计算了M_inv@b（{n}x{sample_size} 向量） ，花了 {tor_mps_time} 秒")

print(f"MPS 进行了 M_inv @ b（{n}x{sample_size} 向量）的计算 ，花了 {mul_time} 秒 \n")
print(f"共计 {inv_time+mul_time} 秒 (优化前)")