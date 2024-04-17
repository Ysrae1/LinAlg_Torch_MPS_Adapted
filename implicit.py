import torch
import time
from torchinvmps import inv_mps
from LQR_SO import LQRSol_SO


class ImMonteCarloYYB:
    def __init__(self, A, t_0, X_0, T, n, dt, sig, sample_size, comp_step_limit = 4096, device = 'mps'):

        self.A = A
        self.t_0 = t_0
        self.X_0 = X_0
        self.T = T
        self.n = n
        self.dt = dt
        self.sig = sig
        self.sample_size = sample_size
        self.device = device

        self.comp_step_limit = comp_step_limit

        self.X0s = (self.X_0.squeeze().repeat(1,self.sample_size)).reshape(len(self.X_0.squeeze()),self.sample_size)

        self.len_b = torch.tensor([1,2*n], dtype = torch.float32, device = self.device)

        diagA = torch.diagonal(self.A,dim1=-2, dim2=-1).flatten()
        self.diagAA = torch.cat((torch.ones(len(self.X_0.squeeze()), dtype = torch.float32, device = self.device), diagA))

        #byproducts

        A_subs = self.A.flip(dims=[-1])
        A_subs_f = torch.diagonal(A_subs, dim1=-2, dim2=-1).flatten()
        u_A_subs_f = A_subs_f[::2] 
        l_A_subs_f = A_subs_f[1::2] 

        A_subs_0s = torch.zeros_like(u_A_subs_f,dtype = torch.float32, device = self.device)[:-1]
        self.A_u = torch.zeros(len(self.X_0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
        self.A_l = torch.zeros(len(self.X_0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
        self.A_u[len(X0.squeeze())::2] = u_A_subs_f
        self.A_u[len(X0.squeeze())+1::2] = A_subs_0s
        self.A_l[len(X0.squeeze())::2] = l_A_subs_f
        self.A_l[len(X0.squeeze())+1::2] = A_subs_0s

        self.negsub = -torch.ones(diagA.numel(), dtype = torch.float32, device = self.device)
    
    def b_solve(self, negsub, A_u, A_l, diagAA, len_b, dt, X_0):

        def diaggen(negsub,A_u,A_l,diagAA):
            
            matrix_negsub = torch.diag(negsub, diagonal=-len(self.X_0.squeeze()))

            matrix_A_u = torch.diag(A_u, diagonal=1)
            
            matrix_A_l = torch.diag(A_l, diagonal=-1)
            
            matrix_diag = torch.diag(diagAA)
            
            AA = matrix_diag+matrix_A_l+matrix_A_u+matrix_negsub

            return AA
        
        def bgen(len_b):

            if len_b[0] == 1:

                ini = torch.zeros_like(self.X_0.squeeze().repeat(1,self.sample_size)).reshape(len(self.X_0.squeeze()),self.sample_size)

                r_1 = (self.sig.squeeze().repeat(int((len_b[1]/2 -1)))*torch.sqrt(dt).repeat_interleave(len(self.X_0.squeeze()))).unsqueeze(1)
                r_2 = torch.randn(len(self.sig.squeeze().repeat(int((len_b[1]/2 -1)))),self.sample_size,dtype=torch.float32, device = self.device)

                return torch.cat((ini,r_1*r_2),dim = 0)
            
            else:

                r_1 = (self.sig.squeeze().repeat(int(len_b[1]/2))*torch.sqrt(dt).repeat_interleave(len(self.X_0.squeeze()))).unsqueeze(1)
                r_2 = torch.randn(len(self.sig.squeeze().repeat(int(len_b[1]/2))),self.sample_size,dtype=torch.float32, device = self.device)

                return r_1*r_2
    
        dim = self.X0s.shape[0]

        N = int(len_b[1]/2)
        
        if N <= self.comp_step_limit:

            b = bgen(len_b)
            
            b[:dim] += X_0

            AA = diaggen(negsub,A_u,A_l,diagAA)

            X_0_N = (inv_mps(AA).to(self.device)@b)
            
            return X_0_N
        
        else:

            X_0_N = torch.clone(self.X0s.repeat(N,1))

            _X_0_N = self.b_solve(negsub[:2*(self.comp_step_limit-1)],
                                  A_u[:2*(self.comp_step_limit)-1],
                                  A_l[:2*(self.comp_step_limit)-1],
                                  diagAA[:2*(self.comp_step_limit)],
                                  torch.tensor([len_b[0],2*self.comp_step_limit]),
                                  dt[:self.comp_step_limit - int(len_b[0])],
                                  X_0_N[:dim])
            
            X_0_N[:2*self.comp_step_limit] = _X_0_N

            negsub = negsub[2*(self.comp_step_limit):]
            A_u = A_u[2*(self.comp_step_limit):]
            A_l = A_l[2*(self.comp_step_limit):]
            diagAA = diagAA[2*(self.comp_step_limit):]    

            dt = dt[self.comp_step_limit - int(len_b[0]):]

            len_b[0] = 0*len_b[0]
            len_b[1] -= 2*self.comp_step_limit
            
            
            X_0_N_ = X_0_N[2*(self.comp_step_limit-1):2*(self.comp_step_limit)].repeat(N,1)
            
            X_0_N[2*self.comp_step_limit:] = self.b_solve(negsub, A_u, A_l, diagAA ,len_b, dt, X_0_N_[:dim])

            return X_0_N


    
if __name__=='__main__':

    device = 'mps'

    H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=torch.float32, device = device)
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.float32, device = device)
    sig = torch.tensor([[[0.8],[1.1]]], dtype=torch.float32, device = device) 
    C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.float32, device = device)  # Positive semi-definite
    D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.float32, device = device)  # Positive definite
    R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.float32, device = device)  # Positive semi-definite
    T = torch.tensor(1.0, dtype=torch.float32, device = device)

    t0 = torch.tensor(0.1,dtype = torch.float32, device = device)


    n = 100000

    sample_size = 400

    time_grid = torch.linspace(t0, T, n, dtype = torch.float32, device = device) # for both MC and Riccati.
    

    LQR_sol = LQRSol_SO(H, M, sig, C, D, R, T)

    #S1 = torch.randn([n,sample_size,2,2], dtype=torch.float32)

    S = LQR_sol.riccati_solver(time_grid.unsqueeze(0))

    X0 = 0.5*torch.ones([1,1,2], dtype=torch.float32, device = device)

    dt = time_grid[1:]-time_grid[:-1]

    multX = - M@torch.linalg.inv(D)@M.T
    multa = - torch.linalg.inv(D)@M.T@S

    I = torch.eye(len(X0.squeeze()),dtype = torch.float32, device = device)

    A = (I - dt.unsqueeze(-1).unsqueeze(-1)*(H + multX))

    s_time = time.time()

    ImMCSim = ImMonteCarloYYB(A,t0,X0,T,n,dt,sig,sample_size)

    X_0_N = ImMCSim.b_solve(ImMCSim.negsub,
                             ImMCSim.A_u, 
                             ImMCSim.A_l, 
                             ImMCSim.diagAA, 
                             ImMCSim.len_b,
                             ImMCSim.dt,
                             ImMCSim.X0s)

    e_time = time.time()-s_time

   # print(f"MPS 进行了 M（{2*n}x{2*n} 矩阵）的求解（求逆加矩阵乘法），花了 {e_time} 秒, 误差为{torch.sum(torch.abs(X_0_N-X_0_N1))}")
    print(f"MPS 完成了 {2*n}x{2*n} 线性方程组 Ax=b 的求解（求 A_inv 加上矩阵乘法 A_inv @ b {2*n}x{sample_size}向量），\n 总共花了 {e_time} 秒（优化后）\n")

    print(f"MPS finished solving the a {2*n}x{2*n} linear equation system Ax=b (solving inverse of A moreover do matrix multiplication M_inv @ b (where b is a{2*n}x{sample_size} vector),\n Totally {e_time} Seconds (Optimized)")