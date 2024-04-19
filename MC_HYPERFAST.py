import os
import torch
import time
from datetime import datetime
from torchinvmps import inv_mps
from LQR_SO import LQRSol_SO

class ImMonteCarloYYB:
    def __init__(self, A, t_0, X_0, T, n, dt, sig, sample_size, scheme,device, comp_step_limit = 2048):

        self.A = A
        self.t_0 = t_0
        self.X_0 = X_0
        self.T = T
        self.n = n
        self.dt = dt
        self.sig = sig
        self.sample_size = sample_size
        self.scheme = scheme
        self.device = device

        self.comp_step_limit = comp_step_limit

        self.X0s = (self.X_0.squeeze().repeat(1,self.sample_size)).reshape(len(self.X_0.squeeze()),self.sample_size)

        self.len_b = torch.tensor([1,2*n], dtype = torch.float32, device = self.device)

        diagA = torch.diagonal(self.A,dim1=-2, dim2=-1).flatten()
        
        #byproducts

        A_subs = self.A.flip(dims=[-1])
        A_subs_f = torch.diagonal(A_subs, dim1=-2, dim2=-1).flatten()
        u_A_subs_f = A_subs_f[::2] 
        l_A_subs_f = A_subs_f[1::2] 

        if self.scheme == 'im':

            self.diagAA = torch.cat((torch.ones(len(self.X_0.squeeze()), dtype = torch.float32, device = self.device), diagA))
            A_subs_0s = torch.zeros_like(u_A_subs_f,dtype = torch.float32, device = self.device)[:-1]
            self.A_u = torch.zeros(len(self.X_0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
            self.A_l = torch.zeros(len(self.X_0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
            self.A_u[len(X0.squeeze())::2] = u_A_subs_f
            self.A_u[len(X0.squeeze())+1::2] = A_subs_0s
            self.A_l[len(X0.squeeze())::2] = l_A_subs_f
            self.A_l[len(X0.squeeze())+1::2] = A_subs_0s

            self.negsub = -torch.ones(diagA.numel(), dtype = torch.float32, device = self.device)

        else:
            self.diagAA = torch.cat((torch.ones(len(self.X_0.squeeze()), dtype = torch.float32, device = self.device), torch.ones_like(diagA)))

            A_subs_0s = torch.zeros_like(u_A_subs_f,dtype = torch.float32, device = self.device)
            self.A_u = torch.zeros(len(self.X_0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
            self.A_l = torch.zeros(len(self.X_0.squeeze()) -1-2 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
            self.A_u[len(X0.squeeze())-1:-1:2] = u_A_subs_f
            self.A_l[0::2] = l_A_subs_f       

            self.negsub = torch.clone(diagA)
    
    def b_solve(self, negsub, A_u, A_l, diagAA, len_b, dt, X_0, AA_0):

        def diaggen(negsub,A_u,A_l,diagAA):
            
            matrix_negsub = torch.diag(negsub, diagonal=-len(self.X_0.squeeze()))
            
            matrix_diag = torch.diag(diagAA)

            if self.scheme == 'im':

                matrix_A_u = torch.diag(A_u, diagonal=1)
                
                matrix_A_l = torch.diag(A_l, diagonal=-1)

            else:

                matrix_A_u = torch.diag(A_u, diagonal=-1)
                
                if A_l.numel() != 0:
                    matrix_A_l = torch.diag(A_l, diagonal=-3)   
                else:
                    matrix_A_l = torch.zeros_like(matrix_diag)
            
            AA = matrix_diag+matrix_A_l+matrix_A_u+matrix_negsub

            return AA
        
        def bgen(len_b):

            if len_b[0] == 1:

                ini = torch.zeros_like(self.X_0.squeeze().repeat(1,self.sample_size)).reshape(len(self.X_0.squeeze()),self.sample_size)

                r_1 = (self.sig.repeat(int((len_b[1]/2 -1)),1,1))*torch.sqrt(dt).unsqueeze(1).unsqueeze(2)

                r_2 = torch.randn(int((len_b[1]/2 - 1)),self.sample_size,1,dim,dtype=torch.float32, device = self.device)

                r_1xr_2 = (r_1.unsqueeze(1)@r_2.transpose(2,3)).squeeze(-1).transpose(1,2).reshape(int(len_b[1]-dim),self.sample_size)

                return torch.cat((ini,r_1xr_2),dim = 0)
            
            else:

                r_1 = (self.sig.repeat(int((len_b[1]/2)),1,1))*torch.sqrt(dt).unsqueeze(1).unsqueeze(2)

                r_2 = torch.randn(int((len_b[1]/2)),self.sample_size,1,dim,dtype=torch.float32, device = self.device)

                r_1xr_2 = (r_1.unsqueeze(1)@r_2.transpose(2,3)).squeeze(-1).transpose(1,2).reshape(int(len_b[1]),self.sample_size)

                return r_1xr_2
    
        dim = X_0.shape[0]

        N = int(len_b[1]/2)
        
        if N <= self.comp_step_limit:

            b = bgen(len_b)

            b[:dim] -= AA_0 @ X_0

            if self.scheme == 'im':
                AA = diaggen(negsub,A_u,A_l,diagAA)
            else:
                AA = diaggen(negsub[:2*(N-1)],A_u[:2*N-1],A_l[:2*(N-1)-1],diagAA)

            X_0_N = (inv_mps(AA).to(self.device)@b)

            AA_0 = - torch.eye(dim ,dtype = torch.float32, device = self.device)

            if (self.scheme == 'ex')and (negsub[-dim:].numel() != 0) :

                AA_0 = torch.diag(negsub[-dim:])+torch.diag(A_u[1-dim:],diagonal=1)+torch.diag(A_l[1-dim:],diagonal=-1)
            
            return X_0_N, AA_0
        
        else:

            X_0_N = torch.clone(X_0.repeat(N,1))

            if self.scheme == 'im':

                _X_0_N,AA_0 = self.b_solve(negsub[:2*(self.comp_step_limit-1)],
                                    A_u[:2*(self.comp_step_limit)-1],
                                    A_l[:2*(self.comp_step_limit)-1],
                                    diagAA[:2*(self.comp_step_limit)],
                                    torch.tensor([len_b[0],2*self.comp_step_limit]),
                                    dt[:self.comp_step_limit - int(len_b[0])],
                                    X_0_N[:dim],
                                    AA_0)
                
            else:

                _X_0_N,AA_0 = self.b_solve(negsub[:2*(self.comp_step_limit)],
                                    A_u[:2*(self.comp_step_limit)],
                                    A_l[:2*(self.comp_step_limit)-1],
                                    diagAA[:2*(self.comp_step_limit)],
                                    torch.tensor([len_b[0],2*self.comp_step_limit]),
                                    dt[:self.comp_step_limit - int(len_b[0])],
                                    X_0_N[:dim],
                                    AA_0)
            
            X_0_N[:2*self.comp_step_limit] = _X_0_N

            negsub = negsub[2*(self.comp_step_limit):]
            A_u = A_u[2*(self.comp_step_limit):]
            A_l = A_l[2*(self.comp_step_limit):]
            diagAA = diagAA[2*(self.comp_step_limit):]    

            dt = dt[self.comp_step_limit - int(len_b[0]):]

            len_b[0] = 0*len_b[0]
            len_b[1] -= 2*self.comp_step_limit
            
            
            X_0_N_ = X_0_N[2*(self.comp_step_limit-1):2*(self.comp_step_limit)].repeat(N,1)
            
            X_0_N[2*self.comp_step_limit:],AA_0 = self.b_solve(negsub, A_u, A_l, diagAA ,len_b, dt, X_0_N_[:dim],AA_0)

            return X_0_N,AA_0
        
    def iter_solve(self,H,multX):

        X_0_N = torch.clone(self.X0s.repeat(self.n,1)).reshape([self.n,1,2])

        for i in range(self.n - 1):

            X_0_N[i+1] = ((torch.eye(2) + self.dt[i]*(H+multX[:,i]))@X_0_N[i].unsqueeze(0).transpose(1,2)).transpose(1,2).squeeze()

        return X_0_N.reshape([2*self.n,1])
        
    def J_computation(self, X_0_N, multa,C,R):
        
        alpha = multa@X_0_N.transpose(2,3)

        int_ = X_0_N@C@X_0_N.transpose(2,3) + alpha.transpose(2,3)@D@alpha

        J = X_0_N[:,-1]@R@X_0_N[:,-1].transpose(1,2) + torch.tensor(0.5)*self.dt.unsqueeze(0)@((int_.squeeze(2)[:,1:]+int_.squeeze(2)[:,:-1]))

        return J

if __name__=='__main__':

    # Initialization

    device = 'cpu'

    device_MC = 'cpu'

    scheme = 'im'

    H = torch.tensor([[0.9, 0.8], [-0.6, 0.9]], dtype=torch.float32, device = device)
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.float32, device = device)
    sig = torch.tensor([[0.4,0.2],[0.1,0.9]], dtype=torch.float32, device = device) 
    C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.float32, device = device) 
    D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.float32, device = device)  
    R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.float32, device = device)  
    T = torch.tensor(1.0, dtype=torch.float32, device = device)

    t0 = torch.tensor(0.5,dtype = torch.float32, device = device)

    n = 5000

    sample_size = 25000

    Runs = 12

    time_grid = torch.linspace(t0, T, n, dtype = torch.float32, device = device) # for both MC and Riccati.

    width = os.get_terminal_size().columns

    print("开始 LQR solver 初始化 ...", end=' ')

    s_time = time.time()

    LQR_sol = LQRSol_SO(H, M, sig, C, D, R, T, n, 'euler',device = device)

    print(f"({time.time() - s_time :.6f} 秒) 完成初始化。\n")
    
    print("开始计算 S ...", end=' ')

    s_time = time.time()

    S = LQR_sol.riccati_solver(time_grid.unsqueeze(0))

    print(f"({time.time() - s_time :.6f} 秒) 完成计算。\n")

    X0 = 1*torch.ones([1,1,2], dtype=torch.float32, device = device)

    dt = time_grid[1:]-time_grid[:-1]

    multX = - M@torch.linalg.inv(D)@M.T@S
    multa = - torch.linalg.inv(D)@M.T@S

    I = torch.eye(len(X0.squeeze()),dtype = torch.float32, device = device)

    if scheme == 'im':
        A = (I - dt.unsqueeze(-1).unsqueeze(-1)*(H + multX[:,1:]))
    if scheme == 'ex':
        A = (I + dt.unsqueeze(-1).unsqueeze(-1)*(H + multX[:,:-1]))

    sim_time = datetime.now()

    time_str = sim_time.strftime("%Y%m%d_%H%M%S")

    path = "simulation_on_"+device_MC+'_'+time_str+"/"

    os.makedirs(path, exist_ok=True)

    print(f"在 {device_MC} 上开始 Monte Carlo 模拟，时间步为 {n} , 样本量为 {sample_size} . (求解线性方程组 Ax=b) ... \n")

    s_time = time.time()

    for run in range(Runs):

        s_i_time = time.time()

        A_MC = A.to(device_MC)
        t0_MC = t0.to(device_MC)
        X0_MC = X0.to(device_MC)
        T_MC = T.to(device_MC)
        dt_MC = dt.to(device_MC)
        sig_MC = sig.to(device_MC)

        ImMCSim = ImMonteCarloYYB(A_MC,t0_MC,X0_MC,T_MC,n,dt_MC,sig_MC,sample_size,scheme=scheme,device=device_MC)

        AA_0 = - torch.eye(ImMCSim.X0s.shape[0],dtype = torch.float32, device = device_MC)

        X_0_N1,AA_0 = ImMCSim.b_solve(ImMCSim.negsub,
                                ImMCSim.A_u, 
                                ImMCSim.A_l, 
                                ImMCSim.diagAA, 
                                ImMCSim.len_b,
                                ImMCSim.dt,
                                ImMCSim.X0s,
                                AA_0)


        print(f'Run {run+1}/{Runs} is done. ({time.time() - s_i_time :.6f} seconds)')
        torch.save(X_0_N1, path + f'X1_{run}.pt')

        del ImMCSim
        del X_0_N1

    device = 'cpu'

    e_time = time.time()-s_time

    print(f"\n完成了 {Runs} 次 {2*n}x{2*n} 线性方程组 Ax=b 的求解（{e_time} 秒）。\n(求 A_inv 并进行矩阵乘法 A_inv @ b {2*n}x{sample_size} 的向量) \n")

    print(f"Solving finished. (in {e_time} seconds) Done the following works:\n 1. the inverse of A ({2*n}x{2*n} matrix) \n 2. the matrix multiplication A_inv @ b ({2*n}x{sample_size} vector)")

    print("开始求解 J ...",end=' ')

    s_time = time.time()

    J1 = torch.zeros(sample_size*Runs)
    # J2 = torch.zeros(sample_size*Runs)

    J1_means = torch.zeros(Runs)
    # J2_means = torch.zeros(Runs)

    ImMCSim = ImMonteCarloYYB(A,t0,X0,T,n,dt,sig,sample_size,scheme = scheme,device=device)

    for run in range(Runs):

        X1_res = torch.load(path + f'X1_{run}.pt').cpu()
        # X2_res = torch.load(path + f'X2_{run}.pt')

        X1_i = X1_res.T.reshape([sample_size,n,1,2])
        # X2_i = X2_res.T.reshape([sample_size,n,1,2])

        J1[run*sample_size:(run+1)*sample_size] = ImMCSim.J_computation(X1_i, multa,C,R).squeeze()
        # J2[run*sample_size:(run+1)*sample_size] = ImMCSim.J_computation(X2_i, multa,C,R).squeeze()

        J1_means[run] = torch.mean(J1[run*sample_size:(run+1)*sample_size])
        # J2_means[run] = torch.mean(J2[run*sample_size:(run+1)*sample_size])

    print(f'({time.time() - s_time:.6f} 秒) 完成。')

    print('J means          是 ', J1_means)

    # print('J2 means 是 ', J2_means)

    print('Value function   是 ', LQR_sol.value_function(t0.unsqueeze(0),X0))








