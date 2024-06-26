# File: MC_H.py
# Description: This is the main class to present High speed Monte Carlo.
# Author: Ysrae1
# Email: Ysrae1@outlook.com
# Date: 2024-04-19
# Copyright: Copyright (c) 2024, Ysrae1
# License: MIT License

import torch
from lib.INV_MPS import inv_mps

class MonteCarlo_H:
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
            self.A_u[len(X_0.squeeze())::2] = u_A_subs_f
            self.A_u[len(X_0.squeeze())+1::2] = A_subs_0s
            self.A_l[len(X_0.squeeze())::2] = l_A_subs_f
            self.A_l[len(X_0.squeeze())+1::2] = A_subs_0s

            self.negsub = -torch.ones(diagA.numel(), dtype = torch.float32, device = self.device)

        else:
            self.diagAA = torch.cat((torch.ones(len(self.X_0.squeeze()), dtype = torch.float32, device = self.device), torch.ones_like(diagA)))

            A_subs_0s = torch.zeros_like(u_A_subs_f,dtype = torch.float32, device = self.device)
            self.A_u = torch.zeros(len(self.X_0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
            self.A_l = torch.zeros(len(self.X_0.squeeze()) -1-2 + u_A_subs_f.numel()*2, dtype = torch.float32, device = self.device)
            self.A_u[len(X_0.squeeze())-1:-1:2] = u_A_subs_f
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
        
    def J_computation(self, X_0_N, multa,C,D,R):
        
        alpha = multa@X_0_N.transpose(2,3)

        int_ = X_0_N@C@X_0_N.transpose(2,3) + alpha.transpose(2,3)@D@alpha

        J = X_0_N[:,-1]@R@X_0_N[:,-1].transpose(1,2) + torch.tensor(0.5)*self.dt.unsqueeze(0)@((int_.squeeze(2)[:,1:]+int_.squeeze(2)[:,:-1]))

        return J
