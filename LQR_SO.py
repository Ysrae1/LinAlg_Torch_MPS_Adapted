import torch
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import warnings

class LQRSolver_SO:
    #default numerical method is Euler.
    #Euler and Runge-Kutta are supported in this code.
    def __init__(self, H, M, sigma, C, D, R, T = 1, method="euler"):
        
        if not self.is_semi_positive_definite(C):
            raise ValueError("Matrix C must be semi-positive definite.")
        if not self.is_semi_positive_definite(R):
            raise ValueError("Matrix R must be semi-positive definite.")
        if not self.is_positive_definite(D):
            raise ValueError("Matrix D must be positive definite.")
 
        self.H = H
        self.M = M
        self.sigma = sigma
        self.C = C
        self.D = D
        self.R = R
        self.T = T
        self.method = method 
        self.N_step = 10000
    
    def is_positive_definite(self, matrix):
        
        eigvals, _ = torch.linalg.eig(matrix)
        real_parts = eigvals.real
        return torch.all(real_parts > 0) 

    def is_semi_positive_definite(self, matrix):

        eigvals, _ = torch.linalg.eig(matrix)
        real_parts = eigvals.real
        return torch.all(real_parts >= 0) 

    def solve_riccati_ode(self, time_grids):
        
        if not isinstance(time_grids, torch.Tensor):
            raise TypeError("time_grid should be an batch_size*1-D torch.Tensor")
        else:
            if not (torch.all(np.abs(time_grids[:,-1] - self.T) <= 1e-12) or torch.all(time_grids[:,0]>=0)):
                print()
                raise Exception("Please ensure that the first entry of time_grid >= 0 and the last entry is equal to T.")
            else:
                time_grids_in = time_grids

        time_grids_in = torch.flip(time_grids, dims=[1])
        S = self.R.clone()
        repl = torch.ones(time_grids_in.shape)
        rep_exd = repl.unsqueeze(-1).unsqueeze(-1)
        S_exd = S.unsqueeze(0).unsqueeze(0)

        S_repl = rep_exd*S_exd
        dt = time_grids_in[:,1:]-time_grids_in[:,:-1]

        for i in range(dt.shape[1]):
            

            if self.method == "euler":
                S_repl[:,i+1] = self.euler_step(S_repl[:,i], dt[:,i])
            elif self.method == "rk4":
                S_repl[:,i+1] = self.rk4_step(S_repl[:,i], dt[:,i])
            else:
                raise ValueError("Unsupported method")

        return torch.flip(S_repl, dims=[1])

    def euler_step(self, S_in, dt_in):
        
        dS_in = -2 * self.H.T @ S_in + S_in @ self.M @ torch.inverse(self.D) @ self.M.T @ S_in - self.C

        dt_resized = dt_in[:, None, None]

        return S_in + dS_in * dt_resized
    
    def rk4_step(self, S_in, dt_in):

        def riccati_derivative(S):
            return -2 * self.H.T @ S_in + S_in @ self.M @ torch.inverse(self.D) @ self.M.T @ S_in - self.C
        
        dt_resized = dt_in[:, None, None]

        k1 = riccati_derivative(S_in)
        k2 = riccati_derivative(S_in + 0.5 *  k1 * dt_resized)
        k3 = riccati_derivative(S_in + 0.5 *  k2 * dt_resized)
        k4 = riccati_derivative(S_in + k3 * dt_resized)

        return S_in + (k1 + 2*k2 + 2*k3 + k4)*dt_resized/6

    def value_function(self, t_batch, x_batch, sol_method = 'interpolation'):
        
        if sol_method == 'interpolation':
            
            N_step = 2*self.N_step
            
            if not (t_batch.dim() == 1 and torch.all((t_batch >= 0) & (t_batch <= 1))):
                raise TypeError("t_batch should be a 1D tensor in which every entry is in [0,1].")
            else:
                if not (x_batch.dim() == 3 and x_batch.size()[0] == len(t_batch) and x_batch.size()[1] == 1 and x_batch.size()[2] == self.H.size()[0]):
                    raise TypeError("x_batch should have shape (%d, 1, %d)."%(len(t_batch),self.H.size(2)))
            
            time_grid = torch.stack([torch.linspace(0, self.T, N_step, dtype=torch.double) for i in [0]])
            
            S_tensor_tensor = self.solve_riccati_ode(time_grid)

            index_s_1 = torch.searchsorted(time_grid[0,:], torch.min(t_batch), right=True) - 1
            time_grid_for_intpl = time_grid[0,index_s_1:]
            S_tensor_tensor_for_intpl = S_tensor_tensor[0,index_s_1:]
            
            time_grid_for_intpl_np = time_grid_for_intpl.numpy()
            S_tensor_tensor_for_intpl_np = S_tensor_tensor_for_intpl.numpy()
            t_batch_np = t_batch.numpy()
            
            S_c_spl = CubicSpline(time_grid_for_intpl_np, S_tensor_tensor_for_intpl_np)
            
            traces = torch.einsum('bii->b', self.sigma @ self.sigma.transpose(1,2) @ S_tensor_tensor_for_intpl).unsqueeze(1).unsqueeze(2)
            
            trace_cubic = CubicSpline(time_grid_for_intpl_np, traces.numpy())
            
            def S_intpl(t_batch_np_in):
                return torch.from_numpy(S_c_spl(t_batch_np_in))
                
            S_t_s = S_intpl(t_batch_np)
            x_batch_T = x_batch.transpose(1, 2) 
            
            xTSx = x_batch @ S_t_s @ x_batch_T

            for i in range(len(t_batch_np)):
                
                t1_index = (time_grid_for_intpl > t_batch[i]).nonzero(as_tuple=True)[0].min().item() if (time_grid_for_intpl > t_batch[i]).any() else None
                time_grid_after_t1 = time_grid_for_intpl[t1_index:]
                dt_t1 = time_grid_after_t1[0] - t_batch[i]
                dt_after_t1 = time_grid_after_t1[1:]-time_grid_after_t1[:-1]
                traces_after_t1 = traces[t1_index:]
                
                traces_after_t1_for_int = torch.tensor(0.5,dtype=torch.double)*(traces_after_t1[1:]+traces_after_t1[:-1])
                int_t1 = torch.tensor(0.5,dtype=torch.double)*(torch.from_numpy(trace_cubic(t_batch_np[i])).to(dtype=torch.double)+traces_after_t1[0])



                intgl = (dt_t1*int_t1).squeeze() + (dt_after_t1.view(1, -1)@traces_after_t1_for_int.squeeze(1))

                xTSx[i,0]+= intgl.squeeze()
            
            v_tx = xTSx.squeeze()
            
            
        if sol_method == 'direct_calcul':
        
            # Verify the shapes of the inputs.

            if not (t_batch.dim() == 1 and torch.all((t_batch >= 0) & (t_batch <= 1))):
                raise TypeError("t_batch should be a 1D tensor in which every entry is in [0,1].")
            else:
                if not (x_batch.dim() == 3 and x_batch.size()[0] == len(t_batch) and x_batch.size()[1] == 1 and x_batch.size()[2] == self.H.size()[0]):
                    raise TypeError("x_batch should have shape (%d, 1, %d)."%(len(t_batch),self.H.size(2)))

            time_grids = torch.stack([torch.linspace(float(t), self.T, self.N_step, dtype=torch.double) for t in t_batch])

            S_tensor_tensor = self.solve_riccati_ode(time_grids)
            #print(S_tensor_tensor.shape)
            S_t_s = S_tensor_tensor[:, 0, :, :]
            S_t_T = S_tensor_tensor[:, 1:, :, :]
            x_batch_T = x_batch.transpose(1, 2) 

            xTSx = x_batch @ S_t_s @ x_batch_T

            dts = time_grids[:, 1:] - time_grids[:, :-1]

            sigma_T = self.sigma.transpose(1,2)

            trace_for_int = torch.einsum('bcii->bc', self.sigma @ self.sigma.transpose(1, 2) @ S_t_T.transpose(0,1)).unsqueeze(1).unsqueeze(2)
            trace_for_int = trace_for_int.squeeze() 

            integral_part = dts @ trace_for_int

            v_tx = xTSx.squeeze() + torch.diag(integral_part).squeeze()

        return v_tx

    def markov_control(self, t_batch, x_batch, sol_method = 'interpolation'):
        
        if sol_method == 'interpolation':
            
            N_step = 2*self.N_step
            
            if not (t_batch.dim() == 1 and torch.all((t_batch >= 0) & (t_batch <= 1))):
                raise TypeError("t_batch should be a 1D tensor in which every entry is in [0,1].")
            else:
                if not (x_batch.dim() == 3 and x_batch.size()[0] == len(t_batch) and x_batch.size()[1] == 1 and x_batch.size()[2] == self.H.size()[0]):
                    raise TypeError("x_batch should have shape (%d, 1, %d)."%(len(t_batch),self.H.size(2)))
            
            time_grid = torch.stack([torch.linspace(0, self.T, N_step, dtype=torch.double) for i in [0]])
            
            S_tensor_tensor = self.solve_riccati_ode(time_grid)
            
            index_s_1 = torch.searchsorted(time_grid[0,:], torch.min(t_batch), right=True) - 1
            time_grid_for_intpl = time_grid[0,index_s_1:]
            S_tensor_tensor_for_intpl = S_tensor_tensor[0,index_s_1:]
            
            time_grid_for_intpl_np = time_grid_for_intpl.numpy()
            S_tensor_tensor_for_intpl_np = S_tensor_tensor_for_intpl.numpy()
            t_batch_np = t_batch.numpy()
            
            S_c_spl = CubicSpline(time_grid_for_intpl_np, S_tensor_tensor_for_intpl_np)
            
            def S_intpl(t_batch_np_in):
                return torch.from_numpy(S_c_spl(t_batch_np_in))
                
            S_t_s = S_intpl(t_batch_np)
            x_batch_T = x_batch.transpose(1, 2) 
            
            MT = self.M.T
            D_inv = torch.inverse(self.D)
            x = torch.transpose(x_batch,dim0 = 2,dim1 = 1)
            a_tx = - D_inv @ MT @ S_t_s @ x_batch_T
            a_tx = torch.transpose(a_tx,dim0 = 1,dim1 = 2).squeeze()    
            
        if sol_method == 'direct_calcul':
        
            # Verify the shapes of the inputs.

            if not (t_batch.dim() == 1 and torch.all((t_batch >= 0) & (t_batch <= 1))):
                raise TypeError("t_batch should be a 1D tensor in which every entry is in [0,1].")
            else:
                if not (x_batch.dim() == 3 and x_batch.size()[0] == len(t_batch) and x_batch.size()[1] == 1 and x_batch.size()[2] == self.H.size()[0]):
                    raise TypeError("x_batch should have shape (%d, 1, %d)."%(len(t_batch),self.H.size(2)))

            time_grids = torch.stack([torch.linspace(float(t), self.T, self.N_step, dtype=torch.double) for t in t_batch])

            S_tensor_tensor = self.solve_riccati_ode(time_grids)

            #print(S_tensor_tensor.shape)

            S_t_s = S_tensor_tensor[:, 0, :, :]
            x_batch_T = x_batch.transpose(1, 2) 

            MT = self.M.T
            D_inv = torch.inverse(self.D)
            x = torch.transpose(x_batch,dim0 = 2,dim1 = 1)
            a_tx = - D_inv @ MT @ S_t_s @ x_batch_T
            a_tx = torch.transpose(a_tx,dim0 = 1,dim1 = 2).squeeze()

        return a_tx.unsqueeze(1)