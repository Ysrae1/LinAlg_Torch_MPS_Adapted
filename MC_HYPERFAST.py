# Copyright (c) 2024, Ysrae1
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.


# Program: Enhanced Monte Carlo Simulator for Apple Silicon
# Purpose: This program features a specialized Monte Carlo simulator optimized for Apple Silicon,
#          utilizing a tailored 2D LQR solver example. It aims to familiarize users with matrix 
#          transformation operations in PyTorch, while exploring the performance limits of Apple Silicon MPS.

# Description: This script employs a custom function designed for inverting large matrices through 
#              matrix partitioning. This method significantly enhances the computational efficiency 
#              required for complex Monte Carlo simulations.

# File: MC_HYPERFAST.py
# Description: This is the main function
# Author: Ysrae1
# Email: Ysrae1@outlook.com
# Date: 2024-04-19
# Copyright: Copyright (c) 2024, Ysrae1
# License: MIT License


import os
import shutil
import torch
import time
from datetime import datetime
from lib.LQR_SO import LQRSol
from lib.MC_H import MonteCarlo_H


if __name__=='__main__':

    # Initialization

    device = 'cpu'

    device_MC = 'mps'

    scheme = 'im'

    del_result = True

    H = torch.tensor([[0.9, 0.8], [-0.6, 0.9]], dtype=torch.float32, device = device)
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.float32, device = device)
    sig = torch.tensor([[10,5],[0.1,11]], dtype=torch.float32, device = device) 
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

    print("Monte Carlo 程序开始。\nMonte Carlo programme starts.\n")

    print("状态过程模拟结果文件最后将被" + "删除。" if del_result else "保留。" )
    print("State process simulation results will be " + "deleted in the end.\n" if del_result else "saved in the end.\n")

    print("开始 LQR solver 初始化 ...", end=' ')

    s_time = time.time()

    LQR_sol = LQRSol(H, M, sig, C, D, R, T, n, 'euler',device = device)

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

    path = "results/simulation_on_"+device_MC+'_'+time_str+"/"

    os.makedirs(path, exist_ok=True)

    print(f"在 {device_MC} 上开始 Monte Carlo 模拟，时间步为 {n} ，样本量为 {sample_size} 。 (求解线性方程组 Ax=b) ... \n")

    s_time = time.time()

    for run in range(Runs):

        s_i_time = time.time()

        A_MC = A.to(device_MC)
        t0_MC = t0.to(device_MC)
        X0_MC = X0.to(device_MC)
        T_MC = T.to(device_MC)
        dt_MC = dt.to(device_MC)
        sig_MC = sig.to(device_MC)

        MCSim = MonteCarlo_H(A_MC,t0_MC,X0_MC,T_MC,n,dt_MC,sig_MC,sample_size,scheme=scheme,device=device_MC)

        AA_0 = - torch.eye(MCSim.X0s.shape[0],dtype = torch.float32, device = device_MC)

        X_0_N1,AA_0 = MCSim.b_solve(MCSim.negsub,
                                MCSim.A_u, 
                                MCSim.A_l, 
                                MCSim.diagAA, 
                                MCSim.len_b,
                                MCSim.dt,
                                MCSim.X0s,
                                AA_0)


        print(f'Run {run+1}/{Runs} is done. ({time.time() - s_i_time :.6f} seconds)')
        torch.save(X_0_N1, path + f'X1_{run}.pt')

        del MCSim
        del X_0_N1

    device = 'cpu'

    e_time = time.time()-s_time

    print(f"\n完成了 {Runs} 次 {2*n}x{2*n} 线性方程组 Ax=b 的求解（{e_time} 秒）。\n（求 A_inv 并进行矩阵乘法 A_inv @ b {2*n}x{sample_size} 的向量）\n")

    print(f"Solving finished. (in {e_time} seconds) Done the following works:\n 1. the inverse of A ({2*n}x{2*n} matrix) \n 2. the matrix multiplication A_inv @ b ({2*n}x{sample_size} vector)")

    print("开始求解 J ...",end=' ')

    s_time = time.time()

    J1 = torch.zeros(sample_size*Runs)
    # J2 = torch.zeros(sample_size*Runs)

    J1_means = torch.zeros(Runs)
    # J2_means = torch.zeros(Runs)

    MCSim = MonteCarlo_H(A,t0,X0,T,n,dt,sig,sample_size,scheme = scheme,device=device)

    for run in range(Runs):

        X1_res = torch.load(path + f'X1_{run}.pt').cpu()
        # X2_res = torch.load(path + f'X2_{run}.pt')

        X1_i = X1_res.T.reshape([sample_size,n,1,2])
        # X2_i = X2_res.T.reshape([sample_size,n,1,2])

        J1[run*sample_size:(run+1)*sample_size] = MCSim.J_computation(X1_i, multa,C,D,R).squeeze()
        # J2[run*sample_size:(run+1)*sample_size] = MCSim.J_computation(X2_i, multa,C,R).squeeze()

        J1_means[run] = torch.mean(J1[run*sample_size:(run+1)*sample_size])
        # J2_means[run] = torch.mean(J2[run*sample_size:(run+1)*sample_size])

    print(f'({time.time() - s_time:.6f} 秒) 完成。')

    print('J means          是 ', J1_means)

    # print('J2 means 是 ', J2_means)

    print('Value function   是 ', LQR_sol.value_function(t0.unsqueeze(0),X0))

    if del_result:
        if os.path.exists(path):
            shutil.rmtree(path)

            print("状态过程模拟结果文件已删除。\n State process simulation results successfully deleted.")
        else:
            print("状态过程模拟结果文件夹不存在。\n State process simulation results folder does not exist.")









