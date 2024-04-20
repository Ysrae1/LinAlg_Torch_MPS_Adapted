# Custom Linear Algebra Solutions for PyTorch MPS on Apple Silicon (M2 Pro) 

This version of `README.md` is only for avoiding the GitHub's lame LaTeX rendering for which most of the math parts is eliminated 🙄 . Check the complete version 👉🏻 [README(complete).pdf](./README(complete).pdf) if necessary.

## Introduction

This program is a self-entertaining game of pushing the computation performance of Apple Silicon MPS to the limit for solving naive Linear Algebra problems. Inspired by the SCDAA coursework 2024, University of Edinburgh. 


A SUPER fast solution for processing Monte Carlo simulations with a high computational cost is built, which you can find in the `MC_HYPERFAST.py`.

## Main Purpose

The main purpose of this program for the author is to make the most of his computation resources at hand (which though was swiftly overtaken by its successors 🤷🏻‍♂️🤷🏻‍♂️🤷🏻‍♂️). The numerical example used to tap into his computer’s computational power is drawn from coursework in Stochastic Control and Dynamic Asset Allocation at the University of Edinburgh, 2024.

The initial challenge arose when the author attempted to speed up a Monte Carlo simulation by shifting from the most basic `for` loop iteration to solving a large sparse linear equation system, effectively needing to compute the inverse of a massive matrix.  Coincidentally, he had just encountered some possibly unreliable claims in an advertisement from **Apple**, stating that the **Metal Performance Shader (MPS)** was particularly adept at handling such issues. However, after expending a great deal of effort to construct the matrix, his silicon cheerfully reported a memory overflow error as its way of saying thanks.

In the end, he had to tackle this problem by delving into the fundamental mathematics himself. Thankfully, there's still a shred of amusement to be found, as this ordeal somewhat resembles character leveling in an unseen video game.

## Mathematical Foundations

### 2-D Linear Quadratic Regulator

#### Dynamic

Consider the controlled process in space, expressed as


$$
dX_s = [HX_s + M \alpha_s]ds + \sigma dW_s , s\in[t,T],X_t = x \text{.} \nonumber
$$



#### Objective

Our aimed is to minimize


$$
J^\alpha(t,x) \coloneqq \mathbb{E}^{t,x}\left[\int_t^T(X^{\top}_sCX_s +\alpha^{\top}_sD\alpha_s)ds + X^{\top}_TRX_T \right]\text{.} \nonumber
$$



#### (Optimal) Value Function

The optimal of the problem above is called value function, denoted as


$$
v(t,x) \coloneqq \inf \limits_{\alpha} J^\alpha(t,x) \text{.} \nonumber
$$


By solving Bellman PDE, we can obtain that


$$
v(t,x) = x^{\top}S(t)x + \int_t^T \mathrm{tr}(\sigma\sigma^{\top}S(r))dr\text{.} \nonumber
$$



#### Riccati ODE and its Solution

The function $S(t)$​​ in the expression of the value function above is the solution of Riccati ODE 


$$
\begin{align*} \frac{dS(r)}{dr} &= -(S(r)H+H^{\top} S(r)) + S(r)MD^{-1}M^{\top}S(r) - C ,r \in [t,T] \text{,} \\\ S(T) & =  R \text{.}\end{align*} \nonumber
$$



#### Optimal Control

The corresponding optimal control is 


$$
a(t,x) = -D^{-1}M^{\top}S(t)x \text{.} \nonumber
$$



### Monte Carlo Verification

#### Basic Illustration

We can bilaterally verify part of the code for Monte Carlo simulation and the part for Riccati equation solver.

Substituting the optimal control $a(t,x) = -D^{-1}M^{\top}S(t)x$ back to the dynamic of $X_s$, we choose Euler as our discretizing numerical scheme to operate Monte Carlo.

#### Iteration Equation System in Matrix Form

For the two schemes above, we can rewrite them in matrices.


### Block Matrix Inversion

#### Basic Idea

For a full-rank matrix, if partitioned into four blocks, then it can be inverted blockwise, which will finally generate a recursion chain.

In general, matrix inversion by partition can be expressed as:


$$
\left[
\begin{matrix}
\Delta & \Lambda \\\ \Xi & \Sigma
\end{matrix}
\right]^{-1} =
\left[
\begin{matrix}
\Delta^{-1}+\Delta^{-1}\Lambda( \Sigma  - \Xi\Delta^{-1} \Lambda )^{-1} \Xi \Delta^{-1} & \Delta^{-1}\Lambda(\Sigma - \Xi\Delta^{-1}\Lambda)^{-1} \\\ -(\Sigma - \Xi \Delta^{-1} \Lambda)^{-1}\Xi \Delta^{-1} & (\Sigma - \Xi \Delta^{-1} \Lambda)^{-1}
\end{matrix}
\right], \nonumber
$$


which means we can first compute $\Delta^{-1}$, then solve the sub-inversion $(\Sigma - \Xi \Delta^{-1} \Lambda)^{-1}$, where we can partition again and again. These steps will form a whole recursion.

#### The Second Level for Partitioning the Computation Workload

An ideal way to partition the workload at a higher level is to consider the characteristics of the problem itself, and for each time take the upper left square part with the row number to the multiple of the dimension of $X_t$. Take the $\mathbb{A}_{\mathrm{I}}^{(2N \times 2N)}$​​ for instance. We can partition it as below,


$$
\left[
\begin{array}{cccc|cc}
1 & 0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\
 -1 & 0 & A_{\mathrm{I},2,11} &  A_{\mathrm{I},2,12} & 0 & 0 &\dots & 0 & 0 & 0 & 0\\
 0 & -1 & A_{\mathrm{I},2,21} &  A_{\mathrm{I},2,22} & 0 & 0 &\dots & 0 & 0 & 0 & 0 \\
 \hline
0 & 0 & -1 & 0 & A_{\mathrm{I},3,11} & A_{\mathrm{I},3,12} &  \dots & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & -1 & A_{\mathrm{I},3,21} &  A_{\mathrm{I},3,22} & \dots & 0 & 0 & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
0 & 0 & 0 & 0 & 0 & 0 & \dots & -1 & 0 & A_{\mathrm{I},N,11} & A_{\mathrm{I},N,12}\\
0 & 0 & 0 & 0 & 0 & 0 & \dots & 0 & -1 & A_{\mathrm{I},N,21} & A_{\mathrm{I},N,22}
\end{array}
\right]
.\\ \nonumber
$$


Next, we solve the first upper-left square problem by solving its inversion by block inversion. Then, we obtain the first $(2 \times 1)$ section of $\mathbb{X}^{(2N \times 1)}$​​ by


$$
\left[
\begin{matrix}
x_{1,t_{1}}^N \\\
x_{2,t_{1}}^N \\\
x_{1,t_{2}}^N \\\
x_{2,t_{2}}^N 
\end{matrix}
\right] =\left[
\begin{array}{cccc}
1 & 0 & 0 & 0\\\
0 & 1 & 0 & 0\\\
 -1 & 0 & A_{\mathrm{I},2,11} &  A_{\mathrm{I},2,12}\\\
 0 & -1 & A_{\mathrm{I},2,21} &  A_{\mathrm{I},2,22}
\end{array}
\right]^{-1}
\left[
\begin{matrix}
B_{\mathrm{I},1,1} \\\
B_{\mathrm{I},1,2} \\\
B_{\mathrm{I},2,1} \\\
B_{\mathrm{I},2,2} 
\end{matrix}
\right]. \nonumber
$$


When dealing with the next block, to keep the correctness, we should change the first terms of the next section of $\mathbb{B}_{\mathrm{I}}^{(2N \times 1)}$ by **massaging a little bit** with the tail of the last solution piece of $\mathbb{X}^{(2N \times 1)}$.

It is probably noticed that the matrix problem with $2$ **time steps** as described essentially duplicates the behavior of the most elementary iteration,  which tackles just $1$ **time step** at a time. However, the advantage of using matrix representation is that it allows us to effortlessly scale the number of time steps we wish to solve simultaneously to virtually any extent. This flexibility enables us to really push our computer to its limits, effectively making it sweat by working at its maximum capacity 😈.

#### Additional Observation

By expanding the second dimension of $\mathbb{B}$ and thus $\mathbb{X}$​​, we can now scale the sampling batch size as well! 🥳
