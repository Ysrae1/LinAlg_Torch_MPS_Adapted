# Custom Linear Algebra Solutions for PyTorch MPS on Apple Silicon (M2 Pro) 

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

Substituting the optimal control $a(t,x) = -D^{-1}M^{\top}S(t)x$ back to the dynamic of $X_s$, and choose Euler as our discretizing numerical scheme to operate Monte Carlo, we'll get the formulae for iteration as follows:

- **Explicit Scheme**

  
  $$
  \begin{align*}
  X_{t_{n+1}}^N &= X_{t_{n}}^N + \tau [HX_{t_{n}}^N-MD^{-1}M^{\top}S(t_n)X_{t_{n}}^N] + \sigma (W_{t_{n+1}}-W_{t_{n}}) \text{,} \\\ n &= k,\dots,N \text{,} \\\ X_{t_{k}}^N &= x.
  \end{align*} \nonumber
  $$
  

- **Implicit Scheme**
  
  
  $$
  \begin{align*}
  X_{t_{n+1}}^N &= X_{t_{n}}^N + \tau [HX_{t_{n+1}}^N-MD^{-1}M^{\top}S(t_{n+1})X_{t_{n+1}}^N] + \sigma (W_{t_{n+1}}-W_{t_{n}}) \text{,} \\\
  n &= k,\dots,N \text{,} \\\ X_{t_{k}}^N &= x.
  \end{align*} \nonumber
  $$



#### Iteration Equation System in Matrix Form

For the two schemes above, we can rewrite them in matrices.

Denote the coefficient matrices as follows:


$$
\begin{align*} I &= \left[\begin{matrix} 1 & 0 \\\ 0 & 1 \end{matrix}\right], & H &= \left[\begin{matrix} H_{11} & H_{12} \\\ H_{21} & H_{22} \end{matrix}\right], & M &= \left[ \begin{matrix} M_{11} & M_{12}\\\ M_{21} & M_{22} \end{matrix}\right], \\\  D^{-1} &= \left[\begin{matrix} D_{11} & D_{12} \\\ D_{21} & D_{22}\end{matrix}\right]^{-1}, & {S(\lambda )} &= \left[\begin{matrix} S_{11}(\lambda ) & S_{12}(\lambda ) \\\ S_{21}(\lambda ) & S_{22}(\lambda ) \end{matrix}\right], & \sigma &= \left[\begin{matrix} \sigma _{11} & \sigma _{12} \\\ \sigma _{21} & \sigma _{22} \end{matrix}\right]. \end{align*} \nonumber
$$



- **Explicit Scheme**

  

$$
\begin{align*}
\left[\begin{matrix} x_{1,t_{n+1}}^N \\\ x_{2,t_{n+1}}^N \end{matrix}
\right]
&=
\left[I + \tau[H-MD^{-1}M^{\top}S(t_n)]\right]
\left[\begin{matrix}
x_{1,t_{n}}^N \\\
x_{2,t_{n}}^N
\end{matrix}\right] + \sqrt{\tau} \sigma 
\left[\begin{matrix}
z_{1,t_{n}}^N \\\
z_{2,t_{n}}^N
\end{matrix}\right], \\\
n &= k,\dots,N \text{,} \\\

\left[\begin{matrix}
x_{1,t_{k}}^N \\\
x_{2,t_{k}}^N\end{matrix}\right] & = \left[\begin{matrix}
x_{1} \\\
x_{2}
\end{matrix}\right],
\end{align*} \nonumber
$$



where


$$
z_{i,t_n}^N \sim \mathcal{N}(0,1), \quad \forall\ i \in \{1,2\}. \nonumber
$$



Denote new coefficient matrices for combination:


$$
\begin{align*}
A_{\mathrm{E},n} & = \left[
\begin{matrix}
A_{\mathrm{E},n,11} & A_{\mathrm{E},n,12}\\\
A_{\mathrm{E},n,21} & A_{\mathrm{E},n,22}
\end{matrix}
\right] \coloneqq \left[I + \tau[H-MD^{-1}M^{\top}S(t_n)]\right], \quad \quad \ \ \ n = k,\dots,N \text{,}\\\
B_{\mathrm{E},n} & = \quad \quad \left[
\begin{matrix}
B_{\mathrm{E},n,1} \\\
B_{\mathrm{E},n,2}
\end{matrix}
\right]\quad \ \ \ \  \ \coloneqq \quad
\left\{ 
\begin{aligned}
&&&\left[\begin{matrix}
x_{1} \\\
x_{2}
\end{matrix}
\right],\quad& \ \ n &= k \\\
&&\tau&\left[\begin{matrix}
\sigma_{11} & \sigma_{12} \\\
\sigma_{21} & \sigma_{22}
\end{matrix}
\right]\left[
\begin{matrix}
z_{1,t_{n}}^N \\\
z_{2,t_{n}}^N
\end{matrix}
\right], & \ \ n &= k+1,\dots,N
\end{aligned} \right.
\text{.}
\end{align*} \nonumber
$$


Let $k = 1$​​, then we can construct the whole matrix system as follows:


$$
\mathbb{A}_{\mathrm{E}}^{(2N \times 2N)} \mathbb{X}^{(2N \times 1)} = \mathbb{B}_{\mathrm{E}}^{(2N \times 1)}, \nonumber
$$


where


$$
\begin{align*}

\mathbb{A}_{\mathrm{E}}^{(2N \times 2N)} & = 
\left[
\begin{matrix}
1 & 0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\\
0 & 1 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\\

- A_{\mathrm{E},1,11} & - A_{\mathrm{E},1,12} & 1 & 0 & 0 & 0 &\dots & 0 & 0 & 0 & 0\\\
- A_{\mathrm{E},1,21} & - A_{\mathrm{E},1,22} & 0 & 1 & 0 & 0 &\dots & 0 & 0 & 0 & 0 \\\
  0 & 0 & - A_{\mathrm{E},2,11} & - A_{\mathrm{E},2,12} & 1 & 0 & \dots & 0 & 0 & 0 & 0\\\
  0 & 0 & - A_{\mathrm{E},2,21} & - A_{\mathrm{E},2,22} & 0 & 1 & \dots & 0 & 0 & 0 & 0\\\
  \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\\
  0 & 0 & 0 & 0 & 0 & 0 & \dots & - A_{\mathrm{E},N-1,11} & - A_{\mathrm{E},N-1,12} & 1 & 0 \\\
  0 & 0 & 0 & 0 & 0 & 0 & \dots & - A_{\mathrm{E},N-1,21} & - A_{\mathrm{E},N-1,22} & 0 & 1
  \end{matrix}
  \right], \\\
  
  \mathbb{X}^{(2N \times 1)} &= 
\left[
\begin{matrix}
x_{1,t_{1}}^N \\\
x_{2,t_{1}}^N \\\
x_{1,t_{2}}^N \\\
x_{2,t_{2}}^N \\\
x_{1,t_{3}}^N \\\
x_{2,t_{3}}^N \\\
\vdots\\\
x_{1,t_{N}}^N \\\
x_{2,t_{N}}^N 
\end{matrix}
\right],
\qquad \qquad \qquad \qquad \qquad 
\mathbb{B}_{\mathrm{E}}^{(2N \times 1)} = 
\left[
\begin{matrix}
B_{\mathrm{E},1,1} \\\
B_{\mathrm{E},1,2} \\\
B_{\mathrm{E},2,1} \\\
B_{\mathrm{E},2,2} \\\
B_{\mathrm{E},3,1} \\\
B_{\mathrm{E},3,2} \\\
\vdots\\\
B_{\mathrm{E},N,1} \\\
B_{\mathrm{E},N,2} 
\end{matrix}
\right].
  \end{align*} \nonumber
$$

- **Implicit Scheme**

  

$$
\begin{align*}
\left[I - \tau[H-MD^{-1}M^{\top}S(t_n)]\right]\left[
\begin{matrix}
x_{1,t_{n+1}}^N \\\
x_{2,t_{n+1}}^N
\end{matrix}
\right]
&=
\left[
\begin{matrix}
x_{1,t_{n}}^N \\\
x_{2,t_{n}}^N
\end{matrix}
\right] + \sqrt{\tau} \sigma \left[
\begin{matrix}
z_{1,t_{n}}^N \\\
z_{2,t_{n}}^N
\end{matrix}
\right], \\\
n &= k,\dots,N \text{,} \\\

\left[
\begin{matrix}
x_{1,t_{k}}^N \\\
x_{2,t_{k}}^N
\end{matrix}
\right] & = \left[
\begin{matrix}
x_{1} \\\
x_{2}
\end{matrix}
\right],
\end{align*} \nonumber
$$

where


$$
z_{i,t_n}^N \sim \mathcal{N}(0,1), \quad \forall\ i \in \{1,2\}. \nonumber
$$


Denote new coefficient matrices for combination:


$$
\begin{align*}
A_{\mathrm{I},n} & = \left[
\begin{matrix}
A_{\mathrm{I},n,11} & A_{\mathrm{I},n,12}\\\
A_{\mathrm{I},n,21} & A_{\mathrm{I},n,22}
\end{matrix}
\right]
\coloneqq \left[I - \tau[H-MD^{-1}M^{\top}S(t_n)]\right], \quad \quad \ \ \ n = k,\dots,N , \\\

B_{\mathrm{I},n} & = \quad \quad  \left[
\begin{matrix}
B_{\mathrm{I},n,1} \\\
B_{\mathrm{I},n,2}
\end{matrix}
\right]\quad \ \ \ \  \coloneqq \quad 
\left\{ \begin{aligned}
&&&\left[\begin{matrix}
x_{1} \\\
x_{2}
\end{matrix}
\right],\quad& \ \ n &= k \\\
&&\tau&\left[\begin{matrix}
\sigma_{11} & \sigma_{12} \\\
\sigma_{21} & \sigma_{22}
\end{matrix}
\right]\left[
\begin{matrix}
z_{1,t_{n}}^N \\\
z_{2,t_{n}}^N
\end{matrix}
\right], & \ \ n &= k+1,\dots,N
\end{aligned} \right.
\text{.}

\end{align*}
$$



Let $k = 1$​​, then we can construct the whole matrix system as follows:


$$
\mathbb{A}_{\mathrm{I}}^{(2N \times 2N)} \mathbb{X}_{\mathrm{I}}^{(2N \times 1)} = \mathbb{B}_{\mathrm{I}}^{(2N \times 1)}, \nonumber
$$

where
$$
\begin{align*}
\mathbb{A}_{\mathrm{I}}^{(2N \times 2N)} &= 
\left[
\begin{matrix}
1 & 0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\\
0 & 1 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\\
 -1 & 0 & A_{\mathrm{I},2,11} &  A_{\mathrm{I},2,12} & 0 & 0 &\dots & 0 & 0 & 0 & 0\\\
 0 & -1 & A_{\mathrm{I},2,21} &  A_{\mathrm{I},2,22} & 0 & 0 &\dots & 0 & 0 & 0 & 0 \\\
0 & 0 & -1 & 0 & A_{\mathrm{I},3,11} & A_{\mathrm{I},3,12} &  \dots & 0 & 0 & 0 & 0 \\\
0 & 0 & 0 & -1 & A_{\mathrm{I},3,21} &  A_{\mathrm{I},3,22} & \dots & 0 & 0 & 0 & 0\\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\\
0 & 0 & 0 & 0 & 0 & 0 & \dots & -1 & 0 & A_{\mathrm{I},N,11} & A_{\mathrm{I},N,12}\\\
0 & 0 & 0 & 0 & 0 & 0 & \dots & 0 & -1 & A_{\mathrm{I},N,21} & A_{\mathrm{I},N,22}
\end{matrix}
\right],\\\
\mathbb{X}^{(2N \times 1)} &= 
\left[
\begin{matrix}
x_{1,t_{1}}^N \\\
x_{2,t_{1}}^N \\\
x_{1,t_{2}}^N \\\
x_{2,t_{2}}^N \\\
x_{1,t_{3}}^N \\\
x_{2,t_{3}}^N \\\
\vdots\\\
x_{1,t_{N}}^N \\\
x_{2,t_{N}}^N 
\end{matrix}
\right],
\qquad \qquad \qquad \qquad \qquad 
\mathbb{B}_{\mathrm{I}}^{(2N \times 1)} =
\left[
\begin{matrix}
B_{\mathrm{I},1,1} \\\
B_{\mathrm{I},1,2} \\\
B_{\mathrm{I},2,1} \\\
B_{\mathrm{I},2,2} \\\
B_{\mathrm{I},3,1} \\\
B_{\mathrm{I},3,2} \\\
\vdots\\\
B_{\mathrm{I},N,1} \\\
B_{\mathrm{I},N,2} 
\end{matrix}
\right].
\end{align*} \nonumber
$$


### Block Matrix Inversion

#### Basic Idea

For a full-rank matrix, if partitioned into four blocks, then it can be inverted blockwise, which will finally generate a recursion chain.

In general, matrix inversion by partition can be expressed as:


$$
\left[
\begin{matrix}
\Delta & \Lambda \\\
\Xi & \Eta
\end{matrix}
\right]^{-1} 
=
\left[
\begin{matrix}
\Delta^{-1}+\Delta^{-1}\Lambda( \Eta - \Xi\Delta^{-1} \Lambda )^{-1} \Xi \Delta^{-1} & \Delta^{-1}\Lambda(\Eta - \Xi\Delta^{-1}\Lambda)^{-1} \\\
-(\Eta - \Xi \Delta^{-1} \Lambda)^{-1}\Xi \Delta^{-1} & (\Eta - \Xi \Delta^{-1} \Lambda)^{-1}
\end{matrix}
\right], \nonumber
$$

which means we can first compute $\Delta^{-1}$, then solve the sub-inversion $(\Eta - \Xi \Delta^{-1} \Lambda)^{-1}$, where we can partition again and again. These steps will form a whole recursion.

#### The Second Level for partitioning the Computation Workload

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

By changing the second dimension of $\mathbb{B}$ and thus $\mathbb{X}$​​, for example,


$$
\begin{align*}
\mathbb{X}^{(2N \times \texttt{sample\_size})} &= 
\left[
\begin{matrix}
\left[\begin{matrix} 
\\\
\\\
\\\
\\\
\\\
\\\
X^{N,1}\\\
\\\
\\\
\\\
\\\
\\\
\end{matrix} \right] & 
\left[\begin{matrix} 
\\\
\\\
\\\
\\\
\\\
\\\
X^{N,2}\\\
\\\
\\\
\\\
\\\
\\\
\end{matrix} \right] & \dots&
\left[\begin{matrix} 
\\\
\\\
\\\
\\\
\\\
\\\
X^{N,\texttt{sample\_size}}\\\
\\\
\\\
\\\
\\\
\\\
\end{matrix} \right]
\end{matrix}
\right],
\qquad \qquad \\
\mathbb{B}_{\mathrm{I}}^{(2N \times \texttt{sample\_size})} &=
\left[
\begin{matrix}
\left[\begin{matrix} 
\\\\
\\\\
\\\\
\\\\
\\\\
\\\\
B_{\mathrm{I}}^{N,1}\\\
\\
\\
\\
\\
\\
\end{matrix} \right] & 
\left[\begin{matrix} 
\\
\\
\\
\\
\\
\\
B_{\mathrm{I}}^{N,2}\\\
\\
\\
\\
\\
\\
\end{matrix} \right] & \dots&
\left[\begin{matrix} 
\\
\\\
\\\
\\\
\\\
B_{\mathrm{I}}^{N,\texttt{sample\_size}}\\\
\\\
\\\
\\\
\\\
\\\
\end{matrix} \right]
\end{matrix}
\right],
\end{align*} \nonumber
$$


we can now scale the sampling batch size as well! 🥳
