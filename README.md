# Sample Average Approximation for Stochastic Programming with Equality Constraints

## About
Code to reproduce the stochastic Mars entry, descent, and landing (EDL) experiment in our paper about sampling-based stochastic programming (T. Lew, R. Bonalli, M. Pavone, "Sample Average Approximation for Stochastic Programming with Equality Constraints", available at [https://arxiv.org/abs/2112.05745](https://arxiv.org/abs/2206.09963), 2022).
<p align="center">
  <img src="figures/mars_descent.png" width="70%"/>
  <br /><em>Mars rocket-powered descent results. Trajectory obtained with a baseline that does not consider uncertainty (top-left) and by solving a stochastic optimal control problem approximated with samples (bottom-left). Right: horizontal and vertical positions at the final time from Monte-Carlo simulations of the system with the controls from the deterministic baseline and from the stochastic algorithm.</em>
</p>
In this work, we apply the sample average approximation (SAA) to general stochastic programs
<p align="left">
  <img src="figures/P.png" width="50%"/>
</p>
The SAA approach consists of approximating this problem by replacing all expectations with Monte-Carlo estimates and solving the resulting deterministic optimization problem. For example, the cost is replaced with 
<p align="left">
  <img src="figures/f_MC.png" width="20%"/>
</p>
In this paper, we prove asymptotic convergence properties of the approach as the number of samples increases. We then apply the approach to stochastic optimal control problems that take the form of
<p align="left">
  <img src="figures/SOCP.png" width="70%"/>
</p>

## Setup
Requires Python >=3.6.  
All dependencies (i.e., numpy, cvxpy, and matplotlib) can be installed by running 
```bash
  pip install -r requirements.txt
```
To reproduce results, run
```bash
  python mars_powered_descent.py
```
This implementation is quite short thanks to [CVXPY](https://www.cvxpy.org/) which easily interfaces with popular solvers such as [ECOS](https://github.com/embotech/ecos). However, this code (mars_powered_descent.py) is not optimized for speed. Real-time implementation could be enabled by (1) directly interfacing with the solver, (2) exploiting the sparsity of the problem (see, e.g., [OSQP](https://osqp.org/)), (3) parallelizing computations on a GPU, etc.
