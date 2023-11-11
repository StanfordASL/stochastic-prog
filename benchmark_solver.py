import enum
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, grad, hessian, jit, vmap
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
import ipyopt
from time import time

from benchmark_optimization_problems import *
 

class SolverReturnStatus(enum.Enum):
    Solve_Succeeded                    = 0
    Solved_To_Acceptable_Level         = 1
    Infeasible_Problem_Detected        = 2
    Search_Direction_Becomes_Too_Small = 3
    Diverging_Iterates                 = 4
    User_Requested_Stop                = 5
    Feasible_Point_Found               = 6
    #
    Maximum_Iterations_Exceeded        = -1
    Restoration_Failed                 = -2
    Error_In_Step_Computation          = -3
    Maximum_CpuTime_Exceeded           = -4
    Maximum_WallTime_Exceeded          = -5
    #
    Not_Enough_Degrees_Of_Freedom      = -10
    Invalid_Problem_Definition         = -11
    Invalid_Option                     = -12
    Invalid_Number_Detected            = -13
    #
    Unrecoverable_Exception            = -100
    NonIpopt_Exception_Thrown          = -101
    Insufficient_Memory                = -102
    Internal_Error                     = -199


class Solver:
    def __init__(
            self,
            program: Program,
            name: str = "ipopt",
            delta_equality_relaxation: float = 0,
            verbose: bool = False):
        ipopt_print_level = 0
        if verbose:
            print("Initializing Solver with")
            print("> name    =", name)
            print("> program =", program)
            ipopt_print_level = 5
            
        # if program not in programs:
        #     msg = "This program is unknown."
        #     raise ValueError(msg)
        if name != "ipopt":
            msg = "This solver is not supported."
            raise NotImplementedError(msg)
        self._program = program
        self._name = name
        self._delta_relaxation = delta_equality_relaxation
        self._ipopt_options = {
            'max_iter': 3000,
            'tol': 1e-8,
            'print_level': ipopt_print_level
            }
        # Initial bounds for IPOPT
        nvar = program.num_variables
        self._x_bounds_low = -np.ones(
            nvar, dtype=np.float_) * 1e3
        self._x_bounds_up = np.ones(
            nvar, dtype=np.float_) * 1e3
        self.define_optimization_problem()

    @partial(jit, static_argnums=(0,))
    def f(self, x: jnp.array) -> float:
        # Objective to minimize
        program = self._program
        assert len(x) == program.num_variables
        return program.objective(x)
    
    @partial(jit, static_argnums=(0,))
    def g(self, x: jnp.array) -> jnp.array:
        # Constraints of the form gL <= g(x) <= gU
        program = self._program
        assert len(x) == program.num_variables
        hs = program.equality_constraints(x)
        gs, _, _ = program.inequality_constraints(x)
        constraints = jnp.concatenate([hs, gs])
        return constraints

    def gL_gU(self, x: jnp.array) -> np.array:
        # -> tuple[jnp.array, jnp.array, jnp.array]:

        # note: this code below, copy-pasted from g(x),
        # is there to get the dimensions of each constraints
        # and define gL, gU such that
        #   gL <= g(x) <= gU
        # (gL, gU) are then passed to IPOPT
        program = self._program
        hs = program.equality_constraints(x)
        gs, gs_l, gs_u = program.inequality_constraints(x)
        num_equality_constraints = len(hs)
        # set constraints
        constraints = np.concatenate([hs, gs])
        constraints_l = np.zeros_like(constraints)
        constraints_u = np.zeros_like(constraints)
        # equality constraints (0 <= h(x) <= 0)
        delta_M = self._delta_relaxation
        if delta_M > 0:
            delta_M = self._delta_relaxation
            constraints_l[:num_equality_constraints] = -delta_M
            constraints_u[:num_equality_constraints] = delta_M
        # inequality constraints (gs_l <= g(x) <= gs_u)
        constraints_l[num_equality_constraints:] = gs_l
        constraints_u[num_equality_constraints:] = gs_u
        return constraints_l, constraints_u
    
    @partial(jit, static_argnums=(0,))
    def f_dx(self, x: jnp.array) -> jnp.array:
        return grad(self.f)(x)
    
    @partial(jit, static_argnums=(0,))
    def f_ddx_hessian(self, x: jnp.array) -> jnp.array:
        return hessian(self.f)(x)
    
    @partial(jit, static_argnums=(0,))
    def g_dx(self, x: jnp.array) -> jnp.array:
        return jacfwd(self.g)(x)

    @partial(jit, static_argnums=(0,))
    def lagrange_dot_g(
            self,
            x: jnp.array,
            lagrange: jnp.array) -> float:
        return jnp.dot(lagrange, self.g(x))

    @partial(jit, static_argnums=(0,))
    def hessian_lagrange_dot_g(
            self,
            x: jnp.array,
            lagrange: jnp.array) -> jnp.array:
        nvar = self._program.num_variables
        hess = jacfwd(jacfwd(self.lagrange_dot_g))(
            x, lagrange)[:nvar,:nvar]
        return hess

    def eval_f(self, x: np.array) -> float:
        return self.f(x)

    def eval_grad_f(
            self,
            x: np.array,
            out: np.array) -> np.array:
        out[:] = self.f_dx(x)
        return out

    def eval_g(
            self,
            x: np.array,
            out: np.array) -> np.array:
        out[:] = self.g(x)
        return out

    def eval_jac_g(
            self,
            x: np.array,
            out: np.array) -> np.array:
        out[:] = self.g_dx(x).flatten()
        return out

    def eval_h(
            self,
            x: np.array,
            lagrange: np.array,
            obj_factor: np.array,
            out: np.array) -> np.array:
        nvar = self._program.num_variables
        hess_cost = self.f_ddx_hessian(x)
        hess_lagrange_prod = self.hessian_lagrange_dot_g(
            x, lagrange)
        hess_cost = hess_cost[np.tril_indices(nvar)].flatten()
        hess_lagrange_prod = hess_lagrange_prod[np.tril_indices(nvar)].flatten()
        out[:] = obj_factor * hess_cost + hess_lagrange_prod
        return out

    def get_sparsity_indices(self):
        # below, we specify that all matrices are dense,
        program = self._program
        nvar = program.num_variables
        ncon = program.num_constraints
        indices_1, indices_2 = np.indices((ncon, nvar))
        jac_g_sparsity_indices = (
            indices_1.flatten(), indices_2.flatten())
        indices_1, indices_2 = np.tril_indices(nvar)
        h_sparsity_indices = (
            indices_1.flatten(), indices_2.flatten())
        return jac_g_sparsity_indices, h_sparsity_indices

    def define_optimization_problem(self):
        program = self._program
        nvar = program.num_variables
        ncon = program.num_constraints
        x0 = program.initial_guess()
        g_L, g_U = self.gL_gU(x0)
        sparsity_indices = self.get_sparsity_indices()
        (eval_jac_g_sparsity_indices, eval_h_sparsity_indices) = (
            sparsity_indices)
        # Precompile
        _ = self.f(x0), self.g(x0), self.f_dx(x0), self.g_dx(x0)
        _ = self.f_ddx_hessian(x0)
        _ = self.hessian_lagrange_dot_g(x0, np.zeros_like(self.g(x0)))
        self.nlp = ipyopt.Problem(
            nvar,
            self._x_bounds_low,
            self._x_bounds_up,
            ncon,
            g_L,
            g_U,
            eval_jac_g_sparsity_indices,
            eval_h_sparsity_indices,
            self.eval_f,
            self.eval_grad_f,
            self.eval_g,
            self.eval_jac_g,
            ipopt_options=self._ipopt_options,
            eval_h=self.eval_h)

    def solve(self, B_return_computation_time=False):
        """ TODO ADD DOCSTRING"""
        x0 = self._program.initial_guess()
        nvar = self._program.num_variables
        ncon = self._program.num_constraints
        start_time = time()
        x, obj, status = self.nlp.solve(
            x0, 
            mult_g=np.zeros(ncon), 
            mult_x_L=np.zeros(nvar), 
            mult_x_U=np.zeros(nvar))
        elapsed_time = time() - start_time
        if not(B_return_computation_time):
            return x, obj, SolverReturnStatus(status)
        else:
            return x, obj, SolverReturnStatus(status), elapsed_time



if __name__=="__main__":
    print("[benchmark_solver.py]")
    p = Program6()
    solver = Solver(program=p)
    x, obj, status = solver.solve()
    out = solver.solve(B_return_computation_time=True)
    x, obj, status, elapsed_time = out
    sampled_p = SampledProgram(
        p, 
        sample_size=2000,
        add_bias=True)
    solver = Solver(
        program=sampled_p,
        delta_equality_relaxation=0)
    out = solver.solve()

    print("Test successful.")