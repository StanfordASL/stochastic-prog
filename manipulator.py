"""Trajectory optimization for a robotic manipulator."""
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
import ipyopt
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc, rcParams

from benchmark_optimization_problems import *
from benchmark_solver import *

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)

np.random.seed(3)


class Manipulator:
    def __init__(self):
        # state and control dimension
        self.n_q = 3
        self.n_u = self.n_q
        # control bounds
        self.umax = 10.
        self.umin = -self.umax
        # joint lengths
        self.l1 = 0.4
        self.l2 = 0.3
        self.l3 = 0.3

    @partial(jit, static_argnums=(0,))
    def p_01(self, q, delta_link_length_1=0.):
        cq1, sq1 = jnp.cos(q[0]), jnp.sin(q[0])
        return (self.l1 + delta_link_length_1) * jnp.array([cq1, sq1])

    @partial(jit, static_argnums=(0,))
    def p_02(self, q, delta_link_lengths_12=jnp.zeros(2)):
        delta_link_length_1 = delta_link_lengths_12[0]
        delta_link_length_2 = delta_link_lengths_12[1]
        cq2, sq2 = jnp.cos(q[1]), jnp.sin(q[1])
        p_12 = (self.l2 + delta_link_length_2) * jnp.array([cq2, sq2])
        return self.p_01(q, delta_link_length_1) + p_12

    @partial(jit, static_argnums=(0,))
    def p_ee(self, q, delta_link_lengths_123=jnp.zeros(3)):
        delta_link_lengths_12 = delta_link_lengths_123[:2]
        delta_link_length_3 = delta_link_lengths_123[2]
        cq3, sq3 = jnp.cos(q[2]), jnp.sin(q[2])
        p_23 = (self.l3 + delta_link_length_3) * jnp.array([cq3, sq3])
        return self.p_02(q, delta_link_lengths_12) + p_23

    @partial(jit, static_argnums=(0,))
    def b(self, q, u):
        return u


class ManipulatorProblem(Program):
    def __init__(
            self,
            model,
            S,
            q0=0.75 * np.pi * np.array([1. / 3, 2. / 3, 1.]),
            qg=np.pi / 8 * np.ones(3)):#np.pi / 6 * np.ones(3)):
        self.model = model
        # number of control switches
        self.S = S
        # initial and final states
        self.q0 = q0
        self.qg = qg
        #
        self.n_q, self.n_u = model.n_q, model.n_u

        self.nb_vars = (self.S-1)*self.n_u + self.S*self.n_q

        num_equality_constraints = len(self.equality_constraints(
            self.initial_guess()))
        num_inequality_constraints = len(self.inequality_constraints(
            self.initial_guess())[0])

        super().__init__(
            name="manipulator_problem",
            num_variables=self.nb_vars,
            num_equality_constraints=num_equality_constraints,
            num_inequality_constraints=num_inequality_constraints)

    def convert_x_to_qs_us(self, x):
        n_q, n_u, S = self.n_q, self.n_u, self.S
        us = x[:(S-1)*n_u]
        qs = x[(S-1)*n_u:]
        us = np.reshape(us, (n_u, S-1), 'F')
        qs = np.reshape(qs, (n_q, S), 'F')
        us = us.T # (S-1,n_u)
        qs = qs.T # (S,n_q)
        return (qs, us)

    def initial_constraints(self, x):
        n_q, n_u, S = self.n_q, self.n_u, self.S
        idx_q0  = (S - 1) * n_u
        q0 = x[idx_q0:(idx_q0+n_q)]
        constraint = q0 - self.q0
        return 1e3 * constraint

    def final_constraints(self, x, delta_link_lengths=jnp.zeros(3)):
        n_q, n_u, S = self.n_q, self.n_u, self.S

        idx_qT = (S - 1) * n_u + (S - 1) * n_q
        qp_T = x[idx_qT:]
        pp_ee = self.model.p_ee(qp_T, delta_link_lengths)
        p_ee_goal = self.model.p_ee(self.qg)
        constraint = pp_ee - p_ee_goal
        return constraint

    def dynamics_constraints(self, x):
        n_q, n_u, S = self.n_q, self.n_u, self.S

        constraints = jnp.zeros((S - 1) * n_q)
        for t in range(S-1):
            idx_con = t*n_q
            idx_ut  = t*n_u
            idx_utn = idx_ut + n_u
            idx_qt  = (S-1)*n_u + t*n_q
            idx_qtn = idx_qt + n_q

            q_t = x[idx_qt:idx_qtn]
            u_t = x[idx_ut:idx_utn]
            q_tn = x[idx_qtn:(idx_qtn+n_q)]

            constraints = constraints.at[idx_con:(idx_con+n_q)].set(
                q_tn - (q_t + u_t))
        return 1e3 * constraints

    def initial_guess(self) -> np.array:
        n_q, n_u, S = self.n_q, self.n_u, self.S

        # initial guess (linearization point) (straight-line)
        x = np.zeros(self.nb_vars)
        for t in range(S):
            idx_ut = t*n_u
            x[idx_ut:idx_ut+n_u] = 1e-5 * np.random.randn(n_u)

            idx_qt = (S-1)*n_u + t*n_q
            alpha1 = ( (S-1) - t ) / (S-1)
            alpha2 =       t       / (S-1)
            x[idx_qt:(idx_qt+n_q)] = self.q0 * alpha1 + self.qg * alpha2 + 1e-6
        return x

    def objective(self, x: jnp.array) -> float:
        n_q, n_u, S = self.n_q, self.n_u, self.S

        f_value = 0.
        for t in range(S - 1):
            idx_ut = t*n_u
            u_t = x[idx_ut:(idx_ut+n_u)]
            f_value = f_value + jnp.sum(u_t**2)
        return f_value

    def equality_constraints(
            self,
            x: jnp.array) -> jnp.array:
        constraints = jnp.concatenate([
            self.initial_constraints(x),
            self.final_constraints(x),
            self.dynamics_constraints(x)])
        return constraints

    def inequality_constraints(
            self,
            x: jnp.array) -> jnp.array:
        _, us = self.convert_x_to_qs_us(x)
        us = us.flatten()
        gx = us
        gl = self.model.umin * jnp.ones_like(us)
        gu = self.model.umax * jnp.ones_like(us)
        return gx, gl, gu


class SampledManipulatorProblem(Program):
    def __init__(
            self,
            deterministic_program: Program,
            sample_size: int = 20):
        p = deterministic_program
        super().__init__(
            name=p.name,
            num_variables=p.num_variables,
            num_equality_constraints=p.num_equality_constraints,
            num_inequality_constraints=p.num_inequality_constraints)
        self._deterministic_program = p
        self._sample_size = sample_size
        self._omegas = self.sample_omegas()

    def sample_omegas(self) -> np.array:
        omega_std = 0.01
        omegas_bias = np.random.normal(
            loc=0, scale=omega_std,
            size=(self._sample_size, 5))
        return omegas_bias

    def convert_x_to_qs_us(self, x):
        return self._deterministic_program.convert_x_to_qs_us(x)

    def initial_guess(self) -> np.array:
        return self._deterministic_program.initial_guess()

    def objective(self, x: jnp.array) -> float:
        return self._deterministic_program.objective(x)

    def equality_constraints(
            self,
            x: jnp.array) -> jnp.array:

        def perturbed_equality_constraint(
                x: jnp.array,
                omega: jnp.array):
            p = self._deterministic_program
            constraints = jnp.concatenate([
                p.initial_constraints(x),
                p.final_constraints(x, omega[:3]) + omega[3:],
                p.dynamics_constraints(x)])
            return constraints

        hs_value = vmap(
            perturbed_equality_constraint, in_axes=(None, 0))(
            x, self._omegas)
        hs_value = jnp.mean(hs_value, axis=0)
        return hs_value

    def inequality_constraints(
            self,
            x: jnp.array):
        return self._deterministic_program.inequality_constraints(x)


model = Manipulator()
S = 15

nominal_problem = ManipulatorProblem(model, S=S)
program = nominal_problem
# program = SampledManipulatorProblem(nominal_problem, sample_size=10)
solver_saa = Solver(
    program=program,
    delta_equality_relaxation=0,
    verbose=True)
out_saa = solver_saa.solve()
x_saa = out_saa[0]
obj_saa = out_saa[1]
status_saa = out_saa[2]
xs, us = program.convert_x_to_qs_us(x_saa)


ptraj_1 = np.zeros((S, 2))
ptraj_2 = np.zeros((S, 2))
ptraj_3 = np.zeros((S, 2))
for t in range(S):
    q = xs[t,:]
    ptraj_1[t,:] = model.p_01(q)
    ptraj_2[t,:] = model.p_02(q)
    ptraj_3[t,:] = model.p_ee(q)
x0_ee = model.p_ee(xs[0,:])
xg_ee = model.p_ee(xs[-1,:])


# plot
s, lw = 80, 4
fig = plt.figure(figsize=[8, 8])
plt.scatter(x0_ee[0], x0_ee[1], s=s, color='k')
plt.scatter(xg_ee[0], xg_ee[1], s=s, color='k')
plt.scatter(ptraj_1[:, 0], ptraj_1[:, 1],
    s=s, c='g', alpha=1)
plt.scatter(ptraj_2[:, 0], ptraj_2[:, 1],
    s=s, c='g', alpha=1)
plt.scatter(ptraj_3[:, 0], ptraj_3[:, 1],
    s=s, c='g', alpha=1)

chains  = np.zeros((S, 3+1, 2))
for t in range(S):
    chains[t, 1, :] = ptraj_1[t,:]
    chains[t, 2, :] = ptraj_2[t,:]
    chains[t, 3, :] = ptraj_3[t,:]
for t in range(1, S-1):
    color = 'g'
    plt.plot(chains[t, :, 0], chains[t, :, 1],
        color=color, linewidth=lw, alpha=1)
for t in [0, S - 1]:
    color = 'k'
    plt.plot(chains[t, :, 0], chains[t, :, 1],
        color=color, linewidth=lw, alpha=1)
    plt.scatter(ptraj_1[t, 0], ptraj_1[t, 1],
        s=s, c=color, alpha=1)
    plt.scatter(ptraj_2[t, 0], ptraj_2[t, 1],
        s=s, c=color, alpha=1)
    plt.scatter(ptraj_3[t, 0], ptraj_3[t, 1],
        s=s, c=color, alpha=1)
plt.xlim([-0.05, 1])
plt.ylim([-0.05, 1])
plt.xlabel(r'$p_x(q)$', fontsize=30)
plt.ylabel(r'$p_y(q)$', fontsize=30)
plt.grid()
plt.tight_layout()
plt.show()



num_repeats = 100
relaxation_delta_M = 1e-2
successes_saa = np.zeros((num_repeats), dtype=bool)
successes_saa_rlx = np.zeros((num_repeats), dtype=bool)
for i in range(num_repeats):
    print("Repeat", i, " /", num_repeats)
    program = SampledManipulatorProblem(nominal_problem, sample_size=10)

    # standard SAA
    solver_saa = Solver(
        program=program,
        delta_equality_relaxation=0)
    out = solver_saa.solve()
    status = out[2]
    if status == SolverReturnStatus.Solve_Succeeded:
        successes_saa[i] = True

    # relaxed SAA
    solver_saa_relaxed = Solver(
        program=program,
        delta_equality_relaxation=relaxation_delta_M)
    out = solver_saa_relaxed.solve()
    status = out[2]
    if status == SolverReturnStatus.Solve_Succeeded:
        successes_saa_rlx[i] = True

print("successes_saa     =", np.sum(successes_saa))
print("successes_saa_rlx =", np.sum(successes_saa_rlx))
