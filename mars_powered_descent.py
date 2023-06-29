import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import gridspec

import jax.numpy as jnp
from jax import jacrev, grad, hessian, jit, vmap
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

np.random.seed(1)

S = 21 # number of controls
N = 20 # number of samples
delta_N = 1e-5
N_scp_iters = 10 # number of SCP iterations

n_x = 7 # (rx,ry,rz,vx,vy,vz,m)
n_u = 4 # (ux,uy,uz,sigma) (sigma(t)=||u(t)|| under some conditions, see [Acikmese & Ploen, 2007])
class Rocket:
    def __init__(self, N, S, delta_N, B_uncertainty=True):
        self.N = N # number of samples
        self.S = S # number of control switches
        # Planning time horizon
        self.T = 60. # sec
        self.dt = self.T / (self.S-1)
        # Initial / final states
        self.n_x = 7
        self.n_u = 4
        # Values inspired from the Mars rover MSL Curiosity landing 
        # (up to sky-crane separation)
        # https://arc.aiaa.org/doi/pdf/10.2514/1.A32866 (Steltzner et al., 2014)
        # https://spaceflight101.com/msl/msl-landing-special/
        self.x0 = np.array([
            300, 0, 1500, 5, 0, -75.,
            1800])
        self.xg = np.array([
            0, 0, 100, 0, 0, -10.])

        # Parameters approx. from https://hal-ensta-paris.archives-ouvertes.fr//hal-03641631/document
        # Clara Leparoux, Bruno Hérissé, Frédéric Jean, Optimal planetary landing with pointing and 
        # glide-slope constraints. 2022.
        self.Thrust_max = 16000 # N
        self.q = 8.0 # kg/s
        self.umin = 0.3
        self.umax = 0.8
        self.g = 3.71 # m/s^2 (Mars)
        self.gamma = 35.0 * (np.pi / 180.0) # glide slope
        self.theta = 45.0 * (np.pi / 180.0) # max thrust angle
        # Uncertainty
        self.Cdrag = 1.0
        self.beta0 = 1e1
        self.beta1 = 2e-1
        # Samples from Brownian motion
        self.DWs = np.zeros((self.N,self.S,self.n_x))
        if B_uncertainty == True:
            for i in range(self.N):
                for t in range(self.S):
                    self.DWs[i,t,:] = np.sqrt(self.dt)*np.random.randn(n_x)
        # Constraints relaxation constant
        self.delta_N = delta_N
    @partial(jit, static_argnums=(0,))
    def convert_Z_to_xs_us(self, Z):
        S, N, n_x, n_u = self.S, self.N, self.n_x, self.n_u
        us = Z[:(S-1)*n_u]
        xs = Z[(S-1)*n_u:]
        us = jnp.reshape(us, (n_u, S-1), 'F')
        xs = jnp.reshape(xs, (n_x, N, S), 'F')
        us = us.T # (S-1, n_u)
        xs = jnp.moveaxis(xs, 0, -1) # (N, S, n_x)
        return (xs, us)
    @partial(jit, static_argnums=(0,))
    def b(self, x, u):
        Tmax, Cd, q = self.Thrust_max, self.Cdrag, self.q
        v, mass = x[3:6], x[-1]
        unorm = jnp.linalg.norm(u[:3])
        p_dot = v
        v_dot = Tmax*u[:3]/mass - jnp.array([0.,0.,self.g])
        v_dot = v_dot + (1.0/mass)*(-Cd*jnp.abs(v)*v)
        m_dot = -q * unorm * jnp.ones(1)
        bvec = jnp.concatenate((
            p_dot, v_dot, m_dot), axis=-1)
        return bvec
    @partial(jit, static_argnums=(0,))
    def sigma(self, x, u):
        b0, b1 = self.beta0, self.beta1
        v, mass, unorm = x[3:6], x[-1], jnp.linalg.norm(u[:3])
        sdiag = (1.0/mass) * (b0 + b1*v**2)
        smat = jnp.zeros((self.n_x, self.n_x))
        smat = smat.at[3:6, 3:6].set(jnp.diag(sdiag))
        return smat
    def initial_constraints(self):
        S, N = self.S, self.N
        x0, n_x, n_u = self.x0, self.n_x, self.n_u
        nb_vars = (S-1)*n_u + S*N*n_x
        Aineq = np.zeros((N*n_x, nb_vars))
        for i in range(N):
            idx_x0 = (S-1)*n_u + i*n_x
            idx_x0n = idx_x0 + n_x
            Aineq[i*n_x:(i+1)*n_x,idx_x0:idx_x0n] = np.eye(n_x)
        lineq = np.hstack([self.x0 for i in range(N)])
        uineq = np.hstack([self.x0 for i in range(N)])
        return Aineq, lineq, uineq
    def final_constraints(self):
        S, N = self.S, self.N
        xg, n_x, n_u = self.xg, self.n_x, self.n_u
        nb_vars = (S-1)*n_u + S*N*n_x
        Aineq = np.zeros((6, nb_vars))
        for i in range(N):
            idx_xf = (S-1)*n_u + (S-1)*N*n_x + i*n_x
            idx_xfn = idx_xf + 6 # no final mass constraint
            Aineq[:, idx_xf:idx_xfn] = np.eye(6) / N
        lineq = xg - self.delta_N
        uineq = xg + self.delta_N
        return Aineq, lineq, uineq
    @partial(jit, static_argnums=(0,))
    def dynamics_constraints_jax(self, Z):
        S, N, dt = self.S, self.N, self.dt
        n_x, n_u = self.n_x, self.n_u
        def dynamic_constraint(x, u, xn, w):
            # modified Euler scheme
            # Numerical integration of SDEs
            # Renfeng Cao and Stephen B. Pope
            # Journal of Computational Physics 185 (2003) 194–212
            # https://tcg.mae.cornell.edu/pubs/Cao_P_JCP_03.pdf
            x_mid = (x + 0.5 * dt * self.b(x, u))
            x_pred = x + dt*self.b(x_mid, u) + self.sigma(x_mid, u) @ w
            return (xn - x_pred)
        xs, us = self.convert_Z_to_xs_us(Z) # (N, S, n_x) and (S-1, n_u)
        Xs = xs[:, :-1, :]
        Xns = xs[:, 1:, :]
        Ws = jnp.array(self.DWs[:, :(S-1), :])
        gs = vmap(vmap(dynamic_constraint), in_axes=(0, None, 0, 0))(
            Xs, us, Xns, Ws)
        return gs.flatten()
    @partial(jit, static_argnums=(0,))
    def dynamics_constraints_jax_dZ(self, Z):
        return jacrev(self.dynamics_constraints_jax)(Z)
    def dynamics_constraints(self, Zp):
        Zp_jax = jnp.array(Zp)
        gdyn_p = model.dynamics_constraints_jax(Zp_jax)
        gdyn_dZ_p = model.dynamics_constraints_jax_dZ(Zp_jax)
        Aeq = gdyn_dZ_p
        leq = -gdyn_p + gdyn_dZ_p@Zp
        ueq = leq
        return Aeq.to_py(), leq.to_py(), ueq.to_py()

class convexified_problem:
    def __init__(self, model):
        n_x, n_u, S, N, dt = model.n_x, model.n_u, model.S, model.N, model.dt
        nb_vars = (S-1)*n_u + S*N*n_x
        self.n_x, self.n_u, self.S, self.N, self.dt = model.n_x, model.n_u, model.S, model.N, model.dt
        self.model = model
        self.nb_vars = nb_vars
    def define(self, Zp):
        n_x, n_u, S, N, dt = self.n_x, self.n_u, self.S, self.N, self.dt
        nb_vars = (S-1)*n_u + S*N*n_x
        model = self.model
        nb_vars = self.nb_vars
        # Define and solve the CVXPY problem
        Z = cp.Variable(nb_vars)
        # Objective
        obj = 0.
        # control
        for t in range(S-1):
            idx_ut = t*n_u
            obj = obj + dt*Z[idx_ut+3]
        # Final deviation to landing
        # if model.xg=E[r_T], then 
        # sum_squares(r_T-model.xg) gives 
        # the trace of the covariance matrix
        for i in range(N):
            idx = (S-1)*n_u + (S-1)*N*n_x + i*n_x
            r_T = Z[idx:(idx+6)]
            obj = obj + 1e-2 * (1.0/N)*cp.sum_squares(r_T[:6]-model.xg[:6])
        obj = cp.Minimize(obj)
        # Constraints
        Aeq_x0, leq_x0, ueq_x0 = model.initial_constraints()
        Aeq_xf, leq_xf, ueq_xf = model.final_constraints()
        Aeq_dyn, leq_dyn, _ = model.dynamics_constraints(Zp)
        con = []
        con.append(Aeq_x0@Z==leq_x0)
        con.append(leq_xf <= Aeq_xf@Z)
        con.append(Aeq_xf@Z <= ueq_xf)
        con.append(Aeq_dyn@Z==leq_dyn)
        # control
        for t in range(S-1):
            idx_ut = t*n_u
            u_t = Z[idx_ut:(idx_ut+3)]
            uz_t, sigma_t = Z[idx_ut+2], Z[idx_ut+3]
            con.append(model.umin<=sigma_t)
            con.append(sigma_t<=model.umax)
            con.append(uz_t>=sigma_t*np.cos(model.theta))
            # cp.SOC(t, x) creates the SOC constraint ||x||_2 <= t.
            con.append(cp.SOC(sigma_t, u_t))
        # altitude
        tan_gamma = np.tan(model.gamma)
        S_max = S-2
        for t in range(S_max):
            xy_t_avg, z_t_avg = 0, 0
            for i in range(N):
                idx_xt = (S-1)*n_u + t*N*n_x + i*n_x
                xy_t = Z[idx_xt:idx_xt+2]
                z_t = Z[idx_xt+2]
                xy_t_avg += xy_t
                z_t_avg += z_t
            xy_t_avg = xy_t_avg / N
            z_t_avg = z_t_avg / N
            con.append( tan_gamma*xy_t_avg[0] - z_t_avg <= delta_N)
            con.append( tan_gamma*xy_t_avg[1] - z_t_avg <= delta_N)
            con.append(-tan_gamma*xy_t_avg[0] - z_t_avg <= delta_N)
            con.append(-tan_gamma*xy_t_avg[1] - z_t_avg <= delta_N)
        self.Z = Z
        self.prob = cp.Problem(obj, con)
    def solve(self):
        self.prob.solve(solver=cp.ECOS, verbose=False)#, ignore_dpp=True)
    def initial_guess(self):
        n_x, n_u, S, N, dt = self.n_x, self.n_u, self.S, self.N, self.dt
        # initial guess (linearization point) (straight-line)
        Zp = np.zeros(self.nb_vars)
        for t in range(S-1):
            idx_ut = t*n_u
            Zp[idx_ut:idx_ut+3] = (model.umin + model.umax) / 2.0
            Zp[idx_ut+3] = np.linalg.norm(Zp[idx_ut:idx_ut+3])
        for t in range(S):
            for i in range(N):
                idx_xt = (S-1)*n_u + t*N*n_x + i*n_x
                alpha1 = ((S - 1) - t ) / (S - 1)
                alpha2 = t / (S-1)
                Zp[idx_xt:idx_xt+6] = model.x0[:6] * alpha1 + model.xg * alpha2 + 1e-6
                Zp[idx_xt+6] = model.x0[-1] # mass
        return Zp
    def extract_solution(self):
        xs, us = self.convert_Z_to_xs_us(self.Z.value)
        return (self.Z.value, xs, us)
    def convert_Z_to_xs_us(self, Z):
        us = Z[:(S-1)*n_u]
        xs = Z[(S-1)*n_u:]
        us = np.reshape(us, (n_u, S-1), 'F')
        xs = np.reshape(xs, (n_x, N, S), 'F')
        us = us.T # (S-1, n_u)
        xs = np.moveaxis(xs, 0, -1) # (N, S, n_x)
        return (xs, us)
    def error_criterion(self, Z, Zp):
        err = np.linalg.norm(Z-Zp)/np.linalg.norm(Zp)
        return err

# -----------------------------------------
model = Rocket(N, S, delta_N)
nb_vars = (S-1)*n_u + S*N*n_x
dt = model.T / (S-1)
# -----------------------------------------
# --------- with uncertainty --------------
print(">>> solving stochastic program")
print("if too slow, consider reducing the sample size N.")
convex_problem = convexified_problem(model)
Zp = convex_problem.initial_guess()
for scp_iter in range(N_scp_iters):
    print("SCP iter. "+str(scp_iter+1)+"/"+str(N_scp_iters))
    convex_problem.define(Zp)
    convex_problem.solve()
    Z, xs, us = convex_problem.extract_solution()
    print("error =", convex_problem.error_criterion(Z, Zp))
    Zp = Z.copy()
# -----------------------------------------
# ---------  no  uncertainty --------------
print(">>> solving deterministic program")
model = Rocket(N, S, delta_N, B_uncertainty=False)
convex_problem = convexified_problem(model)
Zp = convex_problem.initial_guess()
for scp_iter in range(N_scp_iters):
    print("SCP iter. "+str(scp_iter+1)+"/"+str(N_scp_iters))
    convex_problem.define(Zp)
    convex_problem.solve()
    Z, xs_det, us_det = convex_problem.extract_solution()
    print("error =", convex_problem.error_criterion(Z, Zp))
    Zp = Z.copy()
# -----------------------------------------
print("Stochastic: final mass =", np.mean(xs[:,-1,-1], axis=0), "[kg]")
print("Deterministic: final mass =", np.mean(xs_det[:,-1,-1], axis=0), "[kg]")


########### PLOT (deterministic) ###########################
# plot
fig = plt.figure(figsize=[10,3])
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1]) 

plt.subplot(gs[0])
plt.scatter(model.x0[0],model.x0[2], color='k')
plt.scatter(model.xg[0],model.xg[2], color='k')
# mean trajectory and controls
x_traj_mean = np.zeros((S,n_x))
for t in range(S-1):
    xt, ut = np.mean(xs_det[:,t,:],axis=0), us_det[t,:]
    if t == 0:
        plt.plot([xt[0], xt[0]+400*ut[0]],
                 [xt[2], xt[2]+400*ut[2]],
                 c='b', alpha=0.3,
                 label=r'$u(t)$')
    else:
        plt.plot([xt[0], xt[0]+400*ut[0]],
                 [xt[2], xt[2]+400*ut[2]],
                 c='b',alpha=0.3)
    x_traj_mean[t,:] = xt.copy()
x_traj_mean[-1,:] = np.mean(xs_det[:,-1,:],axis=0)
# mean trajectory
plt.plot(x_traj_mean[:,0], x_traj_mean[:,2], 
         c='b', label=r'$\mathbb{E}[x_u(t)]$')
# glide-slope
xs_glideslope = [model.xg[0],model.xg[0]+np.max(xs_det[:,:,0])] 
ys_glideslope = [0,0+(np.max(xs_det[:,:,0])-model.xg[0])*np.sin(model.gamma)]
plt.plot(xs_glideslope, ys_glideslope, 'r--') 
plt.fill_between(xs_glideslope, [-100,-100], ys_glideslope, color='r', alpha=0.2)
plt.xlabel(r'$r_x$', fontsize=16)
plt.ylabel(r'$r_z$', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()

plt.subplot(gs[1])
plt.step(dt*np.arange(us_det.shape[0]),
    np.linalg.norm(us_det[:,:3], axis=1, ord=2),
    where='post')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$||u(t)||$', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()

plt.subplot(gs[2])
plt.plot(dt*np.arange(S), x_traj_mean[:,-1], c='b')
plt.grid()
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$m(t)$', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
fig.savefig('figures/deterministic.png')
plt.close()


########### PLOT (stochastic) ########################

# plot
fig = plt.figure(figsize=[10,3])
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1]) 

plt.subplot(gs[0])
plt.scatter(model.x0[0], model.x0[2], color='k')
plt.scatter(model.xg[0], model.xg[2], color='k')
# mean trajectory and controls
x_traj_mean = np.zeros((S,n_x))
for t in range(S-1):
    xt, ut = np.mean(xs[:,t,:],axis=0), us[t,:]
    if t == 0:
        plt.plot([xt[0], xt[0]+400*ut[0]],
                 [xt[2], xt[2]+400*ut[2]],
                 c='b', alpha=0.3,
                 label=r'$u(t)$')
    else:
        plt.plot([xt[0], xt[0]+400*ut[0]],
                 [xt[2], xt[2]+400*ut[2]],
                 c='b',alpha=0.3)
    x_traj_mean[t,:] = xt.copy()
x_traj_mean[-1,:] = np.mean(xs[:,-1,:],axis=0)
# mean trajectory
plt.plot(x_traj_mean[:,0], x_traj_mean[:,2], 
         c='b', label=r'$\mathbb{E}[x_u(t)]$')
# glide-slope
xs_glideslope = [model.xg[0],model.xg[0]+np.max(xs[:,:,0])] 
ys_glideslope = [0,0+(np.max(xs[:,:,0])-model.xg[0])*np.sin(model.gamma)]
plt.plot(xs_glideslope, ys_glideslope, 'r--') 
plt.fill_between(xs_glideslope, [-100,-100], ys_glideslope, color='r', alpha=0.2)
plt.xlabel(r'$r_x$', fontsize=16)
plt.ylabel(r'$r_z$', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()

plt.subplot(gs[1])
plt.step(dt*np.arange(us.shape[0]),
    np.linalg.norm(us[:,:3], axis=1, ord=2),
    where='post')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$||u(t)||$', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()

plt.subplot(gs[2])
plt.plot(dt*np.arange(S), x_traj_mean[:,-1], c='b')
plt.grid()
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$m(t)$', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
fig.savefig('figures/stochastic.png')
plt.close()

########### MONTE-CARLO ####################################
N_MC = 10000
ws = np.zeros((N_MC, S-1, n_x))
for i in range(N_MC):
    for t in range(S-1):
        ws[i, t, :] = np.sqrt(model.dt)*np.random.randn(n_x)
@jit
def simulate_state_trajectory_monte_carlo(us, ws):
    # us - (S-1, n_u) (control trajectory)
    # ws - (S-1, n_x) (Brownian motion sample path)
    xs = jnp.zeros((S, n_x))
    xs = xs.at[0, :].set(model.x0)
    for t in range(S-1):
        xt, ut, dW = xs[t, :], us[t, :], ws[t, :]
        x_mid = xt + 0.5 * dt * model.b(xt, ut)
        xs = xs.at[t+1,: ].set(
            xt + 
            dt * model.b(x_mid, ut) + 
            model.sigma(x_mid, ut) @ dW)
    return xs
xs_MC = vmap(simulate_state_trajectory_monte_carlo, 
    in_axes=(None, 0))(us, ws)
xs_det_MC = vmap(simulate_state_trajectory_monte_carlo, 
    in_axes=(None, 0))(us_det, ws)

# plot results
N_MC_to_plot = 2000
fig = plt.figure(figsize=[6,6])
plt.grid(linestyle='--')
plt.scatter( 
    xs_det_MC[:N_MC_to_plot, -1, 0], 
    xs_det_MC[:N_MC_to_plot, -1, 2],
    c='tab:orange', alpha=0.3, label='deterministic')
plt.scatter( 
    xs_MC[:N_MC_to_plot, -1, 0], 
    xs_MC[:N_MC_to_plot, -1, 2],
    c='b', alpha=0.3, label='stochastic')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$r_x(T)$', fontsize=18)
plt.ylabel(r'$r_z(T)$', fontsize=18)
plt.legend(fontsize=16)
fig.savefig('figures/generated_montecarlo.png')
plt.close()

print("Z-standard deviation, deterministic method:",
    np.std(xs_det_MC[:, -1, 2]))
print("Z-standard deviation, stochastic method:", 
    np.std(xs_MC[:, -1, 2]))