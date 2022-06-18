import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import gridspec

np.random.seed(1)

S = 21  # number of controls
N = 100 # number of samples
C = 1
delta_N = C/np.sqrt(N) # relaxation constant
N_scp_iters = 10 # number of SCP iterations

n_x = 7 # (rx,ry,rz,vx,vy,vz,m)
n_u = 4 # (ux,uy,uz,sigma) (sigma(t)=||u(t)|| under some conditions, see [Acikmese & Ploen, 2007])
class Rocket:
    def __init__(self, N, S, delta_N, B_uncertainty=True):
        self.N = N # number of samples
        self.S = S # number of control switches
        # Planning time horizon
        self.T  = 60. # sec
        self.dt = self.T/(self.S-1)
        # Initial / final states
        self.n_x = 7
        self.n_u = 4
        # Values inspired from the Mars rover MSL Curiosity landing 
        # (up to sky-crane separation)
        # https://arc.aiaa.org/doi/pdf/10.2514/1.A32866 (Steltzner et al., 2014)
        # https://spaceflight101.com/msl/msl-landing-special/
        self.x0 = np.array([300, 0, 1500, 
                            5,  0, -75.,
                            1800])
        self.xg = np.array([0, 0, 100, 
                            0,  0, -10.])

        # Parameters approx. from https://hal-ensta-paris.archives-ouvertes.fr//hal-03641631/document
        # Clara Leparoux, Bruno Hérissé, Frédéric Jean, Optimal planetary landing with pointing and 
        # glide-slope constraints. 2022.
        self.Thrust_max = 16000 # N
        self.q          = 8 # kg/s
        self.umin       = 0.3
        self.umax       = 0.8
        self.g          = 3.71  # m/s^2 (Mars)
        self.gamma      = 5.0  * (np.pi/180.0) # glide slope
        self.theta      = 45.0 * (np.pi/180.0) # max thrust angle
        # Uncertainty
        self.Cdrag = 1.0
        self.beta0 = 1.0
        self.beta1 = 1.0
        self.beta2 = 1e2
        self.beta3 = 1e-1
        # Samples from Brownian motion
        self.DWs   = np.zeros((self.N,self.S,self.n_x))
        if B_uncertainty==True:
            for i in range(self.N):
                for t in range(self.S):
                    self.DWs[i,t,:] = np.sqrt(self.dt)*np.random.randn(n_x)
        # Constraints relaxation constant
        self.delta_N = delta_N
    def b(self, x, u):
        Tmax, Cd, q = self.Thrust_max, self.Cdrag, self.q
        v, mass     = x[3:6], x[-1]
        unorm       = u[-1]
        bvec        = np.zeros_like(x)
        bvec[:3]    = v
        bvec[3:6]   = (1.0/mass)*Tmax*u[:3] - np.array([0.,0.,self.g])
        bvec[3:6]   += (1.0/mass)*(-Cd*np.abs(v)*v)
        bvec[-1]    = -q * unorm
        return bvec
    def b_dx(self, x, u):
        Tmax, Cd, q   = self.Thrust_max, self.Cdrag, self.q
        v, mass       = x[3:6], x[-1]
        bmat          = np.zeros((self.n_x,self.n_x))
        bmat[0:3,3:6] = np.eye(3)
        bmat[3:6,3:6] = (1.0/mass)      *(-Cd*np.diag(2.0*np.sign(v)*v))
        bmat[3:6, -1] = (-1.0/(mass**2))*(Tmax*u[:3]-Cd*np.abs(v)*v)
        return bmat
    def b_du(self, x, u):
        Tmax, q      = self.Thrust_max, self.q
        mass         = x[-1]
        bmat         = np.zeros((self.n_x,self.n_u))
        bmat[3:6,:3] = (Tmax/mass)*np.eye(3)
        bmat[-1,-1]  = -q
        return bmat
    def sigma(self, x, u):
        b0, b1, b2, b3 = self.beta0, self.beta1, self.beta2, self.beta3
        v, mass, unorm = x[3:6], x[-1], u[-1]
        sdiag          = (1.0/mass) * (b0 + b1*v + b2*(u[:3]**2))
        smat           = np.zeros((self.n_x,self.n_x))
        smat[3:6,3:6]  = np.diag(sdiag)
        smat[ 6 , 6 ]  = b3*unorm
        return smat
    def sigma_dx(self, x, u):
        b0, b1, b2, b3 = self.beta0, self.beta1, self.beta2, self.beta3
        v, mass, unorm = x[3:6], x[-1], u[-1]
        sdiag_dx       = np.zeros((self.n_x,self.n_x,self.n_x))
        for i in range(3):
            sdiag_dx[3+i,3+i,3+i] = b1 / mass
            sdiag_dx[3+i,3+i,-1]  = (-1.0/(mass**2))*(b0 + b1*v[i] + b2*(u[i]**2))
        return sdiag_dx
    def sigma_du(self, x, u):
        b2, b3   = self.beta2, self.beta3
        mass     = x[-1]
        sdiag_du = np.zeros((self.n_x,self.n_x,self.n_u))
        for i in range(3):
            sdiag_du[3+i,3+i,i] = (1.0/mass) * b2*2.0*u[i]
        sdiag_du[6,6,-1] = b3
        return sdiag_du
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
            idx_x0 = (S-1)*n_u + (S-1)*N*n_x + i*n_x
            idx_x0n = idx_x0 + 6
            Aineq[:,idx_x0:idx_x0n] = np.eye(6) / N
        lineq = xg - self.delta_N
        uineq = xg + self.delta_N
        lineq[3:] = xg[3:] - 1e-2*self.delta_N
        uineq[3:] = xg[3:] + 1e-2*self.delta_N
        return Aineq, lineq, uineq
    def dynamics_constraints(self, Zp):
        S, N, dt = self.S, self.N, self.dt
        n_x, n_u = self.n_x, self.n_u
        nb_vars  = (S-1)*n_u + S*N*n_x
        Aeq = np.zeros((N*(S-1)*n_x, nb_vars))
        leq = np.zeros(N*(S-1)*n_x)
        for i in range(N):
            for t in range(S-1):
                idx_con = (i*(S-1)+t)*n_x
                idx_ut  = t*n_u
                idx_utn = idx_ut + n_u
                idx_xt  = (S-1)*n_u + t*N*n_x + i*n_x
                idx_xtn = idx_xt + N*n_x
                xp_t    = Zp[idx_xt:(idx_xt+n_x)]
                up_t    = Zp[idx_ut:idx_utn]
                # Euler–Maruyama
                # x_{t+1} = x_t + Dt*(bp + bp_dx@(x_t-xp_t) + bp_du@(u_t-up_t))
                #               +    (sp + sp_dx@(x_t-xp_t) + sp_du@(u_t-up_t)) @ sqrt(dt)*W_t
                # deterministic terms
                bp       = self.b(xp_t, up_t)
                bp_dx    = self.b_dx(xp_t, up_t)
                bp_du    = self.b_du(xp_t, up_t)
                Aeq[idx_con:(idx_con+n_x), idx_xt:(idx_xt+n_x)]   =  np.eye(n_x)+dt*bp_dx # x_{t}
                Aeq[idx_con:(idx_con+n_x), idx_ut:idx_utn]        =  dt*bp_du             # u_{t}
                Aeq[idx_con:(idx_con+n_x), idx_xtn:(idx_xtn+n_x)] = -np.eye(n_x)          # x_{t+1}
                leq[idx_con:(idx_con+n_x)] = - dt * (bp - bp_dx@xp_t - bp_du@up_t)
                # stochastic terms
                sigp  = self.sigma(xp_t,up_t)
                sp_dx = self.sigma_dx(xp_t,up_t)
                sp_du = self.sigma_du(xp_t,up_t)
                DWt   = self.DWs[i,t,:]
                sp_DWt    = sigp @ DWt
                sp_DWt_dx = np.einsum('ijx,j->ix', sp_dx, DWt)
                sp_DWt_du = np.einsum('iju,j->iu', sp_du, DWt)
                Aeq[idx_con:(idx_con+n_x), idx_xt:(idx_xt+n_x)] += sp_DWt_dx # x_{t}
                Aeq[idx_con:(idx_con+n_x), idx_ut:idx_utn]      += sp_DWt_du # u_{t}
                leq[idx_con:(idx_con+n_x)] += - (sp_DWt - sp_DWt_dx@xp_t - sp_DWt_du@up_t)
                # # Milstein term (does not make a big difference in this case)
                # #   0.5*sigma(x,u)*[(d/dx) sigma(x,u)] ( (W_{t+dt}-W_t)^2 - dt)
                # # - we use the fact that sigma is diagonal
                # # - we neglect second derivatives when linearizing
                # DWtsquared_dt = DWt**2 - dt
                # mils = 0.5*np.diag(sigp)*np.diag(np.einsum('ijx,j->ix',sp_dx,DWtsquared_dt))
                # mils_dx = 0.5*np.einsum('ijx,j->ix',sp_dx,np.diag(np.einsum('ijx,j->ix',sp_dx,DWtsquared_dt)))
                # mils_du = 0.5*np.einsum('iju,j->iu',sp_du,np.diag(np.einsum('ijx,j->ix',sp_dx,DWtsquared_dt)))
                # Aeq[idx_con:(idx_con+n_x), idx_xt:(idx_xt+n_x)] += mils_dx # x_{t}
                # Aeq[idx_con:(idx_con+n_x), idx_ut:idx_utn]      += mils_du # u_{t}
                # leq[idx_con:(idx_con+n_x)] += - (mils - mils_dx@xp_t - mils_du@up_t)
        ueq = leq.copy()
        return Aeq, leq, ueq

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
        for t in range(S-1):
            # control
            idx_ut = t*n_u
            obj = obj + dt*Z[idx_ut+3]
        for i in range(N):
            # Final deviation to landing
            #  if model.xg=E[r_T], then 
            #  sum_squares(r_T-model.xg) gives 
            #  the trace of the covariance matrix
            idx = (S-1)*n_u + (S-1)*N*n_x + i*n_x
            r_T = Z[idx:(idx+6)]
            obj = obj + 1 * (1.0/N)*cp.sum_squares(r_T[:3]-model.xg[:3])
        obj = cp.Minimize(obj)
        # Constraints
        Aeq_x0,  leq_x0,  ueq_x0  = model.initial_constraints()
        Aeq_xf,  leq_xf,  ueq_xf  = model.final_constraints()
        Aeq_dyn, leq_dyn, ueq_dyn = model.dynamics_constraints(Zp)
        con = []
        con.append(Aeq_x0@Z==leq_x0)
        # con.append(Aeq_xf@Z==leq_xf)
        con.append(leq_xf   <= Aeq_xf@Z)
        con.append(Aeq_xf@Z <= ueq_xf)
        con.append(Aeq_dyn@Z==leq_dyn)
        # control
        for t in range(S-1):
            idx_ut = t*n_u
            u_t           = Z[idx_ut:(idx_ut+3)]
            uz_t, sigma_t = Z[idx_ut+2], Z[idx_ut+3]
            con.append(model.umin<=sigma_t)
            con.append(sigma_t<=model.umax)
            con.append(uz_t>=sigma_t*np.cos(model.theta))
            # cp.SOC(t, x) creates the SOC constraint ||x||_2 <= t.
            con.append(cp.SOC(sigma_t, u_t))
        # slew rate control constraint (see [1])
        ### Removed since constraints are inactive 
        ### (typical Tdot_max value is quite large)
        # Tdot_max = 0.5
        # for t in range(S-2):
        #     sigma_t, sigma_tn = Z[t*n_u+3], Z[(t+1)*n_u+3]
        #     con.append(-dt*Tdot_max <= sigma_tn-sigma_t)
        #     con.append( sigma_tn-sigma_t <= dt*Tdot_max)
        # altitude
        tan_gamma = np.tan(model.gamma)
        S_max = S
        for t in range(S_max):
            xy_t_avg, z_t_avg = 0, 0
            for i in range(N):
                idx_xt = (S-1)*n_u + t*N*n_x + i*n_x
                xy_t   = Z[idx_xt:idx_xt+2]
                z_t    = Z[idx_xt+2]
                xy_t_avg += xy_t
                z_t_avg  += z_t
            xy_t_avg = xy_t_avg / N
            z_t_avg  = z_t_avg  / N
            con.append( tan_gamma*xy_t_avg[0] - z_t_avg <= delta_N)
            con.append( tan_gamma*xy_t_avg[1] - z_t_avg <= delta_N)
            con.append(-tan_gamma*xy_t_avg[0] - z_t_avg <= delta_N)
            con.append(-tan_gamma*xy_t_avg[1] - z_t_avg <= delta_N)
        # trust region constraint for scp
        Delta_TrustRegion = 0.7**scp_iter
        if scp_iter>0:
            for t in range(S-1):
                idx_ut = t*n_u
                u_t, up_t = Z[idx_ut:idx_ut+3], Zp[idx_ut:idx_ut+3]
                con.append(u_t-up_t <= Delta_TrustRegion)
                con.append(-Delta_TrustRegion <= u_t-up_t)
        self.Z    = Z
        self.prob = cp.Problem(obj, con)
    def solve(self):
        self.prob.solve(solver=cp.ECOS,verbose=False)
        self.Z = self.Z.value
    def initial_guess(self):
        n_x, n_u, S, N, dt = self.n_x, self.n_u, self.S, self.N, self.dt
        # initial guess (linearization point) (straight-line)
        Zp = np.zeros(self.nb_vars)
        for t in range(S):
            idx_ut = t*n_u
            Zp[idx_ut:idx_ut+3] = (model.umin + model.umax) / 2.0
            for i in range(N):
                idx_xt = (S-1)*n_u + t*N*n_x + i*n_x
                alpha1 = ( (S-1) - t ) / (S-1)
                alpha2 =       t       / (S-1)
                Zp[idx_xt:idx_xt+6] = model.x0[:6] * alpha1 + model.xg * alpha2 + 1e-6
                Zp[idx_xt+6]        = model.x0[-1] # mass
        return Zp
    def extract_solution(self):
        Z = convex_problem.Z
        xs,us = self.convert_Z_to_xs_us(Z)
        return (Z, xs, us)
    def convert_Z_to_xs_us(self, Z):
        us = Z[:(S-1)*n_u]
        xs = Z[(S-1)*n_u:]
        us = np.reshape(us, (n_u,S-1), 'F')
        xs = np.reshape(xs, (n_x,N,S), 'F')
        us = us.T                   # (S-1,n_u)
        xs = np.moveaxis(xs, 0, -1) # (N,S,n_x)
        return (xs, us)

# -----------------------------------------
model = Rocket(N, S, delta_N)
nb_vars = (S-1)*n_u + S*N*n_x
dt = model.T/(S-1)
# -----------------------------------------
# --------- with uncertainty --------------
print(">>> solving stochastic program")
print("if too slow, consider reducing the nb. of samples N")
convex_problem = convexified_problem(model)
Zp = convex_problem.initial_guess()
for scp_iter in range(N_scp_iters):
    print("SCP iter. "+str(scp_iter+1)+"/"+str(N_scp_iters))
    convex_problem.define(Zp)
    convex_problem.solve()
    Z,xs,us = convex_problem.extract_solution()
    Zp = Z.copy()
print("Stochastic: final mass =", np.mean(xs[:,-1,-1], axis=0), "[kg]")
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
    Z,xs_det,us_det = convex_problem.extract_solution()
    Zp = Z.copy()
print("Deterministic: final mass =", np.mean(xs_det[:,-1,-1], axis=0), "[kg]")
# -----------------------------------------


########### PLOT (deterministic) ###########################
# plot
fig = plt.figure(figsize=[10,3])
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1]) 

plt.subplot(gs[0])
plt.scatter(model.x0[0],model.x0[2], color='k')
plt.scatter(model.xg[0],model.xg[2], color='k')
for i in range(N):
    plt.plot(xs_det[i,:,0],xs_det[i,:,2],
             c='g', alpha=0.3)
# mean trajectory and controls
x_traj_mean = np.zeros((S,n_x))
for t in range(S-1):
    xt, ut = np.mean(xs_det[:,t,:],axis=0), us_det[t,:]
    if t == 0:
        plt.plot([xt[0],xt[0]+400*ut[0]],
                 [xt[2],xt[2]+400*ut[2]],
                 c='b', alpha=0.3,
                 label=r'$u(t)$')
    else:
        plt.plot([xt[0],xt[0]+400*ut[0]],
                 [xt[2],xt[2]+400*ut[2]],
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
         np.linalg.norm(us_det[:,:3],axis=1,ord=2),
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
fig.savefig('figures/generated_deterministic.png')
plt.close()
# plt.show()

########### PLOT (stochastic) ########################

# plot
fig = plt.figure(figsize=[10,3])
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1]) 

plt.subplot(gs[0])
plt.scatter(model.x0[0],model.x0[2], color='k')
plt.scatter(model.xg[0],model.xg[2], color='k')
for i in range(N):
    plt.plot(xs[i,:,0],xs[i,:,2],
             c='g', alpha=0.3)
# mean trajectory and controls
x_traj_mean = np.zeros((S,n_x))
for t in range(S-1):
    xt, ut = np.mean(xs[:,t,:],axis=0), us[t,:]
    if t == 0:
        plt.plot([xt[0],xt[0]+400*ut[0]],
                 [xt[2],xt[2]+400*ut[2]],
                 c='b', alpha=0.3,
                 label=r'$u(t)$')
    else:
        plt.plot([xt[0],xt[0]+400*ut[0]],
                 [xt[2],xt[2]+400*ut[2]],
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
         np.linalg.norm(us[:,:3],axis=1,ord=2),
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
fig.savefig('figures/generated_stochastic.png')
plt.close()

########### MONTE-CARLO ####################################
N_MC      = 1000
H_avg     = np.zeros(n_x)
H_det_avg = np.zeros(n_x)
xs_MC     = np.zeros((N_MC,S,n_x))
xs_det_MC = np.zeros((N_MC,S,n_x))
for i in range(N_MC):
    xs_MC[i,0,:]     = model.x0.copy()
    xs_det_MC[i,0,:] = model.x0.copy()
    # simulate dynamics
    for t in range(S-1):
        xt, ut             = xs_MC[i,t,:], us[t,:]
        xt_det, ut_det     = xs_det_MC[i,t,:], us_det[t,:]
        xs_MC[i,t+1,:]     = xt     + dt*model.b(xt,ut)         + model.sigma(xt,ut)@(np.sqrt(dt)*np.random.randn(n_x))
        xs_det_MC[i,t+1,:] = xt_det + dt*model.b(xt_det,ut_det) + model.sigma(xt_det,ut_det)@(np.sqrt(dt)*np.random.randn(n_x))
    H_avg     += xs_MC[i,-1,:]
    H_det_avg += xs_det_MC[i,-1,:]
H_avg     = H_avg     / N_MC
H_det_avg = H_det_avg / N_MC

fig = plt.figure(figsize=[6,6])
plt.grid(linestyle='--')
plt.scatter(xs_det_MC[:,-1,0],xs_det_MC[:,-1,2],
         c='tab:orange', alpha=0.3, label='deterministic')
plt.scatter(xs_MC[:,-1,0],xs_MC[:,-1,2],
         c='b', alpha=0.3, label='stochastic')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$r_x(T)$', fontsize=18)
plt.ylabel(r'$r_z(T)$', fontsize=18)
plt.legend(fontsize=16)
fig.savefig('figures/generated_montecarlo.png')
plt.close()
# plt.show()