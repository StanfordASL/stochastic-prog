import argparse
import enum
from tqdm import tqdm
import numpy as np
import jax
import ipyopt
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc, rcParams

from benchmark_optimization_problems import *
from benchmark_solver import *
from benchmark_compute_true_solutions import *
from benchmark_utils import compute_variance_metric, compute_error_metric

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)

def list_of_strings(arg):
    return arg.split(',')
parser = argparse.ArgumentParser(description='Benchmark script.')
parser.add_argument('--compute_true_solutions',
    action='store_true',
    help="To compute true optimal solutions of stochastic programs")
parser.add_argument('--dont_compute_true_solutions',
    dest='compute_true_solutions', action='store_false')
parser.set_defaults(compute_true_solutions=True)
parser.add_argument('--compute_saa_solutions',
    action='store_true',
    help="To solve the sample average approximations")
parser.add_argument('--dont_compute_saa_solutions',
    dest='compute_saa_solutions', action='store_false')
parser.set_defaults(compute_saa_solutions=True)
parser.add_argument(
    "--programs_to_solve",
    default=[
        "program_6",
        "program_27",
        "program_28",
        "program_42",
        "program_47",
        "program_48",
        "program_49",
        "program_50",
        "program_51",
        "program_52",
        "program_61",
        "program_79",
    ],
    type=list_of_strings,
    help='List of programs to solve (example: program_6,program_27)',
)
parser.add_argument(
    "--num_SAA_repeats",
    default=100,
    type=int,
    help='Number of repeats for the sample average approximation (SAA) approach',
)
args = parser.parse_args()
compute_true_solutions = args.compute_true_solutions
compute_saa_solutions = args.compute_saa_solutions
programs_to_solve = args.programs_to_solve
num_repeats = args.num_SAA_repeats

np.random.seed(0)

B_use_omega_bias = True
sample_size_vec = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
delta_relaxation_vec = [1e-2, 1e-1, 2e-1, 3e-1]
problems = [
    Program6(),
    Program27(),
    Program28(),
    Program42(),
    Program47(),
    Program48(),
    Program49(),
    Program50(),
    Program51(),
    Program52(),
    Program61(),
    Program79(),
    ]

print("--------------------------------------------")
print("[benchmark_main.py] Running with parameters:")
print("compute_true_solutions =", compute_true_solutions)
print("compute_saa_solutions  =", compute_saa_solutions)
print("programs_to_solve      =", programs_to_solve)
print("num_SAA_repeats        =", num_repeats)
print("--------------------------------------------")


if compute_true_solutions:
    print("Computing true solutions.")
    compute_true_optimal_solutions(problems)


if compute_saa_solutions:
    print("Computing solutions")

    print("-----------------------------------------")
    for i, nominal_problem in enumerate(problems):
        if nominal_problem.name in programs_to_solve:
            msg = "Solving " + nominal_problem.name

            # per-problem-arrays are organized according to
            # (sample_size_id, (delta_relaxation, ) num_repeats, ...)
            successes_saa_p = np.zeros((
                len(sample_size_vec), num_repeats), dtype=bool)
            successes_saa_relaxed_p = np.zeros((
                len(sample_size_vec), len(delta_relaxation_vec), num_repeats), dtype=bool)
            solutions_saa_p = np.zeros((
                len(sample_size_vec), num_repeats, 
                nominal_problem.num_variables))
            solutions_saa_relaxed_p = np.zeros((
                len(sample_size_vec), len(delta_relaxation_vec), num_repeats, 
                nominal_problem.num_variables))
            objectives_saa_p = np.zeros((
                len(sample_size_vec), num_repeats))
            objectives_saa_relaxed_p = np.zeros((
                len(sample_size_vec), len(delta_relaxation_vec), num_repeats))

            for j, sample_size_M in enumerate(tqdm(sample_size_vec, desc=msg)):
                for l in range(num_repeats):

                    program = SampledProgram(
                        nominal_problem, 
                        sample_size=sample_size_M,
                        add_bias=B_use_omega_bias)
                    solver_saa = Solver(
                        program=program,
                        delta_equality_relaxation=0)
                    out_saa = solver_saa.solve()
                    x_saa = out_saa[0]
                    obj_saa = out_saa[1]
                    status_saa = out_saa[2]
                    if status_saa == SolverReturnStatus.Solve_Succeeded:
                        successes_saa_p[j, l] = True
                        solutions_saa_p[j, l] = x_saa
                        objectives_saa_p[j, l] = obj_saa

                    for k, relaxation_delta_M in enumerate(delta_relaxation_vec):
                        solver_saa_relaxed = Solver(
                            program=program,
                            delta_equality_relaxation=relaxation_delta_M)
                        out_saa_relaxed = solver_saa_relaxed.solve()
                        x_saa_relaxed = out_saa_relaxed[0]
                        obj_saa_relaxed = out_saa_relaxed[1]
                        status_saa_relaxed = out_saa_relaxed[2]
                        if status_saa_relaxed == SolverReturnStatus.Solve_Succeeded:
                            successes_saa_relaxed_p[j, k, l] = True
                            solutions_saa_relaxed_p[j, k, l] = x_saa_relaxed
                            objectives_saa_relaxed_p[j, k, l] = obj_saa_relaxed
                        
                # # clear memory
                # jax.clear_caches()

            # save results
            fn = 'results/benchmark_results_' + nominal_problem.name + '.npy'
            with open(fn, 'wb') as f:
                np.save(f, successes_saa_p)
                np.save(f, successes_saa_relaxed_p)
                np.save(f, solutions_saa_p)
                np.save(f, solutions_saa_relaxed_p)
                np.save(f, objectives_saa_p)
                np.save(f, objectives_saa_relaxed_p)

    solver_names = ["SAA", "SAA_relaxed"]
    problem_names = [p.name for p in problems]
    with open('results/benchmark_results.npy', 'wb') as f:
        np.save(f, sample_size_vec)
        np.save(f, delta_relaxation_vec)
        np.save(f, num_repeats)
        np.save(f, solver_names)
        np.save(f, problem_names)


# load results
with open('results/benchmark_results.npy', 'rb') as f:
    sample_size_vec = np.load(f, allow_pickle=True)
    delta_relaxation_vec = np.load(f, allow_pickle=True)
    num_repeats = np.load(f, allow_pickle=True)
    solver_names = np.load(f, allow_pickle=True)
    problem_names = np.load(f, allow_pickle=True)
# arrays are organized according to
# (problem_id, sample_size_id, (delta_relaxation, ) num_repeats, ...)
successes_saa = []          # [np.zeros((
                            #     len(sample_size_vec), num_repeats),
                            #     dtype=bool) for p in problems]
successes_saa_relaxed = []  # [np.zeros((
                            #     len(sample_size_vec), len(delta_relaxation_vec), num_repeats),
                            #     dtype=bool) for p in problems]
solutions_saa = []          # [np.zeros((
                            #     len(sample_size_vec), num_repeats,
                            #     p.num_variables)) for p in problems]
solutions_saa_relaxed = []  # [np.zeros((
                            #     len(sample_size_vec), len(delta_relaxation_vec), num_repeats, 
                            #     p.num_variables)) for p in problems]
objectives_saa = []         # [np.zeros((
                            #     len(sample_size_vec), num_repeats)
                            #     ) for p in problems]
objectives_saa_relaxed = [] # [np.zeros((
                            #     len(sample_size_vec), len(delta_relaxation_vec), num_repeats)
                            #     ) for p in problems]
for i, p_name in enumerate(problem_names):
    fn = 'results/benchmark_results_' + p_name + '.npy'
    with open(fn, 'rb') as f:
        successes_saa.append(
            np.load(f, allow_pickle=True))
        successes_saa_relaxed.append(
            np.load(f, allow_pickle=True))
        solutions_saa.append(
            np.load(f, allow_pickle=True))
        solutions_saa_relaxed.append(
            np.load(f, allow_pickle=True))
        objectives_saa.append(
            np.load(f, allow_pickle=True))
        objectives_saa_relaxed.append(
            np.load(f, allow_pickle=True))
# load true solutions
out = load_true_optimal_solutions()
optimal_solutions, optimal_objectives = out



# Plotting
cmap = plt.get_cmap('jet') # winter, plasma, rainbow
N = len(delta_relaxation_vec)
colors = [cmap(float(t)/N) for t in range(N+1)][::-1]
markers = ['o', 's', 'v', '^', 'p', 'X', 'd']

# Success Rate
# 1) data preprocessing
# median over all problems and repeats
# successes_saa - [len(sample_size_vec), num_repeats) for p in problems]
succ_saa_mean = np.mean(
    np.array(successes_saa), axis=(0, 2))
succ_saa_rlx_mean = np.mean(
    np.array(successes_saa_relaxed), axis=(0, 3))
# 2) plotting
plt.figure(figsize=[8, 4.5])
plt.plot(sample_size_vec, 100*succ_saa_mean,
    color='k', label=r'SAA ($\delta_N=0$)')
for k, relaxation_delta_M in enumerate(delta_relaxation_vec):
    plt.plot(sample_size_vec, 100*succ_saa_rlx_mean[:, k],
        color=colors[k], marker=markers[k], linestyle='dashed', 
        label=r'SAA, $\delta_N='+str(relaxation_delta_M)+'$')
plt.xlabel(r"Sample Size $N$", fontsize=28)
plt.ylabel(r"Success Rate (\%)", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(linestyle='--')
plt.tight_layout()
plt.subplots_adjust(left=0.105, right=0.99, bottom=0.17)


# Errors to Optimal Solution
# 1) data preprocessing
# mean over all problems and repeats
# solutions_saa - [(len(sample_size_vec), num_repeats, p.num_variables) for p in problems]
errors_saa = np.zeros((len(problems), len(sample_size_vec), num_repeats))
errors_saa_relaxed = np.zeros((len(problems), len(sample_size_vec), len(delta_relaxation_vec), num_repeats))
for i, (p_name, sol, sol_rlx) in enumerate(zip(
        problem_names, 
        solutions_saa, 
        solutions_saa_relaxed)):
    sol_true = optimal_solutions[()][p_name]
    for j, _ in enumerate(sample_size_vec):
        for r in range(num_repeats):
            err = 1e9
            if successes_saa[i][j, r] > 0:
                err = np.linalg.norm(sol[j, r] - sol_true, axis=-1) / (
                    np.linalg.norm(1 + sol_true, axis=-1))
            errors_saa[i, j, r] = err
            for k, _ in enumerate(delta_relaxation_vec):
                err = 1e9
                if successes_saa_relaxed[i][j, k, r]:
                    err = np.linalg.norm(sol_rlx[j, k, r] - sol_true, axis=-1) / (
                        np.linalg.norm(1 + sol_true, axis=-1))
                errors_saa_relaxed[i, j, k, r] = err
errors_saa_mean = np.median(errors_saa, axis=(0, 2))
errors_saa_rlx_mean = np.median(errors_saa_relaxed, axis=(0, 3))
# 2) plotting
plt.figure(figsize=[8, 4.5])
plt.plot(sample_size_vec, errors_saa_mean,
    color='k', label=r'SAA ($\delta=0$)')
for k, relaxation_delta_M in enumerate(delta_relaxation_vec):
    plt.plot(sample_size_vec, errors_saa_rlx_mean[:, k],
        color=colors[k], marker=markers[k], linestyle='dashed', 
        label=r'SAA, $\delta_N='+str(relaxation_delta_M)+'$')
plt.xlabel(r"Sample Size $N$", fontsize=28)
plt.ylabel(r"Solution Error", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20, labelspacing=0.2)
plt.grid(linestyle='--')
plt.tight_layout()
plt.subplots_adjust(left=0.105, right=0.99, bottom=0.17)




# Errors to Optimal Value
# objectives_saa - [(len(sample_size_vec), num_repeats)
#                       for p in problems]
# objectives_saa_relaxed - [len(sample_size_vec), len(delta_relaxation_vec), num_repeats)
#                       for p in problems]
# 1a) data preprocessing
# compute objective values (what is saved in objectives_saa is the sample
# approximated objective value)
errors_saa = np.zeros((len(problems), len(sample_size_vec)))
errors_saa_relaxed = np.zeros((len(problems), len(sample_size_vec), len(delta_relaxation_vec)))
for i, (problem, p_name, sol, sol_rlx) in enumerate(zip(
        problems,
        problem_names,
        solutions_saa,
        solutions_saa_relaxed)):
    assert problem.name == p_name
    program = StochasticTrueAverageValueProgram(problem)
    obj_true = optimal_objectives[()][p_name]
    for j, _ in enumerate(sample_size_vec):
        for r in range(num_repeats):
            objectives_saa[i][j, r] = program.objective(sol[j, r])
            for k, _ in enumerate(delta_relaxation_vec):
                objectives_saa_relaxed[i][j, k, r] = program.objective(
                    sol_rlx[j, k, r])
# 1b) compute errors
# median over all problems and repeats
errors_saa = np.zeros((
    len(problems), len(sample_size_vec), num_repeats))
errors_saa_relaxed = np.zeros((
    len(problems), len(sample_size_vec), len(delta_relaxation_vec), num_repeats))
for i, (p_name, obj, obj_rlx) in enumerate(zip(
        problem_names,
        objectives_saa,
        objectives_saa_relaxed)):
    obj_true = optimal_objectives[()][p_name]
    for j, _ in enumerate(sample_size_vec):
        for r in range(num_repeats):
            err = 1e9
            if successes_saa[i][j, r]:
                err = np.abs((obj[j, r] - obj_true) / (1 + np.abs(obj_true)))
            errors_saa[i, j, r] = err
            for k, _ in enumerate(delta_relaxation_vec):
                for r in range(num_repeats):
                    err = 1e9
                    if successes_saa_relaxed[i][j, k, r]:
                        err = np.abs((obj_rlx[j, k, r] - obj_true) / (1 + np.abs(obj_true)))
                    errors_saa_relaxed[i, j, k, r] = err
errors_saa_mean = np.median(errors_saa, axis=(0, 2))
errors_saa_rlx_mean = np.median(errors_saa_relaxed, axis=(0, 3))
# 2) plotting
plt.figure(figsize=[8, 4.5])
plt.plot(sample_size_vec, errors_saa_mean,
    color='k', label=r'SAA ($\delta=0$)')
for k, relaxation_delta_M in enumerate(delta_relaxation_vec):
    plt.plot(sample_size_vec, errors_saa_rlx_mean[:, k],
        color=colors[k], marker=markers[k], linestyle='dashed',
        label=r'SAA, $\delta_N='+str(relaxation_delta_M)+'$')
plt.xlabel(r"Sample Size $N$", fontsize=28)
plt.ylabel(r"Optimal Value Error", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20, labelspacing=0.2)
plt.grid(linestyle='--')
plt.tight_layout()
plt.subplots_adjust(left=0.11, right=0.99, bottom=0.17)



plt.show()