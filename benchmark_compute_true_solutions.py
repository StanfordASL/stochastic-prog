import numpy as np

from benchmark_optimization_problems import *
from benchmark_solver import *


def compute_true_optimal_solutions(problems, verbose=True):
    problem_names = [p.name for p in problems]
    optimal_solutions = dict()
    optimal_objectives = dict()
    # solve true problems and save values
    for _, problem in enumerate(problems):
        if verbose:
            print("Solving " + problem.name)
        program = StochasticTrueAverageValueProgram(problem)
        # solve stochastic optimization problem
        # (without SAA approximation)
        solver = Solver(program=program)
        x_sol, obj_sol, status_sol = solver.solve()
        if status_sol != SolverReturnStatus.Solve_Succeeded:
            print(status_sol)
        # save optimal solution
        optimal_solutions[problem.name] = x_sol
        optimal_objectives[problem.name] = obj_sol
    # save to file
    with open('results/benchmark_optimal_solutions.npy', 'wb') as f:
        np.save(f, problem_names)
        np.save(f, optimal_solutions)
        np.save(f, optimal_objectives)


def load_true_optimal_solutions():
    with open('results/benchmark_optimal_solutions.npy', 'rb') as f:
        _ = np.load(f, allow_pickle=True)
        optimal_solutions = np.load(f, allow_pickle=True)
        optimal_objectives = np.load(f, allow_pickle=True)
    return optimal_solutions, optimal_objectives


if __name__=="__main__":
    print("[benchmark_compute_true_solutions.py]")
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
    compute_true_optimal_solutions(problems)
    load_true_optimal_solutions()
    print("Successfully computed solutions.")