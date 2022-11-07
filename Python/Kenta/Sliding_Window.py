import numpy as np
import os, itertools
from typing import Tuple, List, Dict, Any
from Classes_RL.Utilities import Utilities

np.set_printoptions(precision=4, suppress=True)
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from Classes_RL.Utilities import Utilities
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Bandit import BehaviorRndWalkStrategy
from Classes_RL.PlotFactory import PlotResultsFactory
from Classes_RL.Bayesian_solver import Bayesian, BayesianNonStationaryUMUV, BayesianNonStationaryUMKV
from Classes_RL.Bayesian_sliding_solver import BayesianSlidingWindowUMKV, BayesianSlidingWindowUMUV
from Classes_RL.SolverFactory import Models
from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.Plot_RL import PlotSolver

np.set_printoptions(precision=4, suppress=True)

# simple
cwd = os.getcwd()
# Get path one level above the root of the project
path_parent_project = os.path.abspath(os.path.join(cwd, os.pardir))
# Path to the data
path_test = path_parent_project + "\\Test\\"
if not (os.path.exists(path=path_test)):
    os.makedirs(path_test)

bandits = []
n_actions = 4
n_iteration = 100
n_episode = 1

train_parameter = {"n_action": n_actions, "n_episode": n_episode}
parameters = {"name": "sin", "frequency": 2, "amplitude": 1}
parameters = {"name": "RndWalk", "range": 10, "std": 1}

fac = BanditsFactory(parameters=parameters, n_iteration=n_iteration, n_action=n_actions)
bandits = fac.create()

#####
arguments = {"bandits": bandits,
             "n_iteration": bandits[0].get_n_iteration(),
             "n_episode": train_parameter["n_episode"]}

factories = {Models.Bayesian_um_kv_softmax_sliding_non_stationary: BayesianSlidingWindowUMKV(**arguments),
             Models.Bayesian_um_uv_softmax_sliding_non_stationary: BayesianSlidingWindowUMUV(**arguments),
             Models.Bayesian_um_kv_softmax_non_stationary: BayesianNonStationaryUMKV(**arguments),
             Models.Bayesian_um_uv_softmax_non_stationary: BayesianNonStationaryUMUV(**arguments)}

solver_names = [Models.Bayesian_um_kv_softmax_sliding_non_stationary, Models.Bayesian_um_uv_softmax_sliding_non_stationary,
                Models.Bayesian_um_uv_softmax_non_stationary, Models.Bayesian_um_kv_softmax_non_stationary]


def set_bandits_data(bandits: List[Bandit]) -> dict:
    # Get back data from bandits
    # Get max bandit
    bandit_max_list = []
    n_iteration = len(bandits[0].mu_iterations)
    for i in range(n_iteration):
        bandit_max = max([bandits[j].mu_iterations[i] for j in range(len(bandits))])
        bandit_max_list.append(bandit_max)

    bandits_data = {}
    bandits_data["Q_target"] = np.array(bandit_max_list)

    bandits_data["mu_iterations"] = [bandit.mu_iterations for bandit in bandits]
    bandits_data["std_iterations"] = [bandit.std_iterations for bandit in bandits]
    bandits_data["var_iterations"] = [bandit.var_iterations for bandit in bandits]
    bandits_data["tau_iterations"] = [bandit.tau_iterations for bandit in bandits]

    return bandits_data

bandits_data = set_bandits_data(bandits=bandits)

solver_names = [solver_names[0]]
# solver_name = solver_name[1]
for solver_name in solver_names:
    solver = factories[solver_name]

    solver_name_hyper = Hyperparamaters_bandits.get_best_hyperparameters(solver_name=solver_name)
    solver.set_hyperparameters(hyperparameters=solver_name_hyper)

    results = {}
    results_episode, results_mean, oracle = solver.run()
    results[solver_name]=results_mean

    plotFactory = PlotResultsFactory()
    mean_plot_strategy, episode_plot_strategy = plotFactory.create(solver_name=solver_name)
    figure_name = solver_name

    PlotSolver.plot_upper_lower_bounds(result=results_mean, model_name=solver_name, oracle=oracle)
    # mean_plot_strategy.plot(results=results, path_folder_figure=path_test, figure_name=figure_name, bandits_data=oracle)

    result = results[solver_name]
