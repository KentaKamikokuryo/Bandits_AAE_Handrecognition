import numpy as np
import os, itertools
from typing import Tuple, List, Dict, Any
from Classes_RL.Utilities import Utilities
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from Classes_RL.Utilities import Utilities
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Boltzmann_solver import BoltzmannSlidingWindowUCB, BoltzmannUCBNonStationary, BoltzmannNonStationary
from Classes_RL.Bayesian_sliding_solver import BayesianSlidingWindowUMKV, BayesianSlidingWindowUMUV
from Classes_RL.Kalman_filter_solver import KalmanUCBSoftmax
from Classes_RL.SolverFactory import Models
from Classes_RL.BanditsFactory import BanditsFactory
import matplotlib.pyplot as plt
from Classes_RL.Plot_RL import PlotSolver, PlotBehaviorNonStaticStrategy, AnimateSolver
from Classes_RL.Oracle import Oracle
import seaborn as sns

sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=3)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# plt.ioff()

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
n_actions = 3
n_iteration = 100
n_episode = 1

train_parameter = {"n_action": n_actions, "n_episode": n_episode}
parameters = {"name": "RndWalk", "range": 10, "std": 1}
# parameters = {"name": "sin", "amplitude_base": [1, 2], "frequency_range": [1, 3], "std_amplitude": 1}

# parameters = {"name": "sin", "frequency": 2, "amplitude": 1}

fac = BanditsFactory(parameters=parameters, n_iteration=n_iteration, n_action=n_actions)
bandits = fac.create()
plot_strategy = PlotBehaviorNonStaticStrategy()
# fig = plot_strategy.plot(bandits)
# plt.legend(loc=2, frameon=True, fancybox=False, ncol=3, framealpha=0.5, edgecolor="black")
# plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.2)
# plt.show()
# PlotSolver.save_plot(fig=fig, path=path_test, figure_name="behavior_" + parameters["name"])
#####
arguments = {"bandits": bandits,
             "n_iteration": bandits[0].get_n_iteration(),
             "n_episode": train_parameter["n_episode"]}

oracle_manager = Oracle(**arguments)
oracle = oracle_manager.get_oracle_data()
# AnimateSolver.animate_oracle(oracle_data=oracle, path=path_test, gif_name="oracle" + parameters["name"])

arguments_oracle = {"bandits": bandits,
                    "n_iteration": bandits[0].get_n_iteration(),
                    "n_episode": train_parameter["n_episode"],
                    "oracle_data": oracle}

factories = {Models.Boltzmann_UCB_sliding: BoltzmannSlidingWindowUCB(**arguments_oracle),
             Models.Boltzmann_non_stationary: BoltzmannNonStationary(**arguments_oracle),
             Models.Kalman_UCB_softmax: KalmanUCBSoftmax(**arguments_oracle),
             Models.Bayesian_um_kv_softmax_sliding_non_stationary: BayesianSlidingWindowUMKV(**arguments_oracle),
             Models.Bayesian_um_uv_softmax_sliding_non_stationary: BayesianSlidingWindowUMUV(**arguments_oracle)}

solver_names = [Models.Bayesian_um_uv_softmax_sliding_non_stationary]
# solver_name = solver_name[1]
# solver_names = [solver_names[0]]

for solver_name in solver_names:
    solver = factories[solver_name]

    solver_name_hyper = Hyperparamaters_bandits.get_best_hyperparameters(solver_name=solver_name)
    solver.set_hyperparameters(hyperparameters=solver_name_hyper)
    name = Hyperparamaters_bandits.get_hyperparameter_name_latex(hyperparameter=solver_name_hyper)

    results = {}
    results_episode, results_mean = solver.run()
    results[solver_name]=results_mean

    plob = True
    if "boltzmann" in solver_name:
        plob = False

    fig = PlotSolver.plot_Q_lower_upper_bound(result=results_mean, oracle_data=oracle, figure_title=str(name))
    # AnimateSolver.animate_Q_lower_upper_bound(result=results_mean, oracle_data=oracle, path=path_test, gif_name=solver_name, plob=plob)

plt.show()

