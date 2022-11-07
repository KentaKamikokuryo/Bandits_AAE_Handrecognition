import numpy as np
import os, itertools
from typing import Tuple, List, Dict, Any
from Classes_RL.Utilities import Utilities
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from Classes_RL.Utilities import Utilities
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Bandit import BehaviorRndWalkStrategy
from Classes_RL.PlotFactory import PlotResultsFactory
from Classes_RL.Boltzmann_solver import BoltzmannNonStationary
from Classes_RL.Bayesian_solver import Bayesian, BayesianNonStationaryUMUV, BayesianNonStationaryUMKV
from Classes_RL.Bayesian_sliding_solver import BayesianSlidingWindowUMKV, BayesianSlidingWindowUMUV
from Classes_RL.Kalman_filter_solver import KalmanEpsilonGreedy, KalmanThompsonSoftmax, KalmanUCBSoftmax, KalmanThompsonGreedy
from Classes_RL.Boltzmann_solver import BoltzmannSlidingWindowUCB, BoltzmannUCBNonStationary
from Classes_RL.SolverFactory import Models
from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.Oracle import Oracle
import matplotlib.pyplot as plt

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
n_actions = 10
n_iteration = 20
n_episode = 2

train_parameter = {"n_action": n_actions, "n_episode": n_episode}
parameters = {"name": "RndWalk", "range": 10, "std": 1}
# parameters = {"name": "sin", "frequency": 2, "amplitude": 1}

fac = BanditsFactory(parameters=parameters, n_iteration=n_iteration, n_action=n_actions)
bandits = fac.create()

oracle = Oracle(bandits=bandits, n_iteration=n_iteration, n_episode=n_episode)

mu_aa = oracle.mu_known[5, :].tolist()

import matplotlib.pyplot as plt
plt.bar(mu_aa)

