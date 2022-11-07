import numpy as np
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from Classes_RL.Interfaces import IBehavior
import random

class BehaviorStaticStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, n_iteration: int):

        self.parameters = parameters
        self.n_iteration = n_iteration

        self.name = parameters["name"]

    def execute(self):

        mu = np.random.uniform(0., 10.)
        std = 1.

        for i in range(self.n_iteration):

            var = std ** 2.
            tau = 1. / var

            self.mu_iterations.append(mu)
            self.std_iterations.append(std)
            self.var_iterations.append(var)
            self.tau_iterations.append(tau)

class BehaviorSinStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, n_iteration: int):

        self.parameters = parameters
        self.n_iteration = n_iteration

        self.name = parameters["name"]
        self.amplitude_base = parameters["amplitude_base"]
        self.frequency_range = parameters["frequency_range"]
        self.std_amplitude = parameters["std_amplitude"]

    def execute(self):

        mu_base = np.random.uniform(self.amplitude_base[0],  self.amplitude_base[1])
        factor = np.random.uniform(1.0, 1.0 + 2 * self.std_amplitude)
        frequency = np.random.uniform(self.frequency_range[0], self.frequency_range[1])

        std = 1.

        self.phase = random.randrange(0, 360, 60)

        phase = random.randrange(0, 360)

        for i in range(self.n_iteration):

            add = np.sin(np.radians(phase) + 2. * i * np.pi * frequency / self.n_iteration)

            mu = mu_base + add * factor
            std = std
            var = std ** 2.
            tau = 1. / var

            self.mu_iterations.append(mu)
            self.std_iterations.append(std)
            self.var_iterations.append(var)
            self.tau_iterations.append(tau)

        # Check for zero values
        min = np.min(self.mu_iterations)

        if min < 0:
            self.mu_iterations = [m - min * 2 for m in self.mu_iterations]

class BehaviorUnsteadyStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, n_iteration: int):

        self.parameters = parameters
        self.n_iteration = n_iteration

        self.name = parameters["name"]
        self.n_change = parameters["n_change"]
        self.min_iteration_change = parameters["min_iteration_change"]

    def execute(self):

        mu = np.random.uniform(0., 10.)
        std = 1.

        c = 0.
        ci = 0.

        for i in range(self.n_iteration):

            var = std ** 2.
            tau = 1. / var

            n_change = self.n_change
            min_iteration_change = self.min_iteration_change
            ci += 1.
            p = np.random.uniform(0., 1.)
            change1_5 = np.random.uniform(0., 5.)
            change5_10 = np.random.uniform(6., 10.)

            if c <= n_change and ci >= min_iteration_change and p <= 0.1:

                c += 1.
                ci = 0.
                if mu > 5:
                    mu = change1_5
                else:
                    mu = change5_10

            else:

                mu = mu

            self.mu_iterations.append(mu)
            self.std_iterations.append(std)
            self.var_iterations.append(var)
            self.tau_iterations.append(tau)

class BehaviorLogStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, n_iteration: int):

        # parameters = {"name": "log", "range": 20}

        self.parameters = parameters
        self.n_iteration = n_iteration

        self.name = parameters["name"]
        self.range = parameters["range"]

    def execute(self):

        mu_start = np.random.uniform(0., self.range)
        mu_end = mu_start + np.random.uniform(self.range / 2, self.range)

        x = np.arange(1, self.n_iteration + 1, 1)
        y = np.log(x)

        # Scale log curve
        a = (mu_end - mu_start) / (y[-1] - y[0])
        b = mu_start

        y = y * a + b

        std = 1.

        for i in range(self.n_iteration):

            var = std ** 2.
            tau = 1. / var

            self.mu_iterations.append(y[i])
            self.std_iterations.append(std)
            self.var_iterations.append(var)
            self.tau_iterations.append(tau)

class BehaviorExpStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, n_iteration: int):

        # parameters = {"name": "exp", "range": 10, "range_exp": 5}

        self.parameters = parameters
        self.n_iteration = n_iteration

        self.name = parameters["name"]
        self.range = parameters["range"]
        self.range_exp = parameters["range_exp"]

    def execute(self):

        mu = np.random.uniform(0., self.range)

        x_start = np.random.uniform(0., self.range_exp)
        x_end = self.range_exp + np.random.uniform(0., self.range_exp)

        x_temp = np.arange(x_start, x_end, (x_end - x_start) / self.n_iteration)
        y = np.exp(x_temp)

        # Scale log curve
        a = (mu) / (y[-1] - y[0])
        b = mu

        y = y * a + b

        std = 1.

        for i in range(self.n_iteration):

            var = std ** 2.
            tau = 1. / var

            self.mu_iterations.append(y[i])
            self.std_iterations.append(std)
            self.var_iterations.append(var)
            self.tau_iterations.append(tau)

class BehaviorRndWalkStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, n_iteration: int):

        # parameters = {"name": "randomWalk", "range": 10, "std": 1}

        self.parameters = parameters
        self.n_iteration = n_iteration

        self.name = parameters["name"]
        self.range = parameters["range"]
        self.std = parameters["std"]

    def execute(self):

        p = np.random.uniform(0., 1.)
        if p < 0.5:
            mu_start = np.random.uniform(2 * self.range / 3, self.range)
            mu_end =  np.random.uniform(0., self.range/3)

        # if 0.3 <= p < 0.7:
        #     mu_start = np.random.uniform(0., self.range)
        #     mu_end = np.random.uniform(0. , self.range)

        else:
            mu_start = np.random.uniform(0., self.range / 3)
            mu_end = np.random.uniform(2 * self.range / 3,  self.range)


        a = [np.random.normal(0, self.std**2) for i in range(self.n_iteration)]

        steps = np.random.choice(a=a, size=self.n_iteration)
        y = np.concatenate([steps]).cumsum(0) + mu_start

        y = y - np.min(y)

        a = (mu_end - mu_start) / (np.max(y) - np.min(y))
        b = mu_start

        y = y * a + b

        std = 1.

        for i in range(self.n_iteration):

            var = std ** 2.
            tau = 1. / var

            self.mu_iterations.append(y[i])
            self.std_iterations.append(std)
            self.var_iterations.append(var)
            self.tau_iterations.append(tau)

class BehaviorRealStrategy(IBehavior):

    def __init__(self):

        self.mu_iterations = []
        self.std_iterations = []
        self.var_iterations = []
        self.tau_iterations = []

    def set_parameters(self, parameters: dict, mu: list, std: list):

        self.parameters = parameters
        self.n_iteration = len(mu)

        self.name = parameters["name"]

        self.mu = mu
        self.std = std

    def execute(self):

        self.mu_iterations = list(self.mu)
        self.std_iterations = self.std
        self.var_iterations = [self.std_iterations[i]**2 for i in range(self.n_iteration)]
        self.tau_iterations = [1./self.var_iterations[i] for i in range(self.n_iteration)]

        # for i in range(self.n_iteration):
        #
        #     std = self.std[i]  # For now, std is fixed (but will be changed later)
        #     var = self.std[i] ** 2.
        #
        #     tau = 1. / var
        #
        #     self.mu_iterations.append(self.mu[i])
        #     self.std_iterations.append(std)
        #     self.var_iterations.append(var)
        #     self.tau_iterations.append(tau)

        # std_iterations = self.d_std
        # var_iterations = [std_iterations[i]**2 for i in range(self.n_iteration)]
        # tau_iterations = [1./var_iterations[i] for i in range(self.n_iteration)]

class Bandit():

    behavior_strategy: IBehavior

    def __init__(self, behavior_strategy: IBehavior = None):

        if behavior_strategy is not None:
            self.behavior_strategy = behavior_strategy
        else:
            self.behavior_strategy = BehaviorStaticStrategy()

        self.mu_iterations = self.behavior_strategy.mu_iterations
        self.std_iterations = self.behavior_strategy.std_iterations
        self.var_iterations = self.behavior_strategy.var_iterations
        self.tau_iterations = self.behavior_strategy.tau_iterations

    def get_reward(self, iteration) -> float:
        r = np.random.normal(self.behavior_strategy.mu_iterations[iteration], self.behavior_strategy.std_iterations[iteration])
        return r

    def get_max_reward(self, iteration) -> float:
        r = self.behavior_strategy.mu_iterations[iteration] + 3. * self.behavior_strategy.std_iterations[iteration]
        return r

    def get_n_iteration(self):
        return self.behavior_strategy.n_iteration

# plt.figure()
# for i in range(10):
#     parameters = {"name": "log", "range": 20}
#     n_iteration = 1000
#     self = BehaviorLogStrategy()
#     self.set_parameters(parameters=parameters, n_iteration=n_iteration)
#     self.execute()
#     plt.plot(self.mu_iterations)
#
# plt.figure()
# for i in range(10):
#     parameters = {"name": "exp", "range": 10, "range_exp": 5}
#     n_iteration = 1000
#     self = BehaviorExpStrategy()
#     self.set_parameters(parameters=parameters, n_iteration=n_iteration)
#     self.execute()
#     plt.plot(self.mu_iterations)
#
# plt.figure()
# for i in range(10):
#     parameters = {"name": "RndWalk", "range": 10, "std": 1}
#     n_iteration = 1000
#     self = BehaviorRndWalkStrategy()
#     self.set_parameters(parameters=parameters, n_iteration=n_iteration)
#     self.execute()
#     plt.plot(self.mu_iterations)
