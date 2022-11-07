import numpy as np
from typing import Tuple, List, Dict, Any
from Classes_RL.Utilities import Utilities

np.set_printoptions(precision=4, suppress=True)
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
import matplotlib.pyplot as plt
import random
from Classes_RL.Results_solver import ResultsSolver

class Boltzmann(Solver):

    optimistic_Q: int
    Q_mean: float
    Q: np.ndarray
    P: np.ndarray
    temperature: float
    n: int
    N: np.ndarray

    def __init__(self, bandits: List[Bandit], oracle_data, n_iteration, n_episode):

        self.bandits = bandits
        self.oracle_data = oracle_data

        self.a = len(bandits)
        self.A = np.arange(self.a)

        self.n_iteration = n_iteration
        self.n_episode = n_episode
        self.n_action = len(bandits)

        self.results_solver = ResultsSolver(bandits, n_iteration, n_episode)

    def run(self):

        self.reset()
        self.results_solver.start()

        for i in range(self.n_episode):

            self.episode = i

            for k in range(self.n_iteration):

                self.iteration = k

                a, r = self.pull(k=k)

                self.results_solver.P_action[i, k] = self.N / self.n
                self.results_solver.P[i, k] = self.P

                self.results_solver.reward[i, k] = r
                self.results_solver.regret[i, k] = self.oracle_data["max_reward"][k] - r
                self.results_solver.action[i, k] = a

                self.results_solver.Q[i, k] = self.Q
                self.results_solver.Q_mean[i, k] = self.Q_mean
                self.results_solver.Q_mean_weighted[i, k] = np.average(self.results_solver.Q[i, k], weights=self.P)

                self.results_solver.cross_entropy[i, k] = Utilities.cross_entropy(p=self.oracle_data["P_action"][k], q=self.results_solver.P_action[i, k])
                self.results_solver.ae[i, k] = np.abs(self.oracle_data["Q_mean_weighted"][k] - self.results_solver.Q_mean_weighted[i, k])

                # just idea
                self.results_solver.regret_a[i, k] = np.mean((self.oracle_data["mu_known"][k] - self.Q)**2)

            self.results_solver.action_total[i] = self.N
            self.results_solver.cum_reward[i] = np.cumsum(self.results_solver.reward[i, :])
            self.results_solver.cum_regret[i] = np.cumsum(self.results_solver.regret[i, :])
            self.results_solver.cum_cross_entropy[i] = np.cumsum(self.results_solver.cross_entropy[i, :])
            self.results_solver.cum_ae[i] = np.cumsum(self.results_solver.ae[i, :])

            # just idea
            self.results_solver.cum_regret_a[i] = np.cumsum(self.results_solver.regret_a[i, :])

            self.reset()

        results, results_mean = self.results_solver.end()

        return results, results_mean

    def pull(self, k):

        # Select action
        a = self.select_action()

        # Get the reward for the current action and best action
        r = self.bandits[a].get_reward(iteration=self.n)

        self.n += 1
        self.N[a] += 1

        self.update_Q(a=a, r=r)

        return a, r

    def reset(self):

        self.n = 0  # Number of total step
        self.N = np.zeros(self.n_action)

        # Q-value
        self.Q_mean = self.optimistic_Q
        self.Q = np.zeros(self.a) + self.optimistic_Q  # Value of each arm

class BoltzmannStationary(Boltzmann):

    def set_hyperparameters(self, hyperparameters: dict):

        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.temperature = hyperparameters["temperature"]

    def update_Q(self, r: float, a: int):

        self.Q_mean = self.Q_mean + (1. / (self.n)) * (r - self.Q_mean)  # Add 1 to avoid dividing by zero
        self.Q[a] = self.Q[a] + (1. / (self.N[a])) * (r - self.Q[a])

    def select_action(self):

        self.P = Utilities.softmax_temperature(self.Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a

class BoltzmannUCBStationary(Boltzmann):

    def set_hyperparameters(self, hyperparameters: dict):

        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.temperature = hyperparameters["temperature"]
        self.c = hyperparameters["c"]

    def update_Q(self, r: float, a: int):

        self.Q_mean = self.Q_mean + (1. / (self.n)) * (r - self.Q_mean)  # Add 1 to avoid dividing by zero
        self.Q[a] = self.Q[a] + (1. / (self.N[a])) * (r - self.Q[a])

    def select_action(self):

        temp_Q = [self.Q[i] + self.c * np.sqrt(np.log(self.n + 1.) / (self.N[i] + 1.)) for i in range(len(self.Q))]  # Add 1 to avoid dividing by zero
        temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

        self.P = Utilities.softmax_temperature(temp_Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a

class BoltzmannNonStationary(Boltzmann):

    def set_hyperparameters(self, hyperparameters: dict):

        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.solver_name = hyperparameters["solver_name"]
        self.temperature = hyperparameters["temperature"]
        self.alpha = hyperparameters["alpha"]

    def update_Q(self, r: float, a: int):

        self.Q_mean = self.Q_mean + self.alpha * (r - self.Q_mean)
        self.Q[a] += self.alpha * (r - self.Q[a])

    def select_action(self):

        self.P = Utilities.softmax_temperature(self.Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a

class BoltzmannUCBNonStationary(Boltzmann):

    def set_hyperparameters(self, hyperparameters: dict):

        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.temperature = hyperparameters["temperature"]
        self.c = hyperparameters["c"]
        self.alpha = hyperparameters["alpha"]

    def update_Q(self, r: float, a: int):

        self.Q_mean = self.Q_mean + self.alpha * (r - self.Q_mean)
        self.Q[a] += self.alpha * (r - self.Q[a])

    def select_action(self):

        temp_Q = [self.Q[i] + self.c * np.sqrt(np.log(self.n + 1.) / (self.N[i] + 1.)) for i in range(len(self.Q))]  # Add 1 to avoid dividing by zero
        temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

        self.P = Utilities.softmax_temperature(temp_Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a

class BoltzmannSlidingWindowUCB(Boltzmann):

    def set_hyperparameters(self, hyperparameters: dict):

        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.temperature = hyperparameters["temperature"]
        self.c = hyperparameters["c"]
        self.window = hyperparameters["window"]

        self.n_w = 0.
        self.N_w = np.zeros(self.a)

    def update_Q(self, r: float, a: int):

        if self.iteration < self.window:

            self.Q[a] = self.Q[a] + (1. / (self.N[a])) * (r - self.Q[a])
            self.Q_mean = self.Q_mean + (1. / self.n) * (r - self.Q_mean)

        else:
            Q_w = self.results_solver.Q[self.episode, self.iteration - self.window, :].copy() # add optimistic for option don't have any information
            Q_mean_w = self.results_solver.Q_mean[self.episode, self.iteration - self.window].copy()
            # self.Q_mean = self.optimistic_Q
            self.n_w = 0.
            self.N_w = np.zeros(self.a)

            #todo: In order to update the information in the previous time series, the time series is shifted. (self.iteration - self.window + 1)
            for w in range(self.iteration - self.window + 1, self.iteration):

                a_w = int(self.results_solver.action[self.episode, w])
                r_w = self.results_solver.reward[self.episode, w]

                self.N_w[a_w] += 1
                self.n_w += 1

                Q_w[a_w] = Q_w[a_w] + (1. / (self.N_w[a_w])) * (r_w - Q_w[a_w])
                Q_mean_w = Q_mean_w + (1. / self.n_w) * (r_w - Q_mean_w)

            #todo: I think it is necessary to add this information because we are not utilizing the current time series information (a, r).
            self.N_w[a] += 1
            self.n_w += 1

            self.Q[a] = Q_w[a] + (1. / self.N_w[a]) * (r - Q_w[a])
            self.Q_mean = Q_mean_w + (1. / self.n_w) * (r - Q_mean_w)

        self.Q[np.less(self.Q, 0)]=0

    def select_action(self):

        if self.iteration < self.window:

            temp_Q = [self.Q[i] + self.c * np.sqrt(np.log(self.n + 1.) / (self.N[i] + 1.)) for i in range(len(self.Q))]  # Add 1 to avoid dividing by zero
            temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

        else:

            temp_Q = [self.Q[i] + self.c * np.sqrt(np.log(self.n_w + 1) / (self.N_w[i] + 1)) for i in range(len(self.Q))]  # Add 1 to avoid dividing by zero
            temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

        self.P = Utilities.softmax_temperature(temp_Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a