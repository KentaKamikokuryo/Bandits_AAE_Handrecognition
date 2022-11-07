import numpy as np
from typing import Tuple, List, Dict, Any
from Classes_RL.Utilities import Utilities
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
import random
np.set_printoptions(precision=4, suppress=True)
from Classes_RL.Results_solver import ResultsSolver

class Kalman(Solver):

    mu_bar: np.ndarray
    tau_bar: np.ndarray
    var_bar: np.ndarray
    kalman_gain: np.ndarray
    innovation_var: float
    noise_var: float
    optimistic_Q: int
    P: np.ndarray

    def __init__(self, bandits: List[Bandit], oracle_data, n_iteration, n_episode):

        self.bandits = bandits
        self.oracle_data = oracle_data

        self.a = len(self.bandits)
        self.A = np.arange(self.a)

        self.n_iteration = n_iteration
        self.n_episode = n_episode
        self.n_action = len(bandits)

        self.results_solver = ResultsSolver(bandits, n_iteration, n_episode)

    def run(self):

        self.reset()
        self.results_solver.start()

        for i in range(self.n_episode):

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
                self.results_solver.regret_a[i, k] = np.mean((self.oracle_data["mu_known"][k] - self.mu_bar)**2)

                self.results_solver.mu_bar[i, k] = self.mu_bar
                self.results_solver.tau_bar[i, k] = self.tau_bar
                self.results_solver.var_bar[i, k] = self.var_bar

                self.results_solver.kalman_gain[i, k] = self.kalman_gain

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

    def update_Q(self, r: float, a: int):

        self.kalman_gain = np.zeros(self.a)
        self.kalman_gain[a] = (self.var_bar[a] + self.innovation_var) / (self.var_bar[a] + self.innovation_var + self.noise_var)

        self.Q_mean = self.Q_mean + np.mean(self.kalman_gain) * (r - self.Q_mean)
        self.Q[a] = self.Q[a] + self.kalman_gain[a] * (r - self.Q[a])

        self.mu_bar = self.Q.copy()
        self.var_bar = (1 - self.kalman_gain) * (self.var_bar + self.innovation_var)
        self.tau_bar = np.array([1. / v for v in self.var_bar])

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
        self.N = np.zeros(self.n_action)  # Number of step for each action

        # Q-value
        self.Q_mean = 0
        self.Q = np.zeros(self.a) + self.optimistic_Q

        self.mu_bar = np.ones(shape=self.a) + self.optimistic_Q
        self.tau_bar = np.ones(shape=self.a) * 0.001  # the posterior precision
        self.var_bar = np.ones(shape=self.a) * 1./self.tau_bar[0]  # initial variance for kalman filter

class KalmanEpsilonGreedy(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.epsilon = hyperparameters["epsilon"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        p = random.random()

        if p < self.epsilon:
            # with probability epsilon, pull random action.
            a = Utilities.random_choice(self.Q)
            self.P = np.array([1 / self.n_action for i in range(self.n_action)])

        else:
            # with probability 1 - epsilon, pull current-best action.
            a = Utilities.randargmax(self.Q)

            self.P = np.zeros(shape=(self.n_action))
            self.P[a] = 1.0

        return a

class KalmanGreedy(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        a = Utilities.randargmax(self.Q)
        self.P = np.zeros(shape=(self.n_action))
        self.P[a] = 1.0

        return a

class KalmanUCB(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.c = hyperparameters["c"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        temp_Q = [self.Q[i] + self.c * np.sqrt(self.var_bar[i] + self.innovation_var) for i in range(len(self.Q))]
        temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

        a = Utilities.randargmax(temp_Q)

        self.P = np.zeros(shape=(self.n_action))
        self.P[a] = 1.0

        return a

class KalmanSoftmax(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.temperature = hyperparameters["temperature"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        self.P = Utilities.softmax_temperature(self.Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a

class KalmanUCBSoftmax(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.temperature = hyperparameters["temperature"]
        self.c = hyperparameters["c"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        temp_Q = [self.Q[i] + self.c * np.sqrt(self.var_bar[i] + self.innovation_var) for i in range(len(self.Q))]  # Add 1 to avoid dividing by zero
        temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

        self.P = Utilities.softmax_temperature(x=temp_Q, t=self.temperature)
        a = np.random.choice(self.A, p=self.P)

        return a

class KalmanThompsonGreedy(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        values = []

        for i in range(self.a):
            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / self.tau_bar[i]))  # use precision for sampling (Bad for static behavior since the variance become smaller, smaller ....)
            if temp < 0:  # We want positive value only
                temp = 0
            values.append(temp)
        values = np.array(values)

        a = Utilities.randargmax(values)

        self.P = np.zeros(shape=(self.n_action))
        self.P[a] = 1.0

        return a

class KalmanThompsonSoftmax(Kalman):

    def set_hyperparameters(self, hyperparameters: dict):

        self.solver_name = hyperparameters["solver_name"]
        self.temperature = hyperparameters["temperature"]
        self.innovation_var = hyperparameters["innovation_var"]
        self.noise_var = hyperparameters["noise_var"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]

    def select_action(self):

        values = []

        for i in range(self.a):
            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / self.tau_bar[i]))  # use precision for sampling (Bad for static behavior since the variance become smaller, smaller ....)
            if temp < 0:  # We want positive value only
                temp = 0
            values.append(temp)
        values = np.array(values)

        self.P = Utilities.softmax_temperature(x=values, t=self.temperature)  # Can be interesting to have more exploration
        a = np.random.choice(self.A, p=self.P)

        return a