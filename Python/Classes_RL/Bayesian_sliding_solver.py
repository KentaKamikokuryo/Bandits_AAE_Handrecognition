from typing import List
import numpy as np
from Classes_RL.Utilities import Utilities
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
from Classes_RL.Results_solver import ResultsSolver

class Sliding(Solver):

    mu_bar: np.ndarray
    tau_bar: np.ndarray
    var_bar: np.ndarray
    window: int
    optimistic_Q: float
    var_0: float
    results_solver: ResultsSolver
    n: int
    N: np.ndarray
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

            self.episode = i

            for k in range(self.n_iteration):

                self.iteration = k

                a, r = self.pull(k=k)

                # self.results_solver.P_cum[i, k] = self.N / self.n
                # self.results_solver.P[i, k] = Utilities.softmax_temperature(self.Q)

                self.results_solver.P_action[i, k] = self.N / self.n
                self.results_solver.P[i, k] = self.P

                self.results_solver.reward[i, k] = r
                self.results_solver.regret[i, k] = self.oracle_data["max_reward"][k] - r
                self.results_solver.action[i, k] = a

                self.results_solver.Q[i, k] = self.mu_bar
                self.results_solver.Q_mean[i, k] = self.Q_mean
                self.results_solver.Q_mean_weighted[i, k] = np.average(self.results_solver.Q[i, k], weights=self.P)


                # just idea
                self.results_solver.regret_a[i, k] = np.mean((self.oracle_data["mu_known"][k] - self.mu_bar) ** 2)

                # Error
                self.results_solver.cross_entropy[i, k] = Utilities.cross_entropy(p=self.oracle_data["P_action"][k], q=self.results_solver.P_action[i, k])
                self.results_solver.ae[i, k] = np.abs(self.oracle_data["Q_mean_weighted"][k] - self.results_solver.Q_mean_weighted[i, k])

                self.results_solver.mu_bar[i, k] = self.mu_bar
                self.results_solver.tau_bar[i, k] = self.tau_bar
                self.results_solver.var_bar[i, k] = self.var_bar

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

        # Update action count
        self.n += 1
        self.N[a] += 1

        # Update Q
        self.update_Q(a=a, r=r)

        return a, r

    def reset(self):

        self.n = 0  # Number of total step
        self.N = np.zeros(self.n_action)  # Number of step for each action

        # Q-value
        self.Q_mean = 0

        # Bayesian stuff
        self.mu_bar = np.zeros(shape=self.a) + self.optimistic_Q
        self.var_bar = np.ones(shape=self.a) * self.var_0  # initial variance
        self.tau_bar = np.ones(shape=self.a) * 1. / self.var_bar  # the posterior precision

class BayesianSlidingWindowUMKV(Sliding):

    hyperparameters: dict
    solver_name: str
    is_stationary: bool
    t = int
    optimistic_Q: int
    window: int
    var_bar_known: list

    def set_hyperparameters(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.solver_name = hyperparameters["solver_name"]
        self.t = hyperparameters["temperature"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.window = hyperparameters["window"]
        self.var_0 = hyperparameters["initial_var"]

    def update_Q(self, a, r):

        ''' increase the number of times this socket has been used and improve the estimate of the
            value (the mean) by combining the new value 'x' with the current mean '''

        # Update Bayesian model
        if self.iteration < self.window:

            # Update the mean of the posterior
            numerator = (self.tau_bar[a] * self.mu_bar[a]) + (self.N[a] * r)
            denominator = self.tau_bar[a] + self.N[a]
            self.mu_bar[a] = numerator / denominator

            # Increase the sum the precision
            tau = 1  # Considering the same precision in the data
            self.tau_bar[a] = self.N[a] * tau
            self.var_bar[a] = 1 / self.tau_bar[a]  # Variance based on the precision (Variance for the sampling
            self.Q_mean = self.Q_mean + (1 / self.n) * (r - self.Q_mean)

        else:   # Specific for the past n observation

            # Reset observation and set the first values as the values already in mu as the first values
            mu = self.results_solver.mu_bar[self.episode, self.iteration - self.window, :].copy()
            Q_mean_w = self.results_solver.Q_mean[self.episode, self.iteration - self.window].copy()
            # mu = np.zeros(self.n_action) + self.optimistic_Q
            self.tau_bar = np.ones(shape=self.a) * 1 / self.var_0
            self.var_bar = np.ones(shape=self.a) * 1. / self.tau_bar
            self.n_w = 0.
            self.N_w = np.zeros(self.a)

            for w in range(self.iteration - self.window + 1, self.iteration):

                # Get past action and reward
                a_w = int(self.results_solver.action[self.episode, w])
                r_w = self.results_solver.reward[self.episode, w]

                # Update action count
                self.N_w[a_w] += 1
                self.n_w += 1

                # Update the mean of the posterior
                numerator = (self.tau_bar[a_w] * mu[a_w]) + (self.N_w[a_w] * r_w)
                denominator = self.tau_bar[a_w] + self.N_w[a_w]
                mu[a_w] = numerator / denominator

                # Increase the sum the precision
                tau = 1  # Considering the same precision in the data
                self.tau_bar[a_w] = self.N_w[a_w] * tau
                self.var_bar[a_w] = 1 / self.tau_bar[a_w]  # Variance based on the precision (Variance for the sampling

                Q_mean_w = Q_mean_w + (1. / self.n_w) * (r_w - Q_mean_w)

            self.N_w[a] += 1
            self.n_w += 1
            self.mu_bar = mu
            self.Q_mean = Q_mean_w + (1. / self.n_w) * (r - Q_mean_w)

            # Update the mean of the posterior
            numerator = (self.tau_bar[a] * self.mu_bar[a]) + (self.N[a] * r)
            denominator = self.tau_bar[a] + self.N[a]
            self.mu_bar[a] = numerator / denominator

            # Increase the sum the precision
            tau = 1  # Considering the same precision in the data
            self.tau_bar[a] = self.N[a] * tau
            self.var_bar[a] = 1 / self.tau_bar[a]

    def select_action(self):

        values = []

        for i in range(self.a):
            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / self.tau_bar[i]))  # use precision for sampling (Bad for static behavior since the variance become smaller, smaller ....)
            if temp < 0:  # We want positive value only
                temp = 0
            values.append(temp)
        values = np.array(values)
        values = np.nan_to_num(values, neginf=0.00001, posinf=0.00001, nan=0.00001)

        self.P = Utilities.softmax_temperature(x=values, t=self.t)
        a = np.random.choice(self.A, p=self.P)

        return a

class BayesianSlidingWindowUMUV(Sliding):

    hyperparameters: dict
    solver_name: str
    is_stationary: bool
    t = int
    optimistic_Q: int
    window: int
    var_bar_known: list

    def set_hyperparameters(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.solver_name = hyperparameters["solver_name"]
        self.t = hyperparameters["temperature"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.window = hyperparameters["window"]
        self.var_0 = hyperparameters["initial_var"]

        # Gaussian part for unknown mean and unknown variance (normal - gamma)
        self.alpha = np.ones(shape=self.a) * hyperparameters["alpha_g"]  # gamma shape parameter
        self.beta = np.ones(shape=self.a) * hyperparameters["beta_g"]  # gamma rate parameter

        self.alpha_g = np.ones(shape=self.a) * hyperparameters["alpha_g"]
        self.beta_g = np.ones(shape=self.a) * hyperparameters["beta_g"]

    def update_Q(self, a, r):

        ''' increase the number of times this socket has been used and improve the estimate of the
            value (the mean) by combining the new value 'x' with the current mean '''

        self.Q_mean = self.Q_mean + (1 / self.window) * (r - self.Q_mean)

        v = self.N[a]
        n = 1

        if self.iteration < self.window:

            # Update the mean of the posterior

            self.alpha[a] = self.alpha[a] + n / 2

            numerator = n * v * (r - self.mu_bar[a]) ** 2
            denominator = (v + n) * 2

            self.beta[a] = self.beta[a] + numerator / denominator

            # # estimate the variance - calculate the mean from the gamma
            # self.var[a] = self.beta[a] / (self.alpha[a] * v)
            # self.tau[a] = 1 / self.var[a]

            numerator_mu = v * self.mu_bar[a] + n * r
            denominator_mu = v + n
            self.mu_bar[a] = numerator_mu / denominator_mu

        # Specific for the past n observation
        else:

            # Reset observation and set the first values as the values already in mu as the first values
            mu = self.results_solver.mu_bar[self.episode, self.iteration - self.window, :].copy()
            Q_mean_w = self.results_solver.Q_mean[self.episode, self.iteration - self.window].copy()
            # mu = np.zeros(self.n_action) + self.optimistic_Q
            self.alpha = self.alpha_g.copy()
            self.beta = self.beta_g.copy()
            self.n_w = 0.
            self.N_w = np.zeros(self.a)

            for w in range(self.iteration - self.window + 1, self.iteration):

                # Get past action and reward
                a_w = int(self.results_solver.action[self.episode, w])
                r_w = self.results_solver.reward[self.episode, w]

                # Update action count
                self.N_w[a_w] += 1
                self.n_w += 1
                v = self.N_w[a]

                self.alpha[a_w] = self.alpha[a_w] + n / 2

                numerator = n * v * (r_w - mu[a_w]) ** 2
                denominator = (v + n) * 2

                self.beta[a_w] = self.beta[a_w] + numerator / denominator

                # estimate the variance - calculate the mean from the gamma
                # self.var[a_w] = self.beta[a_w] / (self.alpha[a_w] * v)
                # self.tau[a_w] = 1 / self.var[a_w]

                numerator_mu = v * mu[a_w] + n * r_w
                denominator_mu = v + n
                mu[a_w] = numerator_mu / denominator_mu

                Q_mean_w = Q_mean_w + (1. / self.n_w) * (r_w - Q_mean_w)

            self.N_w[a] += 1
            self.n_w += 1
            self.mu_bar = mu
            self.Q_mean = Q_mean_w + (1. / self.n_w) * (r - Q_mean_w)

            # Update the mean of the posterior

            self.alpha[a] = self.alpha[a] + n / 2

            numerator = n * v * (r - self.mu_bar[a]) ** 2
            denominator = (v + n) * 2

            self.beta[a] = self.beta[a] + numerator / denominator

            numerator_mu = v * self.mu_bar[a] + n * r
            denominator_mu = v + n
            self.mu_bar[a] = numerator_mu / denominator_mu

    def select_action(self):

        values = []

        for i in range(self.a):

            precision = np.random.gamma(self.alpha[i], 1 / self.beta[i])  # shape = 1/rate (beta is rate)

            if precision == 0 or self.n == 0:
                precision = 0.001

            # estimate the variance - calculate the mean from the gamma
            self.tau_bar[i] = precision
            self.var_bar[i] = 1 / precision

            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / precision))

            if temp < 0:  # We want positive value only
                temp = 0

            values.append(temp)

        values = np.array(values)
        values = np.nan_to_num(values, neginf=0.00001, posinf=0.00001, nan=0.00001)

        self.P = Utilities.softmax_temperature(x=values, t=self.t)  # Can be interesting to have more exploration
        a = np.random.choice(self.A, p=self.P)

        return a