import numpy as np
from typing import List
from Classes_RL.Utilities import Utilities

np.set_printoptions(precision=4, suppress=True)
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver
from Classes_RL.Results_solver import ResultsSolver

class Bayesian(Solver):

    mu_bar: np.ndarray
    tau_bar: np.ndarray
    var_bar: np.ndarray
    optimistic_Q: float
    var_0: float
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

            for k in range(self.n_iteration):

                self.iteration = k

                a, r = self.pull(k=k)

                # self.results_solver.P_cum[i, k] = self.N / self.n
                # self.results_solver.P[i, k] = Utilities.softmax_temperature(self.Q)

                self.results_solver.P_action[i, k] = self.N / self.n
                self.results_solver.P[i, k] = self.P
                self.results_solver.cross_entropy[i, k] = Utilities.cross_entropy(p=self.oracle_data["P"][k], q=self.results_solver.P[i, k])

                self.results_solver.reward[i, k] = r
                self.results_solver.regret[i, k] = self.oracle_data["max_reward"][k] - r
                self.results_solver.action[i, k] = a

                self.results_solver.Q[i, k] = self.mu_bar
                self.results_solver.Q_mean[i, k] = self.Q_mean
                self.results_solver.Q_mean_weighted[i, k] = np.average(self.results_solver.Q[i, k], weights=self.P)

                self.results_solver.cross_entropy[i, k] = Utilities.cross_entropy(p=self.oracle_data["P_action"][k], q=self.results_solver.P_action[i, k])
                self.results_solver.ae[i, k] = np.abs(self.oracle_data["Q_mean_weighted"][k] - self.results_solver.Q_mean_weighted[i, k])

                # just idea
                self.results_solver.regret_a[i, k] = np.mean((self.oracle_data["mu_known"][k] - self.mu_bar)**2)

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

        self.choice = a

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

        self.mu_bar = np.zeros(shape=self.a) + self.optimistic_Q
        self.var_bar = np.ones(shape=self.a) * self.var_0  # initial variance
        self.tau_bar = np.ones(shape=self.a) * 1. / self.var_bar  # the posterior precision

class BayesianStationaryUMKV(Bayesian):

    hyperparameters: dict
    solver_name: dict
    is_stationary: bool
    optimistic_Q: int

    def set_hyperparameters(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.solver_name = hyperparameters["solver_name"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.var_0 = hyperparameters["initial_var"]

    def update_Q(self, a, r):

        """ increase the number of times this socket has been used and improve the estimate of the
            value (the mean) by combining the new value 'x' with the current mean """

        self.Q_mean = self.Q_mean + (1 / (self.n + 1)) * (r - self.Q_mean)  # Add 1 to avoid dividing by zero

        # Update the mean of the posterior
        numerator = (self.tau_bar[a] * self.mu_bar[a]) + (self.N[a] * r)
        denominator = self.tau_bar[a] + self.N[a]
        self.mu_bar[a] = numerator / denominator

        # Increase the sum the precision
        tau = 1  # Considering the same precision in the data
        self.tau_bar[a] = self.N[a] * tau
        self.var_bar[a] = 1 / self.tau_bar[a]  # Variance based on the precision (Variance for the sampling

    def select_action(self):

        values = []

        for i in range(self.a):
            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / self.tau_bar[i]))  # use precision for sampling (Bad for static behavior since the variance become smaller, smaller ....)
            if temp < 0:  # We want positive value only
                temp = 0
            values.append(temp)
        values = np.array(values)

        self.P = Utilities.softmax_temperature(x=values, t=2)
        a = np.random.choice(self.A, p=self.P)

        return a

class BayesianNonStationaryUMKV(Bayesian):

    hyperparameters: dict
    solver_name: str
    is_stationary: bool
    alpha: int
    t: int
    optimistic_Q: int

    def set_hyperparameters(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.solver_name = hyperparameters["solver_name"]
        self.alpha = hyperparameters["alpha"]
        self.t = hyperparameters["temperature"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.var_0 = hyperparameters["initial_var"]

    def update_Q(self, a, r):

        ''' increase the number of times this socket has been used and improve the estimate of the
            value (the mean) by combining the new value 'x' with the current mean '''

        self.Q_mean = self.Q_mean + self.alpha * (r - self.Q_mean)

        # Update the mean of the posterior
        numerator = (self.tau_bar[a] * self.mu_bar[a]) + (self.N[a] * r)
        denominator = self.tau_bar[a] + self.N[a]
        self.mu_bar[a] = numerator / denominator

        # Increase the sum the precision
        tau = 1  # Considering the same precision in the data
        self.tau_bar[a] = self.N[a] * tau
        self.var_bar[a] = 1 / self.tau_bar[a]  # Variance based on the precision (Variance for the sampling

    def select_action(self):

        values = []

        for i in range(self.a):
            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / self.tau_bar[i]))  # use precision for sampling (Bad for static behavior since the variance become smaller, smaller ....)
            if temp < 0:  # We want positive value only
                temp = 0
            values.append(temp)
        values = np.array(values)

        self.P = Utilities.softmax_temperature(x=values, t=self.t)  # Can be interesting to have more exploration
        a = np.random.choice(self.A, p=self.P)

        return a

class BayesianStationaryUMUV(Bayesian):

    hyperparameters: dict
    solver_name: str
    t: int
    is_stationary: bool
    optimistic_Q: int
    alpha_g: list
    beta_g: list

    def set_hyperparameters(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.solver_name = hyperparameters["solver_name"]
        self.t = hyperparameters["temperature"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.var_0 = hyperparameters["initial_var"]

        # Gaussian part for unknown mean and unknown variance (normal - gamma)
        self.solver_name = hyperparameters["solver_name"]

        self.alpha_g = [hyperparameters["alpha_g"] for i in range(self.a)]  # gamma shape parameter
        self.beta_g = [hyperparameters["beta_g"] for i in range(self.a)]  # gamma rate parameter

    def update_Q(self, a, r):

        ''' increase the number of times this socket has been used and improve the estimate of the
            value (the mean) by combining the new value 'x' with the current mean '''

        self.Q_mean = self.Q_mean + (1 / (self.n + 1)) * (r - self.Q_mean)  # Add 1 to avoid dividing by zero

        v = self.N[a]
        n = 1

        self.alpha_g[a] = self.alpha_g[a] + n / 2

        numerator = n * v * (r - self.mu_bar[a]) ** 2
        denominator = (v + n) * 2

        self.beta_g[a] = self.beta_g[a] + numerator / denominator

        # estimate the variance - calculate the mean from the gamma
        # self.var_bar[a] = self.beta[a] / (self.alpha[a] + 1)

        numerator_mu = v * self.mu_bar[a] + n * r
        denominator_mu = v + n
        self.mu_bar[a] = numerator_mu / denominator_mu

    def select_action(self):

        values = []

        for i in range(self.a):

            precision = np.random.gamma(self.alpha_g[i], 1/self.beta_g[i])  # shape = 1/rate (beta is rate)

            if precision == 0 or self.n == 0:
                precision = 0.001

            self.var_bar[i] = 1 / precision
            self.tau_bar[i] = precision
            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / precision))

            if temp < 0:  # We want positive value only
                temp = 0

            values.append(temp)

        values = np.array(values)

        self.P = Utilities.softmax_temperature(x=values, t=self.t)  # Can be interesting to have more exploration
        a = np.random.choice(self.A, p=self.P)

        return a

class BayesianNonStationaryUMUV(Bayesian):

    hyperparameters: dict
    solver_name: str
    is_stationary: bool
    with_lambda: bool
    optimistic_Q: int
    solver_name: str
    alpha_g: list
    beta_g: list

    def set_hyperparameters(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.solver_name = hyperparameters["solver_name"]
        self.alpha = hyperparameters["alpha"]
        self.t = hyperparameters["temperature"]
        self.optimistic_Q = hyperparameters["optimistic_Q"]
        self.var_0 = hyperparameters["initial_var"]


        # Gaussian part for unknown mean and unknown variance (normal - gamma)
        self.solver_name = hyperparameters["solver_name"]

        self.alpha_g = [hyperparameters["alpha_g"] for i in range(self.a)]  # gamma shape parameter
        self.beta_g = [hyperparameters["beta_g"] for i in range(self.a)]  # gamma rate parameter

        self.mu_bar = [1 for a in range(self.a)]  # the prior (estimated) mean
        self.var_bar = [self.beta_g[i] / (self.alpha_g[i]) for i in range(self.a)]  # the prior (estimated) variance
        self.tau_bar = [0 for a in range(self.a)]

    def update_Q(self, a, r):

        ''' increase the number of times this socket has been used and improve the estimate of the
            value (the mean) by combining the new value 'x' with the current mean '''

        self.Q_mean = self.Q_mean + self.alpha * (r - self.Q_mean)

        v = self.N[a]
        n = 1

        self.alpha_g[a] = self.alpha_g[a] + n / 2

        numerator = n * v * (r - self.mu_bar[a]) ** 2
        denominator = (v + n) * 2

        self.beta_g[a] = self.beta_g[a] + numerator / denominator

        # estimate the variance - calculate the mean from the gamma
        # self.var_bar[a] = self.beta[a] / (self.alpha[a] * v)

        numerator_mu = v * self.mu_bar[a] + n * r
        denominator_mu = v + n
        self.mu_bar[a] = numerator_mu / denominator_mu

    def select_action(self):

        values = []

        for i in range(self.a):

            precision = np.random.gamma(self.alpha_g[i], 1 / self.beta_g[i])  # shape = 1/rate (beta is rate)

            if precision == 0 or self.n == 0:
                precision = 0.001

            self.tau_bar[i] = precision
            self.var_bar[i] = 1 / precision

            temp = np.random.normal(self.mu_bar[i], np.sqrt(1 / precision))

            if temp < 0:  # We want positive value only
                temp = 0

            values.append(temp)

        values = np.array(values)

        self.P = Utilities.softmax_temperature(x=values, t=self.t)
        a = np.random.choice(self.A, p=self.P)

        return a


