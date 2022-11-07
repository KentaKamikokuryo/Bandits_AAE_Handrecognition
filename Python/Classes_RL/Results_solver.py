import numpy as np
from typing import Tuple, List, Dict, Any
from Classes_RL.Bandit import Bandit
from Classes_RL.Utilities import Utilities


class ResultsSolver:

    def __init__(self, bandits: List[Bandit], n_iteration, n_episode):

        self.bandits = bandits
        self.a = len(bandits)
        self.A = np.arange(self.a)

        self.n_iteration = n_iteration
        self.n_episode = n_episode
        self.n_action = len(bandits)

        self.start()

    def start(self):

        self.reward = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.Q = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))
        self.Q_mean = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.Q_mean_weighted = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.action_total = np.zeros(shape=(self.n_episode, self.a))
        self.action = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.regret = np.zeros(shape=(self.n_episode, self.n_iteration))

        # self.P_cum = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))
        self.P_action = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))
        self.P = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))

        # just idea
        self.regret_a = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.cum_regret_a = np.zeros(shape=(self.n_episode, self.n_iteration))

        self.cross_entropy = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.ae = np.zeros(shape=(self.n_episode, self.n_iteration))  # Absolute error

        self.cum_reward = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.cum_regret = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.cum_cross_entropy = np.zeros(shape=(self.n_episode, self.n_iteration))
        self.cum_ae = np.zeros(shape=(self.n_episode, self.n_iteration))

        # Specific to Bayesian and Kalman
        self.mu_bar = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))
        self.tau_bar = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))
        self.var_bar = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))

        # Specific to Kalman only
        self.kalman_gain = np.zeros(shape=(self.n_episode, self.n_iteration, self.a))

    def end(self):

        # # Probability distribution history
        # for i in range(self.n_episode):
        #     for k in range(self.n_iteration):
        #         action_count = {i: 0.0 for i in range(self.n_action)}
        #         for aa in self.action[i, :k + 1]:
        #             action_count[aa] += 1.0
        #             count = np.array([action_count[i] for i in action_count.keys()])
        #             self.P_cum[i, k] = count / np.sum(count)

        results = {}
        results["reward"] = self.reward
        results["Q"] = self.Q
        results["Q_mean"] = self.Q_mean
        results["action_total"] = self.action_total
        results["action"] = self.action
        results["regret"] = self.regret

        results["P"] = self.P
        results["P_action"] = self.P_action

        # just idea
        results["regret_a"] = self.regret_a
        results["cum_regret_a"] = self.cum_regret_a

        results["cross_entropy"] = self.cross_entropy
        results["ae"] = self.ae

        results["cum_reward"] = self.cum_reward
        results["cum_regret"] = self.cum_regret
        results["cum_cross_entropy"] = self.cum_cross_entropy
        results["cum_ae"] = self.cum_ae

        # Specific to Bayesian and Kalman
        results["mu_bar"] = self.mu_bar
        results["tau_bar"] = self.tau_bar
        results["var_bar"] = self.var_bar
        results["std_bar"] = np.sqrt(self.var_bar)

        # Specific to Kalman only
        results["kalman_gain"] = self.kalman_gain

        results_mean = {}

        results_mean["reward"] = np.mean(self.reward, axis=0)
        results_mean["Q"] = np.mean(self.Q, axis=0)
        results_mean["Q_mean"] = np.mean(self.Q_mean, axis=0)
        results_mean["action"] = np.round(np.mean(self.action, axis=0))
        results_mean["action_percentage"] = np.mean(self.action_total, axis=0) / np.sum(np.mean(self.action_total, axis=0))
        results_mean["regret"] = np.mean(self.regret, axis=0)

        results_mean["P_action"] = np.mean(self.P_action, axis=0)
        results_mean["P"] = np.mean(self.P, axis=0)

        # just idea
        results_mean["regret_a"] = np.mean(self.regret_a, axis=0)
        results_mean["cum_regret_a"] = np.mean(self.cum_regret_a, axis=0)

        results_mean["cross_entropy"] = np.mean(self.cross_entropy, axis=0)
        results_mean["ae"] = np.mean(self.ae, axis=0)

        results_mean["cum_reward"] = np.mean(self.cum_reward, axis=0)
        results_mean["cum_regret"] = np.mean(self.cum_regret, axis=0)
        results_mean["cum_cross_entropy"] = np.mean(self.cum_cross_entropy, axis=0)
        results_mean["cum_ae"] = np.mean(self.cum_ae, axis=0)

        results_mean["Q_mean_weighted"] = np.mean(self.Q_mean_weighted, axis=0)

        # Specific to Bayesian and Kalman
        results_mean["mu_bar"] = np.mean(self.mu_bar, axis=0)
        results_mean["tau_bar"] = np.mean(self.tau_bar, axis=0)
        results_mean["var_bar"] = np.mean(self.var_bar, axis=0)
        results_mean["std_bar"] = np.sqrt(np.mean(self.var_bar, axis=0))

        # Specific to Kalman only
        results_mean["kalman_gain"] = np.mean(self.kalman_gain, axis=0)

        return results, results_mean