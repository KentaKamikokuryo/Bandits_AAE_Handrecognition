import numpy as np
from typing import Tuple, List, Dict, Any
from Classes_RL.Bandit import Bandit
from Classes_RL.Utilities import Utilities


class Oracle:

    def __init__(self, bandits: List[Bandit], n_iteration, n_episode):

        self.bandits = bandits
        self.a = len(bandits)
        self.A = np.arange(self.a)

        self.n_iteration = n_iteration
        self.n_episode = n_episode
        self.n_action = len(bandits)

        self.mu_known = np.concatenate([np.array(self.bandits[i].mu_iterations)[:, np.newaxis] for i in range(len(self.bandits))], axis=1)
        self.var_known = np.concatenate([np.array(self.bandits[i].var_iterations)[:, np.newaxis] for i in range(len(self.bandits))], axis=1)
        self.std_known = np.concatenate([np.array(self.bandits[i].std_iterations)[:, np.newaxis] for i in range(len(self.bandits))], axis=1)
        self.tau_known = np.concatenate([np.array(self.bandits[i].tau_iterations)[:, np.newaxis] for i in range(len(self.bandits))], axis=1)

    def get_oracle_data(self, temperature: int = 2):

        # Get oracle data. The oracle is all knowing, it knows everything about the bandits and the action that must be taken
        best_action = np.zeros(shape=(self.n_iteration), dtype=np.int)
        max_reward = np.zeros(shape=(self.n_iteration))

        # Get the reward for the current action and best action
        for k in range(self.n_iteration):
            best_action[k] = int(np.argmax([self.bandits[i].mu_iterations[k] for i in range(len(self.bandits))]))
            max_reward[k] = self.bandits[best_action[k]].get_max_reward(iteration=k)

        bandit_max_list = []
        for k in range(self.n_iteration):
            bandit_max = max([self.bandits[i].mu_iterations[k] for i in range(len(self.bandits))])
            bandit_max_list.append(bandit_max)

        bandit_Q_arm_list = []
        for k in range(self.n_iteration):
            Q = [self.bandits[i].mu_iterations[k] for i in range(len(self.bandits))]
            bandit_Q_arm_list.append(Q)

        oracle = {}

        oracle["n_iteration"] = self.n_iteration
        oracle["n_episode"] = self.n_episode
        oracle["n_action"] = self.n_action

        oracle["Q"] = np.array(bandit_Q_arm_list)
        oracle["Q_max"] = np.array(bandit_max_list)
        oracle["best_action"] = np.array(best_action)
        oracle["max_reward"] = np.array(max_reward)

        oracle["mu_known"] = self.mu_known
        oracle["var_known"] = self.var_known
        oracle["std_known"] = self.std_known
        oracle["tau_known"] = self.tau_known

        P = np.zeros(shape=(self.n_iteration, self.n_action))

        # Probability distribution of taking action through iteration
        for k in range(self.n_iteration):
            mu = [bandit.mu_iterations[k] for bandit in self.bandits]
            P[k] = Utilities.softmax_temperature(np.array(mu), t=temperature)

        oracle["P"] = np.array(P)  # Distribution at each iteration k
        oracle["P_action"] = np.array(P)  # Distribution at each iteration k

        temp = np.mean(oracle["P"], axis=0) * self.n_iteration
        oracle["action_percentage"] = temp / np.sum(temp)

        # Add Q mean based on probability
        Q_mean_weighted_list = []

        for k in range(self.n_iteration):

            Q_weighted = np.average(oracle["Q"][k], weights=oracle["P"][k])
            Q_mean_weighted_list.append(Q_weighted)

        oracle["Q_mean_weighted"] = np.array(Q_mean_weighted_list)

        return oracle
