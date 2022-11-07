import numpy as np
import os, itertools
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from scipy.optimize import minimize

from Classes_RL.PlotFactory import PlotResultsFactory
from Classes_RL.Kalman_filter_solver import KalmanGreedy, KalmanSoftmax, KalmanUCB, KalmanUCBSoftmax, KalmanThompsonGreedy
from Classes_RL.Bandit import BehaviorRndWalkStrategy
from Classes_RL.Bandit import Bandit
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits

class KalmanHyperparametersBandits():

    @staticmethod
    def generateKalmanHyperparameters(solver_name, display_info=False):

        if solver_name == "kalman_filter_Greedy":

            hyperparameters_choices = dict(is_stationary=[True],
                                           innovation_var=[1, 2, 3, 4],
                                           noise_var=[1, 2, 3, 4],
                                           solver_name=[solver_name])

        elif solver_name == "kalman_filter_UCB":

            hyperparameters_choices = dict(is_stationary=[True],
                                           c=[1, 2],
                                           innovation_var=[1, 2, 3, 4],
                                           noise_var=[1, 2, 3, 4],
                                           solver_name=[solver_name])

        elif solver_name == "kalman_filter_Softmax":

            hyperparameters_choices = dict(is_stationary=[True],
                                           temperature=np.linspace(1,2,100).tolist(),
                                           innovation_var=[1],
                                           noise_var=[1],
                                           solver_name=[solver_name])

        elif solver_name == "kalman_filter_SoftmaxUCB":

            hyperparameters_choices = dict(is_stationary=[True],
                                           c=[1, 2],
                                           temperature=np.linspace(1,2,100).tolist(),
                                           innovation_var=[1, 2, 3, 4],
                                           noise_var=[1, 2, 3, 4],
                                           solver_name=[solver_name])

        elif solver_name == "kalman_filter_ThompsonSampling":

            hyperparameters_choices = dict(is_stationary=[True],
                                           innovation_var=[1, 2, 3, 4],
                                           noise_var=[1, 2, 3, 4],
                                           solver_name=[solver_name])

        # Zip function, get multiple kind lists - * Get the keys and values of the dictionary as a iterable
        keys, values = zip(*hyperparameters_choices.items())
        # Generate a direct product (Cartesian product) of multiple lists in Python
        hyperparameters_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if display_info:

            df = pd.DataFrame.from_dict(hyperparameters_all_combination)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return hyperparameters_all_combination

np.set_printoptions(precision=4, suppress=True)

# simple
plt.figure()
cwd = os.getcwd()
# Get path one level above the root of the project
path_parent_project = os.path.abspath(os.path.join(cwd, os.pardir))
# Path to the data
path_test = path_parent_project + "\\Test\\"
if not (os.path.exists(path=path_test)):
    os.makedirs(path_test)

bandits = []
n_actions = 4
n_iteration = 1000
n_episode = 1
train_parameter = {"n_action": n_actions, "n_episode": n_episode}

for i in range(n_actions):
    parameters = {"name": "RndWalk", "range": 10, "std": 1}
    behavior = BehaviorRndWalkStrategy()
    behavior.set_parameters(parameters=parameters, n_iteration=n_iteration)
    behavior.execute()
    bandit = Bandit(behavior)
    bandits.append(bandit)

#####
arguments = {"bandits": bandits,
             "n_iteration": bandits[0].get_n_iteration(),
             "n_episode": train_parameter["n_episode"]}

factories = {"kalman_filter_Greedy": KalmanGreedy(**arguments),
             "kalman_filter_UCB": KalmanUCB(**arguments),
             "kalman_filter_Softmax": KalmanSoftmax(**arguments),
             "kalman_filter_SoftmaxUCB": KalmanUCBSoftmax(**arguments),
             "kalman_filter_ThompsonSampling": KalmanThompsonGreedy(**arguments)}

solver_name = ["kalman_filter_Greedy", "kalman_filter_UCB", "kalman_filter_Softmax", "kalman_filter_SoftmaxUCB", "kalman_filter_ThompsonSampling"]
solver_name = solver_name[2]
solver = factories[solver_name]

solver_name_hyper = Hyperparamaters_bandits.get_best_hyperparameters(solver_name=solver_name, display_info=True)
solver.set_hyperparameters(hyperparameters=solver_name_hyper)

results = {}
results_episode, results_mean = solver.run()
results[solver_name]=results_mean

# plotFactory = PlotResultsFactory()
# plot_strategy = plotFactory.create(solver_name=solver_name)
# figure_name = "test"
# plot_strategy.plot(results=results, path_folder_figure=path_test, figure_name=figure_name)

result = results[solver_name]

def kalman_filter(choice, reward, noption, mu0, sigma0_sq, sigma_xi_sq, sigma_epsilon_sq):

    nt = len(choice)
    no = noption
    m = np.ones(shape=(no, nt+1)) * mu0
    v = np.ones(shape=(no, nt+1)) * sigma0_sq

    for i in range(1, nt, 1):
        kalman_gain = np.zeros(no)
        kalman_gain[int(choice[i])] = (v[int(choice[i]), i] + sigma_xi_sq)/(v[int(choice[i]), i] + sigma_xi_sq + sigma_epsilon_sq)
        m[:, i+1] = m[:, i] + kalman_gain * (int(reward[i]) - m[:, i])
        v[:, i+1] = (1 - kalman_gain) * (v[:, i] + sigma_xi_sq)

    return m, v

def softmax_choice_prob(m, gamma):
    return np.exp(gamma * m) / np.sum(np.exp(gamma * m), axis=0)

def kf_sm_Lik(par, choice, reward) -> float:

    gamma = par
    # choice = result["best_choice"]
    # reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=1, sigma_epsilon_sq=1)
    m = data[0]
    p = softmax_choice_prob(m=m, gamma=gamma)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    lik = np.prod(pchoice[1:])

    return lik

def kf_sm_LogLik(par, choice, reward) -> float:

    gamma = par
    # choice = result["best_choice"]
    # reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=1, sigma_epsilon_sq=1)
    m = data[0]
    p = softmax_choice_prob(m=m, gamma=gamma)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    log_lik = np.sum(np.log(pchoice[1:]))

    return log_lik

def kf_sm_negLogLik(par, choice, reward) -> float:

    gamma = par
    # choice = result["best_choice"]
    # reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=1, sigma_epsilon_sq=1)
    m = data[0]
    p = softmax_choice_prob(m=m, gamma=gamma)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))

    return neg_log_lik

# finding = minimize(fun=kf_sm_negLogLik(par=))

choice = result["action"]
reward = result["reward"]

oute=kf_sm_Lik(par=0.5, choice=choice, reward=reward)

gamma = np.linspace(0,1,1000)
out = np.zeros(shape=len(gamma))
for i in range(0,len(gamma)):
    out[i]=kf_sm_negLogLik(par=gamma[i], choice=choice, reward=reward)
plt.plot(gamma, out)