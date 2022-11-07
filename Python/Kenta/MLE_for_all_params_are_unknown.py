import numpy as np
import os, itertools
from typing import Tuple, List, Dict, Any
import numpy as np
import os, itertools
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from scipy.optimize import minimize
from Classes_RL.Utilities import Utilities
from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.SolverFactory import SolverFactory
from Classes_RL.Kalman_filter_solver import KalmanGreedy, KalmanSoftmax, KalmanUCB, KalmanUCBSoftmax, KalmanThompsonGreedy
np.set_printoptions(precision=4, suppress=True)
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from tqdm import tqdm
from scipy import optimize

bandits = []
n_actions = 4
n_iteration = 1000
n_episode = 1

train_parameter = {"n_action": n_actions, "n_episode": n_episode}
parameters = {"name": "RndWalk", "range": 10, "std": 1}
solver_name = ["kalman_filter_Greedy", "kalman_filter_UCB", "kalman_filter_Softmax", "kalman_filter_SoftmaxUCB", "kalman_filter_ThompsonSampling"]
solver_name = solver_name[2]

bandit_fac = BanditsFactory(parameters=parameters, n_iteration=n_iteration, n_action=n_actions)
bandits = bandit_fac.create()

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

results={}
results[solver_name] = solver.run()
result = results[solver_name]

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
                                           temperature=np.linspace(1,2,10).tolist(),
                                           innovation_var=np.linspace(1,5).tolist(),
                                           noise_var=np.linspace(1,5).tolist(),
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

def kalman_filter(choice, reward, noption, mu0, sigma0_sq, sigma_xi_sq, sigma_epsilon_sq):

    nt = len(choice)
    no = noption
    m = np.ones(shape=(no, nt+1)) * mu0
    v = np.ones(shape=(no, nt+1)) * sigma0_sq

    for i in range(1, nt):
        kalman_gain = np.zeros(no)
        kalman_gain[int(choice[i])] = (v[int(choice[i]), i] + sigma_xi_sq)/(v[int(choice[i]), i] + sigma_xi_sq + sigma_epsilon_sq)
        m[:, i+1] = m[:, i] + kalman_gain * (int(reward[i]) - m[:, i])
        v[:, i+1] = (1 - kalman_gain) * (v[:, i] + sigma_xi_sq)

    return m, v

def ucb(m, v, c, t, innovation_var):

    temp_Q = m + c * np.sqrt(v + innovation_var)  # Add 1 to avoid dividing by zero
    temp_Q = np.nan_to_num(np.array(temp_Q), neginf=0.00001, posinf=0.00001, nan=0.00001)

    P = Utilities.softmax_temperature(temp_Q, t=t)

    return P

def thompson_choice_prob_sampling(m, v):
    # m is a vector with prior predictive means
    # v is a covariance matrix for the prior predictive distributions
    # nsample is an integer
    # initialize a matrix for the choice probabilities

    no = m.shape[0]
    nt = m.shape[1]
    values = np.zeros(shape=(no, nt))


    for i in range(nt):
        for j in range(no):
            temp = np.random.normal(m[j, i], np.sqrt(1 / v[j, i]))  # use precision for sampling (Bad for static behavior since the variance become smaller, smaller ....)
            if temp < 0:  # We want positive value only
                temp = 0

            values[j, i] = temp

    P = Utilities.softmax_temperature(x=values, t=1)  # Can be interesting to have more exploration

    return P

def kf_sm_negLogLik(par: dict, result: dict) -> float:

    t = par["temperature"]
    sigma_xi_sq = par["innovation_var"]
    sigma_epsilon_sq = par["noise_var"]

    choice = result["best_choice"]
    reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)
    m = data[0]
    p = Utilities.softmax_temperature(x=m, t=t)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))

    return neg_log_lik

def kf_ucb_negLogLik(par: dict, result: dict) -> float:

    sigma_xi_sq = par["innovation_var"]
    sigma_epsilon_sq = par["noise_var"]
    c = par["c"]

    choice = result["best_choice"]
    reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)

    m = data[0]
    v = data[1]
    p = ucb(m=m, v=v, c=c, t=1, innovation_var=sigma_xi_sq)

    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = np.sum(np.log(pchoice[1:]))

    return -1 * neg_log_lik

def kf_smucb_negLogLik(par: dict, result: dict) -> float:

    t = par["temperature"]
    sigma_xi_sq = par["innovation_var"]
    sigma_epsilon_sq = par["noise_var"]
    c = par["c"]

    choice = result["best_choice"]
    reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)

    m = data[0]
    v = data[1]
    p = ucb(m=m, v=v, c=c, t=t, innovation_var=sigma_xi_sq)

    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))

    return neg_log_lik

def kf_thompson_negLogLik(par: dict, result: dict) -> float:

    sigma_xi_sq = par["innovation_var"]
    sigma_epsilon_sq = par["noise_var"]

    choice = result["best_choice"]
    reward = result["max_r_iteration"]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=1000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)
    m = data[0]
    v = data[1]
    p = thompson_choice_prob_sampling(m=m, v=v)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))

    return neg_log_lik


hyperparamaters_list = KalmanHyperparametersBandits.generateKalmanHyperparameters(solver_name=solver_name)

out = np.zeros(shape=len(hyperparamaters_list))
num_list = np.linspace(0,len(hyperparamaters_list), num=len(hyperparamaters_list))

for i, hyperparameter in enumerate(tqdm(hyperparamaters_list)):
    out[i] = kf_sm_negLogLik(par=hyperparameter, result=result)

plt.plot(num_list, out)

# x0 = [1,1,1]
# bound = [(0,10)(0,None)]
# prams_MLE = optimize.minimize(fun=kf_sm_negLogLik,x0,args=)






