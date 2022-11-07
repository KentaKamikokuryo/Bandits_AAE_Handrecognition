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
from Classes_RL.Bandit import Bandit
from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.SolverFactory import SolverFactory
from Classes_RL.PlotFactory import PlotResultsFactory
from Classes_RL.Kalman_filter_solver import KalmanGreedy, KalmanSoftmax, KalmanUCB, KalmanUCBSoftmax, KalmanThompsonGreedy
np.set_printoptions(precision=4, suppress=True)
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from tqdm import tqdm
from scipy import optimize

def set_bandits_data(bandits: List[Bandit]) -> dict:
    # Get back data from bandits
    # Get max bandit
    bandit_max_list = []
    n_iteration = len(bandits[0].mu_iterations)
    for i in range(n_iteration):
        bandit_max = max([bandits[j].mu_iterations[i] for j in range(len(bandits))])
        bandit_max_list.append(bandit_max)

    bandits_data = {}
    bandits_data["Q_target"] = np.array(bandit_max_list)

    bandits_data["mu_iterations"] = [bandit.mu_iterations for bandit in bandits]
    bandits_data["std_iterations"] = [bandit.std_iterations for bandit in bandits]
    bandits_data["var_iterations"] = [bandit.var_iterations for bandit in bandits]
    bandits_data["tau_iterations"] = [bandit.tau_iterations for bandit in bandits]

    return bandits_data

# simple
cwd = os.getcwd()
# Get path one level above the root of the project
path_parent_project = os.path.abspath(os.path.join(cwd, os.pardir))
# Path to the data
path_test = path_parent_project + "\\Test\\"

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

bandits_data = set_bandits_data(bandits=bandits)

results = {}
results_episode, results_mean = solver.run()
results[solver_name]=results_mean

plotFactory = PlotResultsFactory()
mean_plot_strategy, episode_plot_strategy = plotFactory.create(solver_name=solver_name)
figure_name = solver_name
mean_plot_strategy.plot(results=results, path_folder_figure=path_test, figure_name=figure_name, bandits_data=bandits_data)

result = []
result.append(results[solver_name]["action"])
result.append(results[solver_name]["reward"])
result = tuple(result)

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

def softmax_choice_prob(m, gamma):
    return np.exp(gamma * m) / np.sum(np.exp(gamma * m), axis=0)

# best_choice - 0, max_r_iteration - 1
# innovation - 0, noise - 1, param1 - 2, param2 - 3

def kf_sm_negLogLik_just_one_param(par: np.array, result: tuple) -> float:

    gamma = np.exp(par[0])
    choice = result[0]
    reward = result[1]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=4, sigma0_sq=10000000, sigma_xi_sq=1, sigma_epsilon_sq=1)
    m = data[0]
    p = softmax_choice_prob(m=m, gamma=gamma)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))


    return neg_log_lik

def kf_sm_negLogLik(par: np.array, result: tuple) -> float:

    sigma_xi_sq = np.exp(par[0])
    sigma_epsilon_sq = np.exp(par[1])
    sigma0_sq = 10000000
    t = np.exp(par[2])
    mu0 = par[3]

    choice = result[0]
    reward = result[1]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=mu0, sigma0_sq=sigma0_sq, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)
    m = data[0]
    p = Utilities.softmax_temperature(x=m, t=t)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))

    return neg_log_lik

def kf_ucb_negLogLik(par: np.array, result: tuple) -> float:

    sigma_xi_sq = np.exp(par[0])
    sigma_epsilon_sq = np.exp(par[1])
    c = np.exp(par[2])

    choice = result[0]
    reward = result[1]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=1, sigma0_sq=10000000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)

    m = data[0]
    v = data[1]
    p = ucb(m=m, v=v, c=c, t=1, innovation_var=sigma_xi_sq)

    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = np.sum(np.log(pchoice[1:]))

    return -1 * neg_log_lik

def kf_smucb_negLogLik(par: np.array, result: tuple) -> float:

    sigma_xi_sq = np.exp(par[0])
    sigma_epsilon_sq = np.exp(par[1])
    t = np.exp(par[2])
    c = np.exp(par[3])

    choice = result[0]
    reward = result[1]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=1, sigma0_sq=10000000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)

    m = data[0]
    v = data[1]
    p = ucb(m=m, v=v, c=c, t=t, innovation_var=sigma_xi_sq)

    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))
    if neg_log_lik == np.Inf or neg_log_lik == np.nan:
        neg_log_lik = 10**300

    return neg_log_lik

def kf_thompson_negLogLik(par: np.array, result: tuple) -> float:

    sigma_xi_sq = np.exp(par[0])
    sigma_epsilon_sq = np.exp(par[1])

    choice = result[0]
    reward = result[1]

    data = kalman_filter(choice=choice, reward=reward, noption=4, mu0=1, sigma0_sq=10000000, sigma_xi_sq=sigma_xi_sq, sigma_epsilon_sq=sigma_epsilon_sq)
    m = data[0]
    v = data[1]
    p = thompson_choice_prob_sampling(m=m, v=v)
    pchoice = np.zeros(shape=len(choice))
    for i in range(1, len(pchoice)):
        pchoice[i] = p[int(choice[i]), i]

    neg_log_lik = -1 * np.sum(np.log(pchoice[1:]))

    return neg_log_lik


x0 = np.array([np.log10(1), np.log10(1), np.log2(2), 1])
bounds = [(0.00001, None), (0.00001, None), (0.001, 10), (0, 4)]

params_MLE = optimize.minimize(fun=kf_sm_negLogLik, x0=x0, args=(result, ),
                               method="l-bfgs-b", jac = None, bounds=bounds,
                               tol=None, callback=None,
                               options={'disp': None, 'maxls': 20, 'iprint': -1,
                                        'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000,
                                        'ftol': 2.220446049250313e-09, 'maxcor': 10,
                                        'maxfun': 15000})

# print(np.exp(params_MLE.x[0:2]), params_MLE.x[2:])
# print(np.exp(params_MLE.x[0:3]), params_MLE.x[3])
print("innovation variance : {0}".format(np.exp(params_MLE.x[0])))
print("noise variance : {0}".format(np.exp(params_MLE.x[1])))
print("tau : {0}".format(np.exp(params_MLE.x[2])))
print("initial value of Q value : {0}".format(params_MLE.x[3]))


