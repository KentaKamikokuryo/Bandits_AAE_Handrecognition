import numpy as np
import pandas as pd
import math, itertools, os
from typing import Dict
from tabulate import tabulate
from Classes_RL.SolverFactory import Models
np.set_printoptions(precision=4, suppress=True)

class Hyperparamaters_bandits():

    hyperparameters_choices: Dict
    hyperparameters_choices_best: Dict

    def __init__(self):

        pass

    @staticmethod
    def generate_hyperparameters(solver_name, display_info=True):

        # Generate dictionary to model search

        # ------- Boltzmann -------

        if solver_name == Models.Boltzmann_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           temperature=[1, 2, 3],
                                           solver_name=[solver_name])

        elif solver_name == Models.Boltzmann_UCB_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           temperature=[1, 2, 3],
                                           c=[1, 2, 3],
                                           solver_name=[solver_name])

        elif solver_name == Models.Boltzmann_non_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           alpha=[1.0],
                                           temperature=[1],
                                           solver_name=[solver_name])

        elif solver_name == Models.Boltzmann_UCB_non_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           alpha=[1.0],
                                           c=[2],
                                           temperature=[1],
                                           solver_name=[solver_name])

        # ------- Boltzmann - Sliding window -------

        elif solver_name == Models.Boltzmann_UCB_sliding:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           temperature=[1, 2, 3],
                                           c=[1, 2, 3],
                                           window=[5, 10],
                                           solver_name=[solver_name])

        # ------- Bayesian -------

        elif solver_name == Models.Bayesian_um_kv_softmax_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           initial_var=[1000],
                                           temperature=[2, 4, 9],
                                           solver_name=[solver_name])

        elif solver_name == Models.Bayesian_um_kv_softmax_non_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           initial_var=[1000],
                                           alpha=[0.1],
                                           temperature=[1, 2, 3],
                                           solver_name=[solver_name])

        elif solver_name == Models.Bayesian_um_uv_softmax_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           initial_var=[1000],
                                           alpha_g=[1, 2],
                                           beta_g=[1, 2],
                                           temperature=[1, 2],
                                           solver_name=[solver_name])

        elif solver_name == Models.Bayesian_um_uv_softmax_non_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           initial_var=[1000],
                                           alpha=[0.1],
                                           alpha_g=[1, 2, 3],
                                           beta_g=[1, 2, 3],
                                           temperature=[1, 2, 3],
                                           solver_name=[solver_name])

        # ------- Bayesian - Sliding window -------

        elif solver_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           initial_var=[1000],
                                           window=[5, 10],
                                           temperature=[1, 2, 3],
                                           solver_name=[solver_name])

        elif solver_name == Models.Bayesian_um_uv_softmax_sliding_non_stationary:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           initial_var=[1000],
                                           window=[5, 10],
                                           alpha_g=[1, 2],
                                           beta_g=[1, 2, 3],
                                           temperature=[1, 2, 3],
                                           solver_name=[solver_name])

        # ------- Kalman Filter -------

        elif solver_name == Models.Kalman_greedy:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           innovation_var=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
                                           noise_var=[0.01, 0.05, 0.1, 0.5, 1],
                                           solver_name=[solver_name])

        elif solver_name == Models.Kalman_e_greedy:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           epsilon=[0.1, 0.2, 0.3],
                                           innovation_var=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
                                           noise_var=[0.01, 0.05, 0.1, 0.5, 1],
                                           solver_name=[solver_name])

        elif solver_name == Models.Kalman_UCB:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           c=[1, 2, 3],
                                           innovation_var=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
                                           noise_var=[0.01, 0.05, 0.1, 0.5, 1],
                                           solver_name=[solver_name])

        elif solver_name == Models.Kalman_softmax:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           temperature=[1],
                                           innovation_var=[2],
                                           noise_var=[0.5],
                                           solver_name=[solver_name])

        elif solver_name == Models.Kalman_UCB_softmax:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           c=[1],
                                           temperature=[3],
                                           innovation_var=[0.05],
                                           noise_var=[0.05],
                                           solver_name=[solver_name])

        elif solver_name == Models.Kalman_Thompson_greedy:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           innovation_var=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
                                           noise_var=[0.01, 0.05, 0.1, 0.5, 1],
                                           solver_name=[solver_name])

        elif solver_name == Models.Kalman_Thompson_softmax:

            hyperparameters_choices = dict(optimistic_Q=[5],
                                           temperature=[1],
                                           innovation_var=[0.01],
                                           noise_var=[0.01],
                                           solver_name=[solver_name])

        # Zip function, get multiple kind lists - * Get the keys and values of the dictionary as a iterable
        keys, values = zip(*hyperparameters_choices.items())
        # Generate a direct product (Cartesian product) of multiple lists in Python
        hyperparameters_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if display_info:

            df = pd.DataFrame.from_dict(hyperparameters_all_combination)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return hyperparameters_all_combination, hyperparameters_choices

    @staticmethod
    def get_best_hyperparameters(solver_name, display_info=True):

        hyperparameters_choices_best = {}

        # Generate dictionary with best hyperparameters

        # ------- Boltzmann -------

        if solver_name == Models.Boltzmann_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                temperature=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Boltzmann_UCB_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                temperature=2,
                                                c=3,
                                                solver_name=solver_name)

        elif solver_name == Models.Boltzmann_non_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=5,
                                                alpha=0.1,
                                                temperature=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Boltzmann_UCB_non_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                alpha=0.1,
                                                c=3,
                                                temperature=2,
                                                solver_name=solver_name)

        # ------- Boltzmann - Sliding Window -------

        elif solver_name == Models.Boltzmann_UCB_sliding:

            hyperparameters_choices_best = dict(optimistic_Q=5,
                                                window=10,
                                                temperature=1,
                                                c=2,
                                                solver_name=solver_name)

        # ------- Bayesian -------

        elif solver_name == Models.Bayesian_um_kv_softmax_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                temperature=3,
                                                solver_name=solver_name)

        elif solver_name == Models.Bayesian_um_kv_softmax_non_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                initial_var=1000,
                                                temperature=3,
                                                alpha=0.1,
                                                solver_name=solver_name)

        elif solver_name == Models.Bayesian_um_uv_softmax_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                initial_variance=1000,
                                                temperature=3,
                                                alpha=0.1,
                                                alpha_g=5,
                                                beta_g=0.2,
                                                solver_name=solver_name)

        elif solver_name == Models.Bayesian_um_uv_softmax_non_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=4,
                                                temperature=3,
                                                alpha=0.5,
                                                alpha_g=5,
                                                beta_g=0.5,
                                                initial_var = 1000,
                                                solver_name=solver_name)

        # ------- Bayesian - Sliding window -------

        elif solver_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=1,
                                                initial_var=1000,
                                                window=10,
                                                temperature=4,
                                                is_stationary=False,
                                                solver_name=solver_name)

        elif solver_name == Models.Bayesian_um_uv_softmax_sliding_non_stationary:

            hyperparameters_choices_best = dict(optimistic_Q=1,
                                                initial_var=1000,
                                                window=5,
                                                is_stationary=False,
                                                alpha_g=1,
                                                beta_g=1,
                                                temperature=2,
                                                with_lambda=False,
                                                solver_name=solver_name)

        # ------- Kalman Filter -------

        elif solver_name == Models.Kalman_greedy:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                optimistic_Q=4,
                                                innovation_var=1,
                                                noise_var=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Kalman_e_greedy:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                optimistic_Q=4,
                                                epsilon=0.3,
                                                innovation_var=4,
                                                noise_var=4,
                                                solver_name=solver_name)

        elif solver_name == Models.Kalman_UCB:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                optimistic_Q=4,
                                                c=2,
                                                innovation_var=1,
                                                noise_var=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Kalman_softmax:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                optimistic_Q=4,
                                                temperature=2,
                                                innovation_var=1,
                                                noise_var=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Kalman_UCB_softmax:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                optimistic_Q=5,
                                                c=1,
                                                temperature=1,
                                                innovation_var=1,
                                                noise_var=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Kalman_Thompson_greedy:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                optimistic_Q=4,
                                                innovation_var=1,
                                                noise_var=1,
                                                solver_name=solver_name)

        elif solver_name == Models.Kalman_Thompson_softmax:

            hyperparameters_choices_best = dict(is_stationary=True,
                                                temperature=2,
                                                optimistic_Q=4,
                                                innovation_var=1,
                                                noise_var=1,
                                                solver_name=solver_name)

        if display_info:

            df = pd.DataFrame.from_dict([hyperparameters_choices_best])
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return hyperparameters_choices_best

    # @staticmethod
    # def get_hyperparameter_name(hyperparameter):
    #
    #     # print(hyperparameter)
    #
    #     solver_name = hyperparameter["solver_name"]
    #
    #     # ------- Boltzmann -------
    #
    #     if solver_name == Models.Boltzmann_stationary:
    #         name = "boltzmann: T " + str(hyperparameter["temperature"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Boltzmann_UCB_stationary:
    #         name = "boltzmann_UCB: T " + str(hyperparameter["temperature"]) + " - c " + str(hyperparameter["c"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Boltzmann_non_stationary:
    #         name = "boltzmann: T " + str(hyperparameter["temperature"]) + " - \u03B1: " + str(hyperparameter["alpha"]) + " - OQ: " + str(hyperparameter["optimistic_Q"]) + " - NS"
    #
    #     elif solver_name == Models.Boltzmann_UCB_non_stationary:
    #         name = "boltzmann_UCB: T " + str(hyperparameter["temperature"]) + " - c " + str(hyperparameter["c"]) + " - \u03B1: " + str(hyperparameter["alpha"]) + " - OQ: " + str(hyperparameter["optimistic_Q"]) +  " - NS"
    #
    #     # ------- Boltzmann - Sliding window -------
    #
    #     elif solver_name == Models.Boltzmann_UCB_sliding:
    #         name = "boltzmann_UCB_sliding: T " + str(hyperparameter["temperature"]) + " - c " + str(hyperparameter["c"]) + " - window:" + str(hyperparameter["window"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     # ------- Bayesian -------
    #
    #     elif solver_name == Models.Bayesian_um_kv_softmax_stationary:
    #         name = "bayesian_um_kv: T " + str(hyperparameter["temperature"])
    #
    #     elif solver_name == Models.Bayesian_um_uv_softmax_stationary:
    #         name = "bayesian_um_uv: - \u03B1: " + str(hyperparameter["alpha_g"]) + " - \u03B2: " + str(hyperparameter["beta_g"]) + " - T:" + str(hyperparameter["temperature"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Bayesian_um_kv_softmax_non_stationary:
    #         name = "bayesian_um_kv: - T:" + str(hyperparameter["temperature"])  + " - \u03B1: " + str(hyperparameter["alpha"]) + " - OQ: " + str(hyperparameter["optimistic_Q"]) + " - NS"
    #
    #     elif solver_name == Models.Bayesian_um_uv_softmax_non_stationary:
    #         name = "bayesian_um_uv: - \u03B1: " + str(hyperparameter["alpha_g"]) + " - \u03B2: " + str(hyperparameter["beta_g"]) + " - T:" + str(hyperparameter["temperature"]) + " - \u03B1: " + str(hyperparameter["alpha"]) + " - OQ: " + str(hyperparameter["optimistic_Q"]) + " - NS"
    #
    #     # ------- Bayesian - Sliding window -------
    #
    #     elif solver_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:
    #         name = "Bayesian_Sliding_um_kv: - T:" + str(hyperparameter["temperature"]) + " - window:" + str(hyperparameter["window"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Bayesian_um_uv_softmax_sliding_non_stationary:
    #         name = "Bayesian_Sliding_um_uv: - \u03B1: " + str(hyperparameter["alpha_g"]) + " - \u03B2: " + str(hyperparameter["beta_g"]) + " - T:" + str(hyperparameter["temperature"]) + " - window:" + str(hyperparameter["window"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     # ------- Kalman filter -------
    #
    #     elif solver_name == Models.Kalman_greedy:
    #         name = "kalman_greedy: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(hyperparameter["noise_var"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Kalman_UCB:
    #         name = "kalman_UCB: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(hyperparameter["noise_var"]) + " - c:" + str(hyperparameter["c"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Kalman_softmax:
    #         name = "kalman_softmax: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(hyperparameter["noise_var"]) + " - T:" + str(hyperparameter["temperature"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Kalman_UCB_softmax:
    #         name = "kalman_UCB_softmax: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(hyperparameter["noise_var"]) + " - c:" + str(hyperparameter["c"]) + " - T:" + str(hyperparameter["temperature"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Kalman_Thompson_greedy:
    #         name = "kalman_filter_Thompson_argmax: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(hyperparameter["noise_var"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     elif solver_name == Models.Kalman_Thompson_softmax:
    #         name = "kalman_filter_Thompson_softmax: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(hyperparameter["noise_var"]) + " - T:" + str(hyperparameter["temperature"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])
    #
    #     else:
    #         name = "None"
    #
    #     return name

    @staticmethod
    def get_hyperparameter_name(hyperparameter):

        # print(hyperparameter)

        solver_name = hyperparameter["solver_name"]

        # ------- Boltzmann -------

        if solver_name == Models.Boltzmann_stationary:
            name = 'boltzmann: T: ' + str(hyperparameter["temperature"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Boltzmann_UCB_stationary:
            name = "boltzmann_UCB: T: " + str(hyperparameter["temperature"]) + \
                   " - c: " + str(hyperparameter["c"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Boltzmann_non_stationary:
            name = "boltzmann: T: " + str(hyperparameter["temperature"]) + \
                   " - \u03B1: " + str(hyperparameter["alpha"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Boltzmann_UCB_non_stationary:
            name = "boltzmann_UCB: T: " + str(hyperparameter["temperature"]) + \
                   " - c: " + str(hyperparameter["c"]) + \
                   " - \u03B1: " + str(hyperparameter["alpha"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"])
            
        # ------- Boltzmann sliding UCB ----------
        
        elif solver_name == Models.Boltzmann_UCB_sliding:
            name = "boltzmann_UCB: T: " + str(hyperparameter["temperature"]) + \
                   " - c: " + str(hyperparameter["c"]) + \
                   " - window_range: " + str(hyperparameter["window"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"])

        # ------- Bayesian -------

        elif solver_name == Models.Bayesian_um_kv_softmax_stationary:
            name = "bayesian_UMKV: T: " + str(hyperparameter["temperature"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"]) + \
                   " - initial_var: " + str(hyperparameter["initial_var"])

        elif solver_name == Models.Bayesian_um_uv_softmax_stationary:
            name = "bayesian_UMUV: T: " + str(hyperparameter["temperature"]) + \
                   " - alpha_g: " + str(hyperparameter["alpha_g"]) + \
                   " - beta_g: " + str(hyperparameter["beta_g"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"]) + \
                   " - initial_var: " + str(hyperparameter["initial_var"])

        elif solver_name == Models.Bayesian_um_kv_softmax_non_stationary:
            name = "bayesian_UMKV: T: " + str(hyperparameter["temperature"]) + \
                   " - \u03B1: " + str(hyperparameter["alpha"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"]) + \
                   " - initial_var: " + str(hyperparameter["initial_var"])

        elif solver_name == Models.Bayesian_um_uv_softmax_non_stationary:
            name = "bayesian_UMUV: T: " + str(hyperparameter["temperature"]) + \
                   " - \u03B1: " + str(hyperparameter["alpha"]) + \
                   " - alpha_g: " + str(hyperparameter["alpha_g"]) + \
                   " - beta_g: " + str(hyperparameter["beta_g"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"]) + \
                   " - initial_var: " + str(hyperparameter["initial_var"])

        # ------- Sliding -------

        elif solver_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:

            name = "Sliding Gaussian: T: " + str(hyperparameter["temperature"]) +\
                   " - window size: " + str(hyperparameter["window"]) + \
                   " - optimistic_Q: " + str(hyperparameter["optimistic_Q"]) + \
                   " - initial_var: " + str(hyperparameter["initial_var"])

        # ------- Kalman filter -------

        elif solver_name == Models.Kalman_greedy:

            name = "kalman_greedy: - \u03c3_\u03be: " + str(
                hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(
                hyperparameter["noise_var"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])


        elif solver_name == Models.Kalman_UCB:

            name = "kalman_UCB: - \u03c3_\u03be: " + str(hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(
                hyperparameter["noise_var"]) + " - c:" + str(hyperparameter["c"]) + " - OQ: " + str(
                hyperparameter["optimistic_Q"])


        elif solver_name == Models.Kalman_softmax:

            name = "kalman_softmax: - \u03c3_\u03be: " + str(
                hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(
                hyperparameter["noise_var"]) + " - T:" + str(hyperparameter["temperature"]) + " - OQ: " + str(
                hyperparameter["optimistic_Q"])


        elif solver_name == Models.Kalman_UCB_softmax:

            name = "kalman_UCB_softmax: - \u03c3_\u03be: " + str(
                hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(
                hyperparameter["noise_var"]) + " - c:" + str(hyperparameter["c"]) + " - T:" + str(
                hyperparameter["temperature"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])


        elif solver_name == Models.Kalman_Thompson_greedy:

            name = "kalman_filter_Thompson_argmax: - \u03c3_\u03be: " + str(
                hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(
                hyperparameter["noise_var"]) + " - OQ: " + str(hyperparameter["optimistic_Q"])


        elif solver_name == Models.Kalman_Thompson_softmax:

            name = "kalman_filter_Thompson_softmax: - \u03c3_\u03be: " + str(
                hyperparameter["innovation_var"]) + " - \u03c3_\u03b5: " + str(
                hyperparameter["noise_var"]) + " - T:" + str(hyperparameter["temperature"]) + " - OQ: " + str(
                hyperparameter["optimistic_Q"])

        else:
            name = "None"

        return name

    @staticmethod
    def get_hyperparameter_name_latex(hyperparameter):

        # print(hyperparameter)

        solver_name = hyperparameter["solver_name"]

        # ------- Boltzmann -------

        if solver_name == Models.Boltzmann_stationary:
            name = r'boltzmann: $\tau$: ' + str(hyperparameter["temperature"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Boltzmann_UCB_stationary:
            name = r"boltzmann_UCB: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $c$: " + str(hyperparameter["c"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Boltzmann_non_stationary:
            name = r"boltzmann: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $\alpha$: " + str(hyperparameter["alpha"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Boltzmann_UCB_non_stationary:
            name = r"boltzmann UCB: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $c$: " + str(hyperparameter["c"]) + \
                   r" - $\alpha$: " + str(hyperparameter["alpha"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])
            
        # ------- Boltzmann - sliding UCB ----------
        
        elif solver_name == Models.Boltzmann_UCB_sliding:
            name = r"boltzmann S-UCB: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $c$:" + str(hyperparameter["c"]) + \
                   r" - $w$:" + str(hyperparameter["window"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        # ------- Bayesian -------

        elif solver_name == Models.Bayesian_um_kv_softmax_stationary:
            name = r"bayesian UMKV: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"]) + \
                   r" - $\sigma_0$: " + str(hyperparameter["initial_var"])

        elif solver_name == Models.Bayesian_um_uv_softmax_stationary:
            name = r"bayesian UMUV: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $\alpha_\gamma$: " + str(hyperparameter["alpha_g"]) + \
                   r" - $\beta_\gamma$: " + str(hyperparameter["beta_g"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"]) + \
                   r" - $\sigma_0$: " + str(hyperparameter["initial_var"])

        elif solver_name == Models.Bayesian_um_kv_softmax_non_stationary:
            name = r"bayesian UMKV: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $\alpha$: " + str(hyperparameter["alpha"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"]) + \
                   r" - $\sigma_0$: " + str(hyperparameter["initial_var"])

        elif solver_name == Models.Bayesian_um_uv_softmax_non_stationary:
            name = r"bayesian UMUV: $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $\alpha$: " + str(hyperparameter["alpha"]) + \
                   r" - $\alpha_\gamma$: " + str(hyperparameter["alpha_g"]) + \
                   r" - $\beta_\gamma$: " + str(hyperparameter["beta_g"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"]) + \
                   r" - $\sigma_0$: " + str(hyperparameter["initial_var"])

        # ------- Sliding window -------

        elif solver_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:

            name = r"Sliding Gaussian: $\tau$: " + str(hyperparameter["temperature"]) +\
                   r" - $window\ size$: " + str(hyperparameter["window"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"]) + \
                   r" - $\sigma_0$: " + str(hyperparameter["initial_var"])

        # ------- Kalman filter -------

        elif solver_name == Models.Kalman_greedy:
            name = r"kalman filter greedy: $\sigma_\epsilon^2$: " + str(hyperparameter["innovation_var"]) + \
                   r" - $\sigma_\xi^2$: " + str(hyperparameter["noise_var"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Kalman_UCB:
            name = r"kalman filter greedy UCB: $\sigma_\epsilon^2$: : " + str(hyperparameter["innovation_var"]) + \
                   r" - $\sigma_\xi^2$: " + str(hyperparameter["noise_var"]) + \
                   r" - $c$: " + str(hyperparameter["c"] + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"]))

        elif solver_name == Models.Kalman_softmax:
            name = r"kalman filter: $\sigma_\xi^2$: " + str(hyperparameter["innovation_var"]) + \
                   r" - $\sigma_\epsilon^2$: " + str(hyperparameter["noise_var"]) + \
                   r" - $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $optimistic Q$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Kalman_UCB_softmax:
            name = r"kalman filter UCB: $\sigma_\xi^2$: " + str(hyperparameter["innovation_var"]) + \
                   r" - $\sigma_\epsilon^2$: " + str(hyperparameter["noise_var"]) + \
                   r" - $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $c$: " + str(hyperparameter["c"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Kalman_Thompson_greedy:
            name = r"kalman filter greedy TS: $\sigma_\xi^2$: " + str(hyperparameter["innovation_var"]) + \
                   r" - $\sigma_\epsilon^2$: " + str(hyperparameter["noise_var"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])

        elif solver_name == Models.Kalman_Thompson_softmax:
            name = r"kalman filter TS: $\sigma_\xi^2$: " + str(hyperparameter["innovation_var"]) + \
                   r" - $\sigma_\epsilon^2$: " + str(hyperparameter["noise_var"]) + \
                   r" - $\tau$: " + str(hyperparameter["temperature"]) + \
                   r" - $Q_0$: " + str(hyperparameter["optimistic_Q"])
        else:
            name = "None"

        return name

    # @staticmethod
    # def result_name(models_enum_name):
    #
    #     name = ""
    #
    #     # Boltzmann
    #
    #     if models_enum_name == Models.Boltzmann_stationary:
    #         name = "Boltzmann-S"
    #
    #     elif models_enum_name == Models.Boltzmann_UCB_stationary:
    #         name = "Boltzmann-S UCB"
    #
    #     elif models_enum_name == Models.Boltzmann_non_stationary:
    #         name = "Boltzmann"
    #
    #     elif models_enum_name == Models.Boltzmann_UCB_non_stationary:
    #         name = "Boltzmann UCB"
    #
    #     # Bayesian
    #
    #     elif models_enum_name == Models.Bayesian_um_kv_softmax_stationary:
    #         name = "Bayesian-S UMKV"
    #
    #     elif models_enum_name == Models.Bayesian_um_kv_softmax_non_stationary:
    #         name = "Bayesian UMKV"
    #
    #     elif models_enum_name == Models.Bayesian_um_uv_softmax_stationary:
    #         name = "Bayesian-S UMUV"
    #
    #     elif models_enum_name == Models.Bayesian_um_uv_softmax_non_stationary:
    #         name = "Bayesian UMUV"
    #
    #     # Sliding
    #
    #     elif models_enum_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:
    #         name = "Sliding UMKV"
    #
    #     elif models_enum_name == Models.Bayesian_um_uv_softmax_sliding_non_stationary:
    #         name = "Sliding UMUV"
    #
    #     elif models_enum_name == Models.Boltzmann_UCB_sliding:
    #         name = "Boltzmann SUCB"
    #
    #     # Kalman
    #
    #     elif models_enum_name == Models.Kalman_greedy:
    #         name = "Kalman greedy"
    #
    #     elif models_enum_name == Models.Kalman_softmax:
    #         name = "Kalman"
    #
    #     elif models_enum_name == Models.Kalman_UCB:
    #         name = "Kalman greedy UCB"
    #
    #     elif models_enum_name == Models.Kalman_UCB_softmax:
    #         name = "Kalman UCB"
    #
    #     elif models_enum_name == Models.Kalman_Thompson_greedy:
    #         name = "Kalman greedy TS"
    #
    #     elif models_enum_name == Models.Kalman_Thompson_softmax:
    #         name = "Kalman TS"
    #
    #     elif models_enum_name == Models.Kalman_e_greedy:
    #         name = "Kalman e-greedy"
    #
    #     return name
