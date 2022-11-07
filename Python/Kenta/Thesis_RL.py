from typing import List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from Classes_RL.Bandit import Bandit
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Info_RL import SolverInfo
from Classes_RL.Interfaces import Solver
from Classes_RL.Path_info_RL import PathInfoRL
from Classes_RL.SolverFactory import SolverFactory
from Classes_RL.Plot_RL import PlotSolver
from Classes_RL.Oracle import Oracle


class Ranked_MABP():

    hyperparameter_best: dict

    def __init__(self, behavior_name, solver_name, path_search_models, train_param, segmentation, metric_type = "loss"):

        self.behavior_name = behavior_name
        self.solver_name = solver_name
        self.path = path_search_models
        self.metric_type = metric_type
        self.n_iteration = train_param["n_iteration"]
        self.n_action = train_param["n_action"]

        # segmentation = [0.1, 0.5, 1]
        self.metric_seg = {}
        self.metric_seg_sorted = {}

        for seg in segmentation:
            self.metric_seg[seg] = []

        self.hyperparameters = []
        self.hyperparameters_sorted = []

        self.name = "behavior_" + str(self.behavior_name) +\
                    "_arms_" + str(self.n_action) + \
                    "_hyperparameter_search_RL_" + str(self.solver_name) + \
                    "_metric_" + str(self.metric_type)

    def add(self, hyperparameter, metric):

        self.hyperparameters.append(hyperparameter)

        for seg in self.metric_seg:
            if seg == 1:
                self.metric_seg[seg].append(metric[-1])
            else:
                self.metric_seg[seg].append(metric[int(self.n_iteration*seg)])

    def rank(self):

        idx_seg = {}
        n_hyparams = len(self.hyperparameters)
        all = np.zeros(n_hyparams)

        for seg in self.metric_seg:
            idx_seg[seg] = np.argsort(self.metric_seg[seg])

            for i in range(n_hyparams):
                all[idx_seg[seg][i]] += i

        idx = np.argsort(all.tolist())

        # sort hyperparameters and results as metrics
        self.hyperparameters_sorted = np.array(self.hyperparameters)[idx].tolist()
        for seg in self.metric_seg:
            self.metric_seg_sorted[seg] = np.array(self.metric_seg[seg])[idx].tolist()

        self.hyperparameter_best = self.hyperparameters_sorted[0]

    def display_ranked_metric(self):

        df = pd.DataFrame.from_dict(self.hyperparameters_sorted)

        for seg in self.metric_seg_sorted:
            df["cum_"+str(self.metric_type)+"_"+str(seg)] = self.metric_seg_sorted[seg]

        print(tabulate(df, headers="keys", tablefmt="psql"))

    def save_ranked_metric(self):

        df = pd.DataFrame.from_dict(self.hyperparameters_sorted)

        for seg in self.metric_seg_sorted:
            df["cum_" + str(self.metric_type) + "_" + str(seg)] = self.metric_seg_sorted[seg]

        name = "ranked_" + self.name + ".csv"
        df.to_csv(self.path + name)
        print('hyperparameters_list_sorted has been saved to: ' + self.path + name)

    def load_ranked_metric(self):

        name = "ranked_" + self.name + ".csv"
        self.hyperparameters_sorted = pd.read_csv(self.path + name)
        print('hyperparameters_list_sorted has been load from:  ' + self.path + name)

    def save_best(self):

        name = self.name + "_best.npy"
        np.save(self.path + name, self.hyperparameter_best)
        print('hyperparameter_best has been saved to: ' + self.path + name)

    def load_best(self):

        name = self.name + "_best.npy"
        self.hyperparameter_best = np.load(self.path + name, allow_pickle=True).item()
        print('hyperparameter_best has been load from: ' + self.path + name)

    def display_loaded_ranked_metric(self):

        df = pd.DataFrame.from_dict(self.hyperparameters_sorted)
        print(tabulate(df, headers='keys', tablefmt='psql'))


class Saved_results():

    def __init__(self):

        self.results_behavior = {}
        self.results_mean_behavior = {}
        self.oracle_behavior = {}

    def add_results(self, behavior_name, results, results_mean):

        self.results_behavior[behavior_name] = results
        self.results_mean_behavior[behavior_name] = results_mean

    def add_oracle(self, behavior, oracle):

        self.oracle_behavior[behavior] = oracle


class Manager():

    bandits: List[Bandit]
    ranked: Ranked_MABP
    oracle: dict
    solver: Solver

    def __init__(self, RL_info: dict, evaluation_info: dict, search_info: bool, bandits: List[Bandit]):

        self.RL_info = RL_info

        self.bandits = bandits
        self.behavior_name = self.bandits[0].behavior_strategy.name

        solver_info = SolverInfo(**RL_info)
        self.solver_names = solver_info.solver_names_thesis
        self.train_parameter = solver_info.get_parameter()

        self.evaluation_info = evaluation_info
        self.search_info = search_info

        # generate path for each behavior
        self.path_info = PathInfoRL()
        self.path_save = self.path_info.get_path_search(behavior_name=self.behavior_name, arm=RL_info["n_action"])

        oracle_manager = Oracle(bandits=self.bandits, n_episode=RL_info["n_episode"], n_iteration=RL_info["n_iteration"])
        self.oracle = oracle_manager.get_oracle_data(temperature=2)

    def run(self) -> Tuple[dict, dict]:

        results_solver = {}
        results_mean_solver = {}

        # self.solver_names = self.solver_names[6:7]

        for solver_name in self.solver_names:
            print("running while solver names : " + solver_name)

            arguments = {"behavior_name": self.behavior_name,
                         "solver_name": solver_name,
                         "path_search_models": self.path_save,
                         "train_param": self.train_parameter,
                         "metric_type": self.evaluation_info["metric_type"],
                         "segmentation": self.evaluation_info["segmentation"]}

            self.ranked = Ranked_MABP(**arguments)

            if self.search_info:

                results, results_mean = self.run_search(solver_name=solver_name)

            else:

                results, results_mean = self.run_best(solver_name=solver_name)

            results_solver[solver_name] = results
            results_mean_solver[solver_name] = results_mean

        return results_solver, results_mean_solver

    def run_search(self, solver_name) -> Tuple[dict, dict]:

        self.path_result = self.path_info.get_path_result(behavior_name=self.behavior_name,
                                                          arm=self.RL_info["n_action"],
                                                          solver_name=solver_name)

        hyperparameters_list, \
        hyperparameters_choices = Hyperparamaters_bandits.generate_hyperparameters(solver_name=solver_name,
                                                                                   display_info=True)

        results = {}
        results_mean = {}

        for i, hyperparameter in tqdm(enumerate(hyperparameters_list)):

            # for attaching name hyper model
            solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=hyperparameter)
            solver_name_hyper_latex = Hyperparamaters_bandits.get_hyperparameter_name_latex(hyperparameter=hyperparameter)
            solver = self.create_solver(hyperparameter=hyperparameter)
            # get results for each hyper-parameter
            results[solver_name_hyper], results_mean[solver_name_hyper] = solver.run()

            # if "boltzmann" in solver_name_hyper:
            #
            #     fig = PlotSolver.plot_Q(result=results_mean[solver_name_hyper],
            #                             model_name=solver_name_hyper_latex, oracle=oracle)
            #
            #     PlotSolver.save_plot_result(fig=fig, path_folder_figure=self.path_result,
            #                                 figure_name=solver_name+"_"+str(i), close_figure=True)
            #
            # else:
            #
            #     fig = PlotSolver.plot_upper_lower_bounds(result=results_mean[solver_name_hyper],
            #                                              model_name=solver_name_hyper_latex, oracle=oracle)
            #
            #     PlotSolver.save_plot_result(fig=fig, path_folder_figure=self.path_result,
            #                                 figure_name=solver_name+"_"+str(i), close_figure=True)

            # Divide the iteration into three parts
            if self.evaluation_info["metric_type"] == "loss":

                cum_cross_entropy = results_mean[solver_name_hyper]["cum_cross_entropy"]
                self.ranked.add(hyperparameter=hyperparameter, metric=cum_cross_entropy)

            elif self.evaluation_info["metric_type"] == "regret":
                cum_regret = results_mean[solver_name_hyper]["cum_regret"]
                self.ranked.add(hyperparameter=hyperparameter, metric=cum_regret)

        self.ranked.rank()
        self.ranked.display_ranked_metric()
        self.ranked.save_best()
        self.ranked.save_ranked_metric()

        return results, results_mean

    def run_best(self, solver_name) -> Tuple[dict, dict]:

        self.path_result = self.path_info.get_path_result(behavior_name=self.behavior_name,
                                                          arm=self.RL_info["n_action"],
                                                          solver_name=solver_name)

        self.ranked.load_best()
        self.ranked.load_ranked_metric()
        # self.ranked.display_ranked_metric()

        hyperparameter_best = self.ranked.hyperparameter_best
        hyper_model_list_sorted = self.ranked.hyperparameters_sorted

        solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=hyperparameter_best)
        solver_name_hyper_latex = Hyperparamaters_bandits.get_hyperparameter_name_latex(hyperparameter=hyperparameter_best)

        solver = self.create_solver(hyperparameter=hyperparameter_best)

        results, results_mean = solver.run()

        # if "boltzmann" in solver_name_hyper:
        #
        #     fig = PlotSolver.plot_Q(result=results_mean,
        #                             model_name=solver_name_hyper_latex, oracle=oracle)
        #
        #     PlotSolver.save_plot_result(fig=fig, path_folder_figure=self.path_result,
        #                                 figure_name="Best_" + solver_name, close_figure=True)
        #
        # else:
        #
        #     fig = PlotSolver.plot_upper_lower_bounds(result=results_mean,
        #                                              model_name=solver_name_hyper_latex, oracle=oracle)
        #
        #     PlotSolver.save_plot_result(fig=fig, path_folder_figure=self.path_result,
        #                                 figure_name="Best_"+solver_name, close_figure=True)

        return results, results_mean

    def create_solver(self, hyperparameter):

        solver_fac = SolverFactory()
        solver = solver_fac.create(bandits=self.bandits,
                                   n_episode=self.train_parameter["n_episode"],
                                   hyper_model=hyperparameter,
                                   oracle_data=self.oracle)

        return solver