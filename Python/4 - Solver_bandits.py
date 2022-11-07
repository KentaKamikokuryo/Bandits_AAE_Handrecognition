import pandas as pd
from tabulate import tabulate
from typing import List

from Classes_data.Info import DB_info

from Classes_data.Data_RL import Data_Z_RL
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Info_RL import BanditInfo, SolverInfo
from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.SolverFactory import SolverFactory
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import Solver

from Classes_results.Results_RL import Results_bandits
from Classes_results.Ranked import Ranked
from Classes_results.Genetic import Genetic_algorithm

class Manager:

    bandits: List[Bandit]
    oracle: dict
    solver: Solver

    def __init__(self, data_Z_RL: Data_Z_RL, train_parameter: dict, solver_name: str):

        self.data_Z_RL = data_Z_RL
        self.train_parameter = train_parameter
        self.solver_name = solver_name

        self.bandit_info = BanditInfo()
        self.behavior_parameters = self.bandit_info.get_parameters(name="real")

        self.path_search_models_RL_DB_N = self.data_Z_RL.db_info.path_search_models_RL_DB_N

        if "boltzmann" in self.solver_name:
            self.path_figure = self.data_Z_RL.db_info.path_figure_bandit_DB_N
        elif "bayesian" in self.solver_name:
            self.path_figure = self.data_Z_RL.db_info.path_figure_bandit_bayesian_DB_N
        elif "kalman_filter" in self.solver_name:
            self.path_figure = self.data_Z_RL.db_info.path_figure_bandit_kalman_DB_N
        else:
            self.path_figure = None

    def set_interface(self, interface_dict):

        self.hyper_model_search = interface_dict["hyper_model_search"]
        self.train_final_model = interface_dict["train_final_model"]
        self.run_analysis = interface_dict["run_analysis"]
        self.save_model = interface_dict["save_model"]
        self.save_results = interface_dict["save_results"]
        self.hyper_model_search_type = interface_dict["hyper_model_search_type"]
        self.metric_type = interface_dict["metric_type"]

    def set_hyper_model_search(self):

        if self.hyper_model_search:

            self.ranked = Ranked(model_name=self.solver_name,
                                 search_type=self.hyper_model_search_type,
                                 path=self.path_search_models_RL_DB_N,
                                 metric_order="descending",
                                 metric_name="regret")

            self.hyperparameters_list, self.hyperparameters_choices = Hyperparamaters_bandits.generate_hyperparameters(solver_name=self.solver_name, display_info=True)

        else:

            self.ranked = Ranked(model_name=self.solver_name,
                                 search_type=self.hyper_model_search_type,
                                 path=self.path_search_models_RL_DB_N,
                                 metric_order="descending",
                                 metric_name="regret")

            self.ranked.load_ranked_metric()
            self.ranked.display_loaded_ranked_metric()
            self.ranked.load_best_hyperparameters()

            self.hyper_model_best = self.ranked.hyperparameter_best
            self.hyper_model_list_sorted = self.ranked.hyperparameters_list_sorted

        # search with genetic algorithm
        self.n_agent_generation = 10
        self.n_generation = 2

        arguments = {"n_generation": self.n_generation,
                     "n_agent_generation": self.n_agent_generation,
                     "nn_param_choices": self.hyperparameters_choices,
                     "display_info": True}

        self.GA = Genetic_algorithm(**arguments)

        selection_parameters = dict(name="tournament",
                                    metric_order="descending",
                                    mutate_chance=0.05,
                                    n_agent_fight=2,
                                    keep=0.2,
                                    keep_best=0.05)

        self.GA.set_selection_methods(selection_parameters=selection_parameters)

    def create_solver(self, hyper_model, bandits):

        solverFactory = SolverFactory()
        solver = solverFactory.create(bandits=bandits, n_episode=self.train_parameter["n_episode"], hyper_model=hyper_model)

        return solver

    def create_bandits(self):

        d_std = self.data_Z_RL.Z_test["d_std"]

        bandits_subjects = {}

        for k in d_std.keys():

            parameters = self.behavior_parameters.copy()
            parameters["mu_list"] = self.data_Z_RL.d_mu[k]
            parameters["std_list"] = self.data_Z_RL.d_std[k]

            bandit_manager = BanditsFactory(parameters=parameters, n_iteration=self.data_Z_RL.n_iteration, n_action=self.data_Z_RL.n_action)
            bandits = bandit_manager.create()
            bandits_subjects[k] = bandits

            self.results_bandits.save_behavior(parameters=parameters, subjects=k, bandits=bandits)

        return bandits_subjects

    def run_search(self, genetic: bool = False):

        self.results_bandits = Results_bandits(path=self.path_figure, solver_name=solver_name, metric_type=self.metric_type)
        
        # create bandits
        self.bandits_subjects = self.create_bandits()
        self.results_all = {}

        if genetic:  # TODO: organize it

            hyperparameters_gen = self.GA.generate_first_population()

            for gen in range(self.GA.n_generation):

                metrics_mean_list, metrics_sd_list, param_count = [], [], []

                print("Current generation sum up: DB_N " + str(data_Z_RL.DB_N) + " - " + str(self.GA.current_generation + 1) + "/" + str(self.GA.n_generation) + " generation - Model: " + data_Z_RL.model_ML_name)

                for i, hyper_model in enumerate(hyperparameters_gen):

                    print("Current information: DB_N " + str(data_Z_RL.DB_N) + " - " + str(self.GA.current_generation + 1) + "/" + str(self.GA.n_generation) + " - Agent " + str(i + 1) + "/" + str(self.GA.n_agent_generation))
                    df = pd.DataFrame.from_dict([hyper_model])
                    print(tabulate(df, headers='keys', tablefmt='psql'))

                    solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=hyper_model)
                    results_subject, results_mean_subjects, oracle = self.fit(hyper_model=hyper_model)
                    metric_mean, metric_std = self.results_bandits.evaluate(results_subjects=results_mean_subjects)

                    self.results_all[solver_name_hyper] = results_mean_subjects

                    self.ranked.add(hyper_model, metric_mean, metric_std, count_params=i, id=str(gen) + "_" + str(i))

                    metrics_mean_list.append(metric_mean)
                    metrics_sd_list.append(metric_std)
                    param_count.append(i)

                hyperparameters_gen = self.GA.generate_ith_generation(metrics_mean_list=metrics_mean_list, metrics_sd_list=metrics_sd_list)

        else:

            # Grid search
            for hyperparameter in self.hyperparameters_list:

                solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=hyperparameter)
                results_subjects, results_mean_subjects, oracle = self.fit(hyper_model=hyperparameter)  # Fit Artificial data solver, get results
                metric_mean, metric_std = self.results_bandits.evaluate(results_subjects=results_mean_subjects)  # Evaluate all subjects and return mean and std values
                self.results_all[solver_name_hyper] = results_mean_subjects
                self.ranked.add(hyperparameter=hyperparameter, metric_mean=metric_mean, metric_std=metric_std)  # add result got from previous

            # Gather results for all subjects
            self.results_bandits.plot_search_results(results_all=self.results_all, oracle=oracle, save_figure=True)  # TODO change oracle

        self.ranked.ranked()
        self.ranked.display_ranked_metric()
        self.ranked.save_ranked_metric()
        self.ranked.save_best_hyperparameter()

    def run_final(self):

        self.results_all = {}

        self.results_bandits = Results_bandits(path=self.path_figure, solver_name=solver_name, metric_type=self.metric_type)

        # Create bandits
        self.bandits_subjects = self.create_bandits()

        solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=self.hyper_model_best)
        results_subjects, results_mean_subjects, oracle = self.fit(hyper_model=self.hyper_model_best)
        metric_mean, metric_std = self.results_bandits.evaluate(results_subjects=results_subjects)  # Evaluate all subjects and return mean and std values
        self.results_all[solver_name_hyper] = results_subjects
        self.results_bandits.plot_final_results(results_all=self.results_all, oracle=oracle)

    def fit(self, hyper_model: dict):

        results = {}
        results_mean = {}

        for k in self.bandits_subjects.keys():

            self.bandits = self.bandits_subjects[k]
            self.solver = self.create_solver(hyper_model=hyper_model, bandits=self.bandits)

            results[k], results_mean[k], oracle = self.solver.run()

        return results, results_mean, oracle

    def run(self):

        if self.hyper_model_search:

            self.run_search(genetic=True)

            print("Run seach done ")

        elif self.train_final_model:

            self.run_final()

            print("Run final done ")

        elif self.run_analysis:

            pass

I = 0
Is = [0]

db_info = DB_info()
DB_Ns = [i for i in range(len(db_info.DB_N_configurations))]
DB_Ns = [0]

model_ML_name = "ICA"

DB_N = 0

for DB_N in DB_Ns:

    for I in Is:

        if I == 0:

            interface_dict = {"hyper_model_search": True,
                              "train_final_model": False,
                              "run_analysis": False,
                              "save_model": False,
                              "save_results": False,
                              "hyper_model_search_type": "full",
                              "metric_type": "regret"}

        elif I == 1:

            interface_dict = {"hyper_model_search": False,
                              "train_final_model": True,
                              "run_analysis": False,
                              "save_model": True,
                              "save_results": False,
                              "hyper_model_search_type": "full",
                              "metric_type": "behavior_knn"}

        elif I == 2:

            interface_dict = {"hyper_model_search": False,
                              "train_final_model": False,
                              "run_analysis": True,
                              "save_model": False,
                              "save_results": True,
                              "hyper_model_search_type": "full",
                              "metric_type": "behavior_knn"}

        else:

            interface_dict = {}

        db_info = DB_info()
        db_info.get_DB_info(DB_N=DB_N)

        data_Z_RL = Data_Z_RL(db_info=db_info, model_ML_name=model_ML_name)
        data_Z_RL.get_DB_N()

        solver_info = SolverInfo(n_action=data_Z_RL.n_action, n_iteration=data_Z_RL.n_iteration, n_episode=data_Z_RL.n_episode)
        train_parameter = solver_info.get_parameter()
        solver_names = solver_info.solver_names

        solver_names = [solver_names[4]]

        for solver_name in solver_names:

            manager = Manager(data_Z_RL=data_Z_RL, train_parameter=train_parameter, solver_name=solver_name)
            manager.set_interface(interface_dict)
            manager.set_hyper_model_search()
            manager.run()
