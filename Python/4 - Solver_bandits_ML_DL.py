from typing import List, Tuple, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Classes_data.Data_ML import Data, Data_Z

from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Info_RL import BanditInfo, SolverInfo
from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.SolverFactory import SolverFactory
from Classes_RL.Bandit import Bandit
from Classes_RL.Oracle import Oracle
from Classes_RL.Interfaces import Solver
from Classes_RL.Plot_RL import PlotSolver, PlotResults
from Classes_RL.PlotFactory import PlotBehaviorFactory

from Classes_results.Ranked import Ranked
from Classes_results.Genetic import Genetic_algorithm
from Classes_data.Data_RL import Data_Z_RL

from Classes_data.Info import DB_info
from Classes_results.Results_RL import Results_RL

class Manager:

    bandits: List[Bandit]
    oracle: dict
    solver: Solver

    def __init__(self, DB_Ns: List, model_ML_name:str):

        self.DB_Ns = DB_Ns
        self.model_ML_name = model_ML_name

        solver_info = SolverInfo(n_action=None, n_iteration=None, n_episode=None)
        self.solver_names = solver_info.solver_names

        # TODO: this part is highly repeated in various code (shoudl be in a class in the next project)
        self.model_ML_names = ["PCA", "T_SNE", "ICA", "ISO", "LDA", "Laplacian"]
        self.model_DL_names = ["AE", "VAE", "UAAE", "SSAAE"]

        if  self.model_ML_name in self.model_DL_names:
            self.is_from_DL = True
        else:
            self.is_from_DL = False

    def set_interface(self, interface_dict):

        self.hyper_model_search = interface_dict["hyper_model_search"]
        self.train_final_model = interface_dict["train_final_model"]
        self.perform_analysis = interface_dict["perform_analysis"]
        self.save_model = interface_dict["save_model"]
        self.save_results = interface_dict["save_results"]
        self.hyper_model_search_type = interface_dict["hyper_model_search_type"]
        self.metric_type = interface_dict["metric_type"]

    def set_solver_hyper_search(self, solver_name):

        if self.hyper_model_search:

            self.ranked = Ranked(model_name=solver_name,
                                 search_type=self.hyper_model_search_type,
                                 path=self.path_search,
                                 metric_order="descending",
                                 metric_name=self.metric_type)

            self.hyperparameters_list, self.hyperparameters_choices = Hyperparamaters_bandits.generate_hyperparameters(solver_name=solver_name, display_info=True)

        else:

            self.ranked = Ranked(model_name=solver_name,
                                 search_type=self.hyper_model_search_type,
                                 path=self.path_search,
                                 metric_order="descending",
                                 metric_name=self.metric_type)

            self.ranked.load_ranked_metric()
            self.ranked.display_loaded_ranked_metric()
            self.ranked.load_best_hyperparameters()

            self.hyper_model_best = self.ranked.hyperparameter_best
            self.hyper_model_list_sorted = self.ranked.hyperparameters_list_sorted

    def set_behavior(self, DB_N: str):

        self.db_info = DB_info()
        self.db_info.get_DB_info(DB_N=DB_N)

        self.data = Data(db_info=self.db_info)

        self.data_Z = Data_Z(self.data, model_name=self.model_ML_name)

        if self.is_from_DL:
            self.path_data = self.db_info.path_results_DL_DB_N
        else:
            self.path_data = self.db_info.path_results_ML_DB_N

        print("ML/DL data will be loaded from " + self.path_data)

        if self.hyper_model_search:

            self.path_search = self.db_info.path_search_models_RL_DB_N + self.model_ML_name + "\\"
            print("Search results will be saved at " + self.path_search)

            if not (os.path.exists(self.path_search)):
                os.makedirs(self.path_search)

        else:

            # Search only from DB_0 and UAAE
            # TODO: here Kenta
            db_info = DB_info()
            db_info.get_DB_info(DB_N=0)
            self.path_search = db_info.path_search_models_RL_DB_N + 'UAAE' + "\\"

            self.path_figure = self.db_info.path_figure_RL_DB_N + self.model_ML_name + "\\"
            # self.path_search = self.db_info.path_search_models_RL_DB_N + self.model_ML_name + "\\"
            self.path_results = self.db_info.path_results_RL_DB_N + self.model_ML_name + "\\"
            self.path_comparison = self.db_info.path_RL + "Comparison\\" + self.model_ML_name + "\\"

            print("Search results will be loaded from " + self.path_search)
            print("Results will be saved at " + self.path_search)
            print("Figures will be saved at " + self.path_figure)
            print("Comparison figure will be saved at " + self.path_comparison)

            if not (os.path.exists(self.path_figure)):
                os.makedirs(self.path_figure)
            if not (os.path.exists(self.path_results)):
                os.makedirs(self.path_results)
            if not (os.path.exists(self.path_comparison)):
                os.makedirs(self.path_comparison)

    def create_solver(self, hyper_model, bandits, oracle_data):

        solverFactory = SolverFactory()
        solver = solverFactory.create(bandits=bandits, oracle_data=oracle_data, n_episode=oracle_data["n_episode"], hyper_model=hyper_model)

        return solver

    def create_bandits(self, DB_N):

        # Load data from ML/DL
        Z_fit_trajectories, Z_transform_trajectories, Z_centroids_dict = self.data_Z.load_latent_analysis(path=self.path_data, is_standardized=False)
        Z_fit_trajectories_std, Z_transform_trajectories_std, Z_centroids_dict_std = self.data_Z.load_latent_analysis(path=self.path_data, is_standardized=True)

        # Z_transform_trajectories_std keys level
            # K-fold
                # std or not
                    # Subjects
                        # Labels (arms)

        subjects = list(Z_transform_trajectories_std[0]["d_std"].keys())
        self.n_subjects = len(subjects)

        self.n_episode = 10

        bandit_info = BanditInfo()
        bandit_parameters = bandit_info.get_parameters(name="real")
        print("bandit_parameters: " + str(bandit_parameters))

        bandits_subjects = {}
        oracle_data_subjects = {}

        for s in subjects:

            labels = list(Z_transform_trajectories_std[0]["d_std"][s].keys())
            self.n_action = len(labels)

            d_std = np.concatenate([Z_transform_trajectories_std[0]["d_std"][s][l][np.newaxis, :] for l in labels])
            n_iteration = d_std.shape[1]

            # for l in labels:
            #     plt.plot(Z_transform_trajectories_std[0]["d_std"][s][l], label=l)
            #
            # plt.legend()

            solver_info = SolverInfo(n_action=self.n_action, n_iteration=n_iteration, n_episode=self.n_episode)

            train_parameter = solver_info.get_parameter()

            bandit_parameters["mu_list"] = d_std
            bandit_parameters["std_list"] = np.ones(d_std.shape) * 0.2

            subject_name = "Artificial_" + str(s)
            bandit_manager = BanditsFactory(parameters=bandit_parameters, n_iteration=train_parameter["n_iteration"], n_action=train_parameter["n_action"])
            bandits = bandit_manager.create()
            bandits_subjects[subject_name] = bandits

            oracle = Oracle(bandits, n_iteration=train_parameter["n_iteration"], n_episode=train_parameter["n_episode"])
            oracle_data = oracle.get_oracle_data()
            oracle_data_subjects[subject_name] = oracle_data

            if not self.hyper_model_search:

                Results_RL.plot_bandits(bandits=bandits, DB_N=str(DB_N), subject_name=subject_name, path=self.path_figure)

        return bandits_subjects, oracle_data_subjects

    def fit(self, bandits_subjects: dict, oracle_data_subjects: dict, hyper_model: dict):

        results = {}
        results_mean = {}

        for s in bandits_subjects.keys():

            oracle_data = oracle_data_subjects[s]
            bandits = bandits_subjects[s]

            solver = self.create_solver(hyper_model=hyper_model,
                                        bandits=bandits,
                                        oracle_data=oracle_data)

            results[s], results_mean[s] = solver.run()

        return results, results_mean

    def switch_keys(self, results: dict):

        # Change order of the last two keys of dictionary
        # Usually, key (Solver_hyper / subject) become (subject / Solver_hyper)
        # We end up with a dictionary with key levels such as: subject / Solver_hyper / results

        subjects = list(results[list(results.keys())[0]].keys())

        results_2 = {}

        for k in subjects:
            results_2[k] = {}
            for k2 in results.keys():
                results_2[k][k2] = {}
                for k3 in results[k2][k].keys():
                    results_2[k][k2][k3] = results[k2][k][k3]

        return results_2

    def run_search(self):

        for DB_N in self.DB_Ns:

            self.set_behavior(DB_N=DB_N)

            bandits_subjects, oracle_data_subjects = self.create_bandits(DB_N=DB_N)

            for solver_name in self.solver_names:

                self.set_solver_hyper_search(solver_name)

                results_search = {}
                results_mean_search = {}

                id = 0

                for hyperparameter in self.hyperparameters_list:

                    solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=hyperparameter)

                    results, results_mean = self.fit(bandits_subjects=bandits_subjects,
                                                     oracle_data_subjects=oracle_data_subjects,
                                                     hyper_model=hyperparameter)

                    metric_mean, metric_std = Results_RL.get_metrics(results_mean=results_mean, metric_type=self.metric_type)

                    self.ranked.add(hyperparameter=hyperparameter, metric_mean=metric_mean, metric_std=metric_std, id=str(id))

                    results_search[solver_name_hyper] = results
                    results_mean_search[solver_name_hyper] = results_mean

                    id += 1

                results_search = self.switch_keys(results=results_search)
                results_mean_search = self.switch_keys(results=results_mean_search)

                # # Gather results for all subjects
                # Results_RL.plot_results_episodes(results_mean_all=results_mean_search,
                #                                                oracle_data_subjects=oracle_data_subjects,
                #                                                save_figure=True, DB_N=DB_N,
                #                                                solver_name=solver_name, metric=self.metric_type,
                #                                                path=self.path_figure)

                self.ranked.ranked()
                self.ranked.display_ranked_metric()
                self.ranked.save_ranked_metric()
                self.ranked.save_best_hyperparameter()

                # self.results_all[DB_N][solver_name] = results_search
                # self.results_mean_all[DB_N][solver_name] = results_mean_search

                print("Run search done on " + str(DB_N) + " with solver " + solver_name)

            print("Bandit search done on DB_N " + str(DB_N))

    def run_final(self):

        for DB_N in self.DB_Ns:

            self.set_behavior(DB_N=DB_N)

            bandits_subjects, oracle_data_subjects = self.create_bandits(DB_N=DB_N)

            for solver_name in self.solver_names:

                self.set_solver_hyper_search(solver_name)

                results, results_mean = self.fit(bandits_subjects=bandits_subjects, oracle_data_subjects=oracle_data_subjects,
                                                 hyper_model=self.hyper_model_best)

                # Save results
                name = solver_name + "_results.npy"
                np.save(self.path_results + name, results, allow_pickle=True)

                name = solver_name + "_results_mean.npy"
                np.save(self.path_results + name, results_mean, allow_pickle=True)

            name = "oracle_data_subjects.npy"
            np.save(self.path_results + name, oracle_data_subjects, allow_pickle=True)

            name = "bandits_subjects.npy"
            np.save(self.path_results + name, bandits_subjects, allow_pickle=True)

    def run_analysis(self):

        results_all = {}
        results_mean_all = {}

        results_boltzmann_all = {}
        results_bayesian_all = {}
        results_kalman_all = {}

        results_boltzmann_mean_all = {}
        results_bayesian_mean_all = {}
        results_kalman_mean_all = {}

        for DB_N in self.DB_Ns:

            results_all[DB_N] = {}
            results_mean_all[DB_N] = {}

            results_boltzmann_all[DB_N] = {}
            results_bayesian_all[DB_N] = {}
            results_kalman_all[DB_N] = {}

            results_boltzmann_mean_all[DB_N] = {}
            results_bayesian_mean_all[DB_N] = {}
            results_kalman_mean_all[DB_N] = {}

            self.set_behavior(DB_N=DB_N)

            ranked_solvers = Ranked(model_name="Comparison_solvers",
                                    search_type=self.hyper_model_search_type, path=self.path_results,
                                    metric_order="descending", metric_name="ae")

            name = "oracle_data_subjects.npy"
            oracle_data_subjects = np.load(self.path_results + name, allow_pickle=True).item()

            name = "bandits_subjects.npy"
            bandits_subjects = np.load(self.path_results + name, allow_pickle=True).item()

            results_solvers = {}
            results_mean_solvers = {}

            id = 0

            for solver_name in self.solver_names:

                self.set_solver_hyper_search(solver_name)
                solver_name_hyper = Hyperparamaters_bandits.get_hyperparameter_name(hyperparameter=self.hyper_model_best)

                name = solver_name + "_results.npy"
                results = np.load(self.path_results + name, allow_pickle=True).item()

                name = solver_name + "_results_mean.npy"
                results_mean = np.load(self.path_results + name, allow_pickle=True).item()

                results_solvers[solver_name] = results
                results_mean_solvers[solver_name] = results_mean

                metric_mean, metric_std = Results_RL.get_metrics(results_mean=results_mean, metric_type=self.metric_type)

                hyperparameter = {"Name": solver_name, "Hyper_name": solver_name_hyper}

                ranked_solvers.add(hyperparameter=hyperparameter, metric_mean=metric_mean, metric_std=metric_std, id=str(id))  # add result got from previous

                id += 1

            ranked_solvers.ranked()
            ranked_solvers.display_ranked_metric()
            ranked_solvers.save_ranked_metric()
            ranked_solvers.save_best_hyperparameter()

            results_all[DB_N] = self.switch_keys(results=results_solvers)
            results_mean_all[DB_N] = self.switch_keys(results=results_mean_solvers)

            # Separate Boltzmann, Bayesian and Kalman
            for s in results_mean_all[DB_N].keys():

                results_boltzmann_all[DB_N][s] = {}
                results_bayesian_all[DB_N][s] = {}
                results_kalman_all[DB_N][s] = {}

                results_boltzmann_mean_all[DB_N][s] = {}
                results_bayesian_mean_all[DB_N][s] = {}
                results_kalman_mean_all[DB_N][s] = {}

                for k in results_mean_all[DB_N][s].keys():
                    if "boltzmann" in k:
                        results_boltzmann_all[DB_N][s][k] = results_all[DB_N][s][k]
                        results_boltzmann_mean_all[DB_N][s][k] = results_mean_all[DB_N][s][k]
                    elif "bayesian" in k:
                        results_bayesian_all[DB_N][s][k] = results_all[DB_N][s][k]
                        results_bayesian_mean_all[DB_N][s][k] = results_mean_all[DB_N][s][k]
                    elif "kalman" in k:
                        results_kalman_all[DB_N][s][k] = results_all[DB_N][s][k]
                        results_kalman_mean_all[DB_N][s][k] = results_mean_all[DB_N][s][k]

            # Plot one episode results
            Results_RL.plot_results_episode(results_all=results_boltzmann_all[DB_N],
                                            oracle_data_subjects=oracle_data_subjects,
                                            save_figure=True,
                                            DB_N=str(DB_N),
                                            solver_name="boltzmann", metric=self.metric_type,
                                            path=self.path_figure)

            Results_RL.plot_results_episode(results_all=results_bayesian_all[DB_N],
                                            oracle_data_subjects=oracle_data_subjects,
                                            save_figure=True,
                                            DB_N=str(DB_N),
                                            solver_name="bayesian", metric=self.metric_type,
                                            path=self.path_figure)

            Results_RL.plot_results_episode(results_all=results_kalman_all[DB_N],
                                            oracle_data_subjects=oracle_data_subjects,
                                            save_figure=True,
                                            DB_N=str(DB_N),
                                            solver_name="kalman", metric=self.metric_type,
                                            path=self.path_figure)

            Results_RL.plot_results_episode(results_all=results_all[DB_N],
                                            oracle_data_subjects=oracle_data_subjects,
                                            save_figure=True,
                                            DB_N=str(DB_N),
                                            solver_name="all", metric=self.metric_type,
                                            path=self.path_figure)



            # Plot mean results of all episodes
            Results_RL.plot_results_episodes(results_mean_all=results_boltzmann_mean_all[DB_N],
                                           oracle_data_subjects=oracle_data_subjects,
                                           save_figure=True,
                                           DB_N=str(DB_N),
                                           solver_name="boltzmann", metric=self.metric_type,
                                           path=self.path_figure)

            Results_RL.plot_results_episodes(results_mean_all=results_bayesian_mean_all[DB_N],
                                           oracle_data_subjects=oracle_data_subjects,
                                           save_figure=True,
                                           DB_N=str(DB_N),
                                           solver_name="bayesian", metric=self.metric_type,
                                           path=self.path_figure)

            Results_RL.plot_results_episodes(results_mean_all=results_kalman_mean_all[DB_N],
                                           oracle_data_subjects=oracle_data_subjects,
                                           save_figure=True,
                                           DB_N=str(DB_N),
                                           solver_name="kalman", metric=self.metric_type,
                                           path=self.path_figure)

            Results_RL.plot_results_episodes(results_mean_all=results_mean_all[DB_N],
                                             oracle_data_subjects=oracle_data_subjects,
                                             save_figure=True,
                                             DB_N=str(DB_N),
                                             solver_name="all", metric=self.metric_type,
                                             path=self.path_figure)

        Results_RL.plot_results_comparison_solvers(results_mean_all=results_boltzmann_mean_all, solver_name="boltzmann", metric=self.metric_type, path=self.path_comparison)
        Results_RL.plot_results_comparison_solvers(results_mean_all=results_bayesian_mean_all, solver_name="bayesian", metric=self.metric_type, path=self.path_comparison)
        Results_RL.plot_results_comparison_solvers(results_mean_all=results_kalman_mean_all, solver_name="kalman", metric=self.metric_type, path=self.path_comparison)
        # Results_RL.plot_results_comparison_solvers(results_mean_all=results_mean_all, solver_name="all", metric=self.metric_type, path=self.path_comparison)

        results_boltzmann_kalman_mean_all = {}

        for DB_N in DB_Ns:
            results_boltzmann_kalman_mean_all[DB_N] = {}
            for s in results_boltzmann_mean_all[DB_N].keys():
                results_boltzmann_kalman_mean_all[DB_N][s] = {}
                results_boltzmann_kalman_mean_all[DB_N][s]['boltzmann_UCB_non_stationary'] = results_boltzmann_mean_all[DB_N][s]['boltzmann_UCB_non_stationary']
                results_boltzmann_kalman_mean_all[DB_N][s]['kalman_filter_Thompson_softmax'] = results_kalman_mean_all[DB_N][s]['kalman_filter_Thompson_softmax']

        Results_RL.plot_results_comparison_solvers(results_mean_all=results_boltzmann_kalman_mean_all, solver_name="boltzmann_kalman", metric=self.metric_type, path=self.path_comparison)

    def run(self):

        if self.hyper_model_search:

            self.run_search()

        elif self.train_final_model:

            self.run_final()

            print("Run final done ")

        elif self.perform_analysis:

            self.run_analysis()

I = 0
Is = [0, 1, 2]

sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.2)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

for I in Is:

    if I == 0:

        db_info = DB_info()
        DB_Ns = [0]

        model_ML_names = ["UAAE"]

    else:

        db_info = DB_info()
        DB_Ns = [i for i in range(len(db_info.DB_N_configurations))]

        model_ML_names = ["UAAE", "SSAAE"]

    if I == 0:

        interface_dict = {"hyper_model_search": True,
                          "train_final_model": False,
                          "perform_analysis": False,
                          "save_model": False,
                          "save_results": False,
                          "hyper_model_search_type": "full",
                          "metric_type": "ae"}

    elif I == 1:

        interface_dict = {"hyper_model_search": False,
                          "train_final_model": True,
                          "perform_analysis": False,
                          "save_model": True,
                          "save_results": True,
                          "hyper_model_search_type": "full",
                          "metric_type": "ae"}

    elif I == 2:

        interface_dict = {"hyper_model_search": False,
                          "train_final_model": False,
                          "perform_analysis": True,
                          "save_model": True,
                          "save_results": True,
                          "hyper_model_search_type": "full",
                          "metric_type": "ae"}

    else:

        interface_dict = {}

    for model_ML_name in model_ML_names:

        manager = Manager(DB_Ns=DB_Ns, model_ML_name=model_ML_name)
        manager.set_interface(interface_dict)
        manager.run()

