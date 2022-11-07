from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from Classes_RL.Bandit import Bandit
from Classes_RL.Plot_RL import PlotSolver, PlotResults, DedicaterName

class Results_RL:

    @staticmethod
    def plot_bandits(bandits: List[Bandit], path: str, DB_N: str, subject_name: str):

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        n_actions = len(bandits)

        for i in range(n_actions):

            ax.plot(bandits[i].mu_iterations, label="arm " + str(i))
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Expected value for each arm")

        ax.legend()

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
        figure_name = "Bandits_" + DB_N + "_" + subject_name
        PlotSolver.save_plot(fig=fig, path=path, figure_name=figure_name, close_figure=True)

    @staticmethod
    def get_metrics(results_mean: dict, metric_type: str):

        metric_list = []

        if metric_type == "regret":

            for k in results_mean.keys():
                metric_list.append(np.max(results_mean[k]["cum_regret"]))

        elif metric_type == "reward":

            for k in results_mean.keys():
                metric_list.append(np.max(results_mean[k]["cum_reward"]))

        elif metric_type == "cross_entropy":

            for k in results_mean.keys():
                metric_list.append(results_mean[k]["cum_cross_entropy"][-1])

        elif metric_type == "ae":

            for k in results_mean.keys():
                metric_list.append(results_mean[k]["cum_ae"][-1])

        metric_mean = np.mean(metric_list)
        metric_std = np.std(metric_list)

        return metric_mean, metric_std

    @staticmethod
    def plot_results_episode(results_all: dict, oracle_data_subjects: dict, DB_N: str, solver_name: str, path: str, metric: str, save_figure: bool = True):

        " Specific to plot only one episode "

        plot_results = PlotResults()

        # Just plot two subjects
        subjects = list(results_all.keys())[0:2]

        for s in subjects:

            results = results_all[s]
            oracle_data = oracle_data_subjects[s]

            if save_figure:
                plot_results.plot_episode(results=results, oracle_data=oracle_data, path=path,
                                          subject_name=str(s), solver_name=solver_name)

    @staticmethod
    def plot_results_episodes(results_mean_all: dict, oracle_data_subjects: dict, DB_N: str, solver_name: str, path: str, metric: str, save_figure: bool = True):

        " Specific to plot the mean results of all episodes "

        plot_results = PlotResults()

        # Just plot two subjects
        subjects = list(results_mean_all.keys())[0:2]

        for s in subjects:

            results_mean = results_mean_all[s]
            oracle_data = oracle_data_subjects[s]

            if save_figure:
                plot_results.plot_episodes(results_mean=results_mean, oracle_data=oracle_data, path=path,
                                           subject_name=str(s), solver_name=solver_name,
                                           metric=metric)

    @staticmethod
    def plot_results_comparison_solvers(results_mean_all, path: str, metric: str, solver_name: str):

        behaviors = list(results_mean_all.keys())
        subjects = list(results_mean_all[behaviors[0]].keys())
        solvers = list(results_mean_all[behaviors[0]][subjects[0]].keys())
        iterations = [5, 10, 30, 60]

        n_behaviors = len(behaviors)
        n_solvers = len(solvers)
        n_iterations = len(iterations)

        behaviors_name = ["X", "Y", "R", "XY", "XR", "YR", "XYR"]

        # Get a 3D matrix
        results = np.empty(shape=(n_behaviors + 1, n_solvers, n_iterations))

        for k, behavior in enumerate(behaviors):
            for i, solver in enumerate(solvers):
                for j, iteration in enumerate(iterations):
                    if metric == "cross_entropy":
                        metrics = [results_mean_all[behavior][s][solver]["cum_cross_entropy"][iteration-1] for s in subjects]
                    elif metric == "ae":
                        metrics = [results_mean_all[behavior][s][solver]["cum_ae"][iteration-1] for s in subjects]
                    results[k, i, j] = np.nanmean(metrics)

        results[k + 1, :, :] = np.nanmean(results[:-1, :, :], axis=0)

        figs = PlotSolver.plot_3D_heatmap(results=results, solvers_name=solvers, behaviors_name=behaviors_name + ["Mean"], iterations=iterations)

        for i, fig in enumerate(figs):
            figure_name = "Comparison_" + solver_name + "_" + str(i)
            PlotSolver.save_plot(fig=fig, path=path, figure_name=figure_name, close_figure=True)
