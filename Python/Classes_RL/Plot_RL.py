import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from Classes_RL.Utilities import Utilities
import scipy.stats as stats
import imageio
from typing import List
from Classes_RL.Hyperparameters_RL import Hyperparamaters_bandits
from Classes_RL.Bandit import Bandit
from Classes_RL.Interfaces import IPlotBehavior
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from Classes_RL.SolverFactory import Models
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)

class DedicaterName():

    @staticmethod
    def result_name(models_enum_name):

        name = ""

        # Boltzmann

        if models_enum_name == Models.Boltzmann_stationary:
            name = "Boltzmann-S"

        elif models_enum_name == Models.Boltzmann_UCB_stationary:
            name = "Boltzmann-S UCB"

        elif models_enum_name == Models.Boltzmann_non_stationary:
            name = "Boltzmann"

        elif models_enum_name == Models.Boltzmann_UCB_non_stationary:
            name = "Boltzmann UCB"

        # Bayesian

        elif models_enum_name == Models.Bayesian_um_kv_softmax_stationary:
            name = "Bayesian-S UMKV"

        elif models_enum_name == Models.Bayesian_um_kv_softmax_non_stationary:
            name = "Bayesian UMKV"

        elif models_enum_name == Models.Bayesian_um_uv_softmax_stationary:
            name = "Bayesian-S UMUV"

        elif models_enum_name == Models.Bayesian_um_uv_softmax_non_stationary:
            name = "Bayesian UMUV"

        # Sliding

        elif models_enum_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:
            name = "Sliding UMKV"

        elif models_enum_name == Models.Bayesian_um_uv_softmax_sliding_non_stationary:
            name = "Sliding UMUV"

        elif models_enum_name == Models.Boltzmann_UCB_sliding:
            name = "Boltzmann SUCB"

        # Kalman

        elif models_enum_name == Models.Kalman_greedy:
            name = "Sibling Kalman Filter greedy"

        elif models_enum_name == Models.Kalman_softmax:
            name = "Sibling Kalman Filter"

        elif models_enum_name == Models.Kalman_UCB:
            name = "Sibling Kalman Filter greedy UCB"

        elif models_enum_name == Models.Kalman_UCB_softmax:
            name = "Sibling Kalman Filter UCB"

        elif models_enum_name == Models.Kalman_Thompson_greedy:
            name = "Sibling Kalman Filter greedy TS"

        elif models_enum_name == Models.Kalman_Thompson_softmax:
            name = "Sibling Kalman Filter TS"

        elif models_enum_name == Models.Kalman_e_greedy:
            name = "Sibling Kalman Filter e-greedy"

        return name

    @staticmethod
    def behavior_name(behavior_name):

        name = ""

        if behavior_name == "static":

            name = "Static"

        elif behavior_name == "sin":

            name = "Sin wave"

        elif behavior_name == "log":

            name = "Logarithm"

        elif behavior_name == "exp":

            name = "Exponential"

        elif behavior_name == "RndWalk":

            name = "Random walk"

        return name

class PlotBehaviorStaticStrategy(IPlotBehavior):

    def plot(self, bandits: List[Bandit]):

        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        n_actions = len(bandits)

        A = [i for i in range(n_actions)]
        mus = [bandits[i].mu_iterations[0] for i in range(n_actions)]
        stds = [bandits[i].std_iterations[0] for i in range(n_actions)]

        # Violin plot
        ax1.set_xlabel("Action")
        ax1.set_ylabel("Q")
        ax1.set_xticks(np.arange(min(A), max(A) + 1, 1.0))

        data = [np.random.normal(mean, std, size=1000) for mean, std in zip(mus, stds)]
        ax1.violinplot(data, A, points=20, widths=0.3, showmeans=True)

        for i in range(n_actions):
            textstr = '\n'.join((r'$\mu=%.2f$' % (mus[i],), r'$\sigma^2=%.2f$' % (stds[i],)))
            props = dict(boxstyle='round', facecolor='blue', alpha=0.0)
            ax1.text(A[i] + 0.1, mus[i], textstr, fontsize=14, bbox=props, ha='center', va='center')

        # Bar plot
        # Plot wanted % action from initial behavior
        ax2.set_xlabel("Action")
        ax2.set_ylabel("Action (%)")
        ax2.set_xticks(np.arange(min(A), max(A) + 1, 1.0))

        data = Utilities.softmax_temperature(np.array(mus), t=2)

        df = pd.DataFrame(data)
        df.plot.bar(ax=ax2)
        ax2.set_title("Optimal action")

        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

class PlotBehaviorNonStaticStrategy(IPlotBehavior):

    def plot(self, bandits: List[Bandit]):

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        n_actions = len(bandits)
        colors = PlotSolver.get_colors_map(n_actions)

        for i in range(n_actions):
            ax.plot(bandits[i].mu_iterations, label="arm " + str(i), color=colors[i])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Expected value for each arm")
        ax.legend()

        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

class PlotResults():

    results: dict
    solver_name: str
    path: str

    def plot_episode(self, results, path, oracle_data, subject_name, solver_name):

        episode = 0

        for i, model_name in enumerate(results.keys()):

            path_model = path + model_name + "\\"

            if not os.path.exists(path_model):
                os.makedirs(path_model)

            result = results[model_name]

            figure_name = solver_name + "_Q_" + str(episode) + "_" + model_name + "_" + subject_name
            figure_title = DedicaterName.result_name(model_name)

            if "boltzmann" in model_name:

                fig2 = PlotSolver.plot_Q_episode(result=result, oracle_data=oracle_data, episode=episode, figure_title=figure_title)
                PlotSolver.save_plot(fig=fig2, path=path_model, figure_name=figure_name, close_figure=True)

            elif "bayesian" in model_name or "kalman" in model_name or "sliding" in model_name:

                fig2 = PlotSolver.plot_Q_episode_lower_upper_bound(result=result, oracle_data=oracle_data, episode=episode, figure_title=figure_title)
                PlotSolver.save_plot(fig=fig2, path=path_model, figure_name=figure_name, close_figure=True)

            figure_name = solver_name + "_P_" + str(episode) + "_" + model_name + "_" + subject_name

            fig2 = PlotSolver.plot_P_episode(result=result, oracle_data=oracle_data, episode=episode, figure_title=figure_title)
            PlotSolver.save_plot(fig=fig2, path=path_model, figure_name=figure_name, close_figure=True)

    def plot_episodes(self, results_mean, path, oracle_data, subject_name, solver_name, metric):

        figure_name = solver_name + "_" + subject_name
        figure_title = solver_name + " - " + subject_name

        fig1 = PlotSolver.plot_simple_results(results_mean=results_mean, oracle_data=oracle_data, figure_title=figure_title)
        PlotSolver.save_plot(fig=fig1, path=path, figure_name=figure_name + "_simple", close_figure=True)

        fig2 = PlotSolver.plot_results(results_mean=results_mean, oracle_data=oracle_data, metric=metric, figure_title=figure_title)
        PlotSolver.save_plot(fig=fig2, path=path, figure_name=figure_name, close_figure=True)

        for i, model_name in enumerate(results_mean.keys()):

            path_model = path + model_name + "\\"

            if not os.path.exists(path_model):
                os.makedirs(path_model)

            result = results_mean[model_name]

            figure_name = solver_name + "_Q_mean_" + model_name + "_" + subject_name
            figure_title = DedicaterName.result_name(model_name)

            if "boltzmann" in model_name:

                fig2 = PlotSolver.plot_Q_episodes(result=result, oracle_data=oracle_data, figure_title=figure_title)
                PlotSolver.save_plot(fig=fig2, path=path_model, figure_name=figure_name, close_figure=True)

            elif "bayesian" in model_name or "kalman" in model_name or "sliding" in model_name:

                fig2 = PlotSolver.plot_Q_episodes_lower_upper_bound(result=result, oracle_data=oracle_data, figure_title=figure_title)
                PlotSolver.save_plot(fig=fig2, path=path_model, figure_name=figure_name, close_figure=True)

            figure_name = solver_name + "_P_mean_" + model_name + "_" + subject_name

            fig2 = PlotSolver.plot_P_episodes(result=result, oracle_data=oracle_data, figure_title=figure_title)
            PlotSolver.save_plot(fig=fig2, path=path_model, figure_name=figure_name, close_figure=True)

class PlotSolver:

    @staticmethod
    def get_colors_map(size: int):

        from matplotlib import cm
        colors = plt.cm.jet(np.linspace(0, 1, size)).tolist()
        colors.append([0.0, 0.0, 0.0, 1.0])

        return colors

    @staticmethod
    def plot_simple_results(results_mean: dict, oracle_data: dict, figure_title: str = ""):

        colors = PlotSolver.get_colors_map(size=len(results_mean.keys()))

        plt.ioff()

        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        fig.suptitle(figure_title)

        idx = 0

        # Cumulative reward
        for k in results_mean.keys():

            ax1.plot(results_mean[k]["cum_reward"], label=DedicaterName.result_name(k), color=colors[idx])
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Cumulative reward")

            ax2.plot(results_mean[k]["cum_regret"], label=DedicaterName.result_name(k), color=colors[idx])
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Cumulative regret")

            ax3.plot(results_mean[k]["Q_mean_weighted"], label=DedicaterName.result_name(k), color=colors[idx])
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Weighted Q mean")

            idx += 1

        ax3.plot(oracle_data["Q_mean_weighted"], color=colors[-1], linestyle='-', linewidth=2, label="Oracle")

        ax1.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        ax2.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        ax3.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        ax4.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")

        # Optimal action
        action = {}
        label = []
        for i in results_mean:
            action[DedicaterName.result_name(i)] = results_mean[i]["action_percentage"]
            label.append(i)

        action["Oracle"] = oracle_data["action_percentage"]

        df = pd.DataFrame.from_dict(action)
        df.plot.bar(ax=ax4, rot=0, color=colors)
        ax4.set_xlabel("Arm")
        ax4.set_ylabel("Optimal action [%]")
        ax4.legend(action.keys())

        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_results(results_mean: dict, oracle_data: dict, metric: str, figure_title: str = ""):

        colors = PlotSolver.get_colors_map(size=len(results_mean.keys()))

        plt.ioff()

        # Plot cumulative reward, cumulative regret, Q_mean, Percentage action
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        fig.suptitle(figure_title)

        idx = 0

        # Cumulative reward
        for k in results_mean.keys():

            if metric == "cross_entropy":

                ax1.plot(results_mean[k]["cum_cross_entropy"], label=DedicaterName.result_name(k), color=colors[idx])
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Cumulative cross entropy")

            elif metric == "ae":

                ax1.plot(results_mean[k]["cum_ae"], label=DedicaterName.result_name(k), color=colors[idx])
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Cumulative absolute error")

            ax2.plot(results_mean[k]["Q_mean_weighted"], label=DedicaterName.result_name(k), color=colors[idx])
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Q weighted mean")

            idx += 1

        ax2.plot(oracle_data["Q_mean_weighted"], color=colors[-1], label="Oracle", linestyle='-', linewidth=2)

        ax1.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        ax2.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")

        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_Q_episodes(result: dict, oracle_data, figure_title: str = ""):

        plt.ioff()

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        fig.suptitle(figure_title)

        mu = result["Q"]
        Q_oracle = oracle_data["Q"]
        n_iteration = mu.shape[0]
        n_actions = mu.shape[1]
        arm_names = ["Cube", "Cylinder", "Heart", "Infinite", "Sphere", "Triangle"]
        colors_artificial = ["Blues", "Reds", "RdPu", "Purples", "YlOrBr", "Greens"]


        colors = PlotSolver.get_colors_map(size=n_actions)

        for n in range(n_actions):

            ax.plot(np.arange(n_iteration), mu[:, n], label="Arm (Model) : " + str(n), color=colors[n])
            ax.plot(Q_oracle[:, n], color=colors[n], linestyle="--", alpha=0.7, label="Arm (Oracle) : " + str(n))

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Expected reward")

        ax.set_ylim(bottom=0)
        ax.legend(markerscale=1, title="Bandits", bbox_to_anchor=(1., 0.), loc='lower left', edgecolor="black")

        plt.suptitle(figure_title)

        plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.8, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_Q_episode(result: dict, oracle_data, episode: int = 0, figure_title: str = ""):

        plt.ioff()

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        fig.suptitle(figure_title)

        mu = result["Q"][episode]
        action = result["action"][episode]
        Q_oracle = oracle_data["Q"]
        n_iteration = mu.shape[0]
        n_actions = mu.shape[1]
        arm_names = ["Cube", "Cylinder", "Heart", "Infinite", "Sphere", "Triangle"]
        colors_artificial = ["Blues", "Reds", "RdPu", "Purples", "YlOrBr", "Greens"]

        colors = PlotSolver.get_colors_map(size=n_actions)

        for n in range(n_actions):

            ax.plot(np.arange(n_iteration), mu[:, n], label="Arm (Model) : " + str(n), color=colors[n])
            ax.plot(Q_oracle[:, n], color=colors[n], linestyle="--", alpha=0.7, label="Arm (Oracle) : " + str(n))

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Expected reward")

        plt.suptitle(figure_title)

        #  plot color as arm pulled
        for i in range(n_iteration):
            ax.scatter(i, -0.1, color=colors[int(action[i])], s=5)

        ax.set_ylim(bottom=-0.2)
        ax.legend(markerscale=1, title="Bandits", bbox_to_anchor=(1., 0.), loc='lower left', edgecolor="black")

        # plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.8, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_P_episodes(result: dict, oracle_data, figure_title: str = ""):

        plt.ioff()

        # Plot cumulative reward, cumulative regret, Q_mean, Percentage action
        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        P_true = oracle_data['P']
        P_model = result["P"]

        n_iteration = P_true.shape[0]
        n_actions = P_true.shape[1]

        colors = PlotSolver.get_colors_map(size=n_actions)

        for n in range(n_actions):

            ax.plot(np.arange(n_iteration), P_true[:, n], linestyle="-", label="Arm (Oracle) : " + str(n), color=colors[n])
            ax.plot(np.arange(n_iteration), P_model[:, n], linestyle="--", label="Arm (Model) : " + str(n), color=colors[n])

        ax.set_xlabel("iteration")
        ax.set_ylabel("Probabilities")

        fig.suptitle(figure_title)

        ax.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        ax.set_ylim(bottom=0)

        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_P_episode(result: dict, oracle_data, episode: int = 0, figure_title: str = ""):

        plt.ioff()

        # Plot cumulative reward, cumulative regret, Q_mean, Percentage action
        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        P_true = oracle_data['P']
        P_model = result["P"][episode]
        action = result["action"][episode]

        n_iteration = P_true.shape[0]
        n_actions = P_true.shape[1]

        colors = PlotSolver.get_colors_map(size=n_actions)

        for n in range(n_actions):

            ax.plot(np.arange(n_iteration), P_true[:, n], linestyle="-", label="Arm (Oracle) : " + str(n), color=colors[n])
            ax.plot(np.arange(n_iteration), P_model[:, n], linestyle="--", label="Arm (Model) : " + str(n), color=colors[n])

        ax.set_xlabel("iteration")
        ax.set_ylabel("Probabilities")

        fig.suptitle(figure_title)

        ax.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")

        #  plot color as arm pulled
        for i in range(n_iteration):
            ax.scatter(i, -0.1, color=colors[int(action[i])], s=5)

        ax.set_ylim(bottom=-0.2)

        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_3D_heatmap(results: np.ndarray, solvers_name: List, behaviors_name: List, iterations: List):

        import seaborn as sns
        sns.set(style='ticks', rc={"grid.linewidth": 0.1})
        sns.set_context("paper", font_scale=2.2)
        color = sns.color_palette("Set2", 6)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        " Results: should be of size (n_behaviors, n_solvers, n_iterations) "

        n_solvers = len(solvers_name)
        max_solver_per_plot = 4
        n_plots = math.ceil(n_solvers/max_solver_per_plot)
        index = []

        if n_plots == 1:
            index.append([0, n_solvers])
        else:
            for i in range(n_plots):
                if i == (n_plots - 1):
                    index.append([max_solver_per_plot*(i-1) + max_solver_per_plot, n_solvers])
                else:
                    index.append([max_solver_per_plot*(i-1) + max_solver_per_plot, max_solver_per_plot*i + max_solver_per_plot])

        figs = []

        for ind in index:

            start = ind[0]
            end = ind[1]

            fig, axes = plt.subplots(1, end - start, figsize=(20, 10), sharex=True, sharey=True)
            #cbar_ax = fig.add_axes([.91, .1, .03, .8])

            i = 0

            for k in range(start, end):

                df = pd.DataFrame(results[:, k, :].transpose(), columns=behaviors_name, index=iterations)
                sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", cbar=False, linewidths=.5, ax=axes[i], cbar_ax=None)

                if k == (end - 1):
                    pass
                    #sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", cbar=True, linewidths=.5, ax=axes[i], cbar_ax=cbar_ax)
                else:
                    pass
                    # sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", cbar=False, linewidths=.5, ax=axes[i], cbar_ax=None)

                axes[i].set_title(DedicaterName.result_name(solvers_name[k]), fontsize=20)
                axes[i].tick_params(axis='x', labelsize=15)
                axes[i].tick_params(axis='y', labelsize=15)

                for tick in axes[i].get_xticklabels():
                    tick.set_rotation(45)

                if k == 0:
                    axes[i].set_ylabel("Iterations", fontsize=20)

                i += 1

            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.05, hspace=0.1)
            # fig.tight_layout(rect=[0, 0, .9, 1])
            plt.show()

            figs.append(fig)

        return figs

    @staticmethod
    def plot_Q_episode_lower_upper_bound(result: dict, oracle_data, episode: int = 0, figure_title: str = ""):

        plt.ioff()

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        fig.suptitle(figure_title)

        mu = result["Q"][episode]
        var = result["var_bar"][episode]
        action = result["action"][episode]
        mu_known = oracle_data["mu_known"]

        n_iteration = mu.shape[0]
        n_actions = mu.shape[1]

        colors = PlotSolver.get_colors_map(size=n_actions)
        arm_names = ["Cube", "Cylinder", "Heart", "Infinite", "Sphere", "Triangle"]
        colors_artificial = ["deepskyblue", "crimson", "orchid", "blueviolet", "gold", "lime"]

        mmax = np.zeros(shape=(n_iteration, n_actions))
        mmin = np.zeros(shape=(n_iteration, n_actions))

        min_points = []
        max_points = []

        for n in range(n_actions):

            ax.plot(mu_known[:, n], color=colors_artificial[n], linestyle="--", alpha=0.7, lw=2)
            ax.plot(np.arange(n_iteration), mu[:, n], label=arm_names[n], color=colors_artificial[n], lw=2)

            # draw behavior of each variance
            mmax[:, n] = mu[:, n] + np.sqrt(var[:, n])
            mmin[:, n] = mu[:, n] - np.sqrt(var[:, n])

            min_points.append(min(mu_known[:, n]))
            max_points.append(max(mu_known[:, n]))
            min_points.append(min(mu[:, n]))

            ax.plot(np.arange(n_iteration), mmax[:, n], alpha=0.3, color=colors_artificial[n], lw=2)
            ax.plot(np.arange(n_iteration), mmin[:, n], alpha=0.3, color=colors_artificial[n], lw=2)
            ax.fill_between(np.arange(n_iteration), mmax[:, n], mmin[:, n], alpha=0.1, color=colors_artificial[n])

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Expected reward")

        min_point = min(min_points)
        max_point = max(max_points)

        ax.set_ylim(min_point - 0.1, max_point + 0.1)

        plt.suptitle(figure_title)

        #  plot color as arm pulled
        for i in range(n_iteration):
            ax.scatter(i, min_point - 0.05, color=colors_artificial[int(action[i])], s=40)

        ax.legend(markerscale=1, title="Bandits", bbox_to_anchor=(1., 0.), loc='lower left', edgecolor="black")

        # plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.8, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_Q_episodes_lower_upper_bound(result: dict, oracle_data, figure_title: str = ""):

        plt.ioff()

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        fig.suptitle(figure_title)

        mu = result["Q"]
        var = result["var_bar"]
        action = result["action"]
        mu_known = oracle_data["mu_known"]

        n_iteration = mu.shape[0]
        n_actions = mu.shape[1]

        colors = PlotSolver.get_colors_map(size=n_actions)
        arm_names = ["Cube", "Cylinder", "Heart", "Infinite", "Sphere", "Triangle"]
        colors_artificial = ["blue", "crimson", "fuchsia", "darkviolet", "gold", "limegreen"]

        mmax = np.zeros(shape=(n_iteration, n_actions))
        mmin = np.zeros(shape=(n_iteration, n_actions))

        min_points = []
        max_points = []

        for n in range(n_actions):

            ax.plot(mu_known[:, n], color=colors_artificial[n], linestyle="--", alpha=0.7)
            ax.plot(np.arange(n_iteration), mu[:, n], label=arm_names[n], color=colors_artificial[n])

            # draw behavior of each variance
            mmax[:, n] = mu[:, n] + np.sqrt(var[:, n])
            mmin[:, n] = mu[:, n] - np.sqrt(var[:, n])

            min_points.append(min(mu_known[:, n]))
            max_points.append(max(mu_known[:, n]))

            ax.plot(np.arange(n_iteration), mmax[:, n], alpha=0.3, color=colors_artificial[n])
            ax.plot(np.arange(n_iteration), mmin[:, n], alpha=0.3, color=colors_artificial[n])
            ax.fill_between(np.arange(n_iteration), mmax[:, n], mmin[:, n], alpha=0.1, color=colors_artificial[n])

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Expected reward")

        min_point = min(min_points)
        max_point = max(max_points)

        ax.set_ylim(min_point - 0.2, max_point + 0.2)
        ax.legend(markerscale=1, title="Bandits", bbox_to_anchor=(1., 0.), loc='lower left', edgecolor="black")

        plt.suptitle(figure_title)

        # plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.8, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_gaussian_results_iterations(result: dict, oracle_data: dict, iteration: int, violin: bool = True):

        fig = plt.figure(figsize=(18, 8))

        ax = fig.add_subplot(111)

        mu_iterations = oracle_data["mu_known"]
        std_iterations = oracle_data["std_known"]

        n_actions = mu_iterations.shape[1]

        A = [i for i in range(n_actions)]
        mu = [mu_iterations[int(iteration-1)][a] for a in range(n_actions)]
        std = [std_iterations[int(iteration-1)][a] for a in range(n_actions)]

        A_bar = [i + 0.1 for i in range(n_actions)]
        mu_bar = [result["mu_bar"][int(iteration-1)][a] for a in range(n_actions)]
        var_bar = [result["var_bar"][int(iteration-1)][a] for a in range(n_actions)]
        std_bar = [np.sqrt(v) for v in var_bar]

        ax.set_xlabel("Action")
        ax.set_ylabel("Q")
        ax.set_title(str(iteration))
        ax.set_xticks(np.arange(min(A), max(A) + 1, 1.0))

        if violin:
            data = [np.random.normal(mean, std, size=1000) for mean, std in zip(mu, std)]
            violin_parts_p = ax.violinplot(data, A, points=20, widths=0.3, showmeans=True)


            data_bar = [np.random.normal(mean, std, size=1000) for mean, std in zip(mu_bar, std_bar)]
            violin_parts = ax.violinplot(data_bar, A_bar, points=1000,
                                         widths=0.3, showmeans=True)

            for pc in violin_parts['bodies']:
                pc.set_facecolor('red')
                pc.set_edgecolor('red')

        ax.scatter(A, mu, label="True mean")
        ax.scatter(A_bar, mu_bar, label="Inferred mean")

        for i in range(n_actions):

            if not violin:
                ax.errorbar(A[i] - 0.1, mu[i], yerr=std[i], color="b", marker="o", linestyle="none", capsize=6,
                            capthick=1)
                ax.errorbar(A[i] + 0.1, mu_bar[i], yerr=std_bar[i], color="r", marker="o", linestyle="none", capsize=6,
                            capthick=1)

            textstr = '\n'.join((r'$\mu=%.2f$' % (mu[i],), r'$\sigma^2=%.2f$' % (std[i],)))
            props = dict(boxstyle='round', facecolor='blue', alpha=0.0)
            ax.text(A[i] + 0.1, mu[i], textstr, fontsize=14, bbox=props, ha='center', va='center')

            textstr_bar = '\n'.join((r'$\mu=%.2f$' % (mu_bar[i],), r'$\sigma^2=%.2f$' % (std_bar[i],)))
            props = dict(boxstyle='round', facecolor='blue', alpha=0.0)
            ax.text(A[i] + 0.5, mu[i] - 1, textstr_bar, fontsize=14, color="r", bbox=props, ha='center', va='center')

        plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def bar_iterations(results: dict, arm_n: int, iteration: int, segmentation, metric_type):

        behaviors = [name for name in results]

        iterations = [int(iteration*i) for i in segmentation]

        x = [Hyperparamaters_bandits.result_name(name) for name in results[behaviors[0]][arm_n]]
        y = {}

        for behav in behaviors:

            for i in iterations:

                if metric_type == "regret":

                    y["iteration : " + str(i)] = [round(results[behav][arm_n][name]["cum_regret_a"][i-1])for name in results[behav][arm_n]]

                elif metric_type == "loss":

                    y["iteration : " + str(i)] = [round(results[behav][arm_n][name]["cum_cross_entropy"][i-1])for name in results[behav][arm_n]]

            df = pd.DataFrame(y, index=x)

            ax = df.plot(kind="bar", alpha=0.7, rot=30)

            for container in ax.containers:
                ax.bar_label(container)

            if metric_type == "regret":

                plt.ylabel("Cumulative regret each arm has")

            elif metric_type == "loss":

                plt.ylabel("Cumulative Cross-entropy")

            max_point = max(y["iteration : " + str(iterations[-1])])
            plt.ylim(0, max_point+100)

            plt.title("Behavior : " + str(behav))
            plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)
            plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
            plt.show()

        return ax

    @staticmethod
    def bar_arms_result(results_arms: dict, iteration: int, behav_name:str, segmentation:list, metric_type):

        arms = [a for a in results_arms]
        algos = [a for a in results_arms[arms[0]]]
        iterations = [int(iteration*i) for i in segmentation]

        x = ["Arm : " + str(a) for a in results_arms]

        y = {}

        for name in algos:

            for i in iterations:

                if metric_type == "regret":

                    y["iteration : " + str(i)] = [round(results_arms[a][name]["cum_regret_a"][i-1]) for a in arms]

                if metric_type == "loss":

                    y["iteration : " + str(i)] = [round(results_arms[a][name]["cum_cross_entropy"][i-1]) for a in arms]

            df = pd.DataFrame(y, index=x)

            ax = df.plot(kind="bar", alpha=0.7, rot=30)

            for container in ax.containers:
                ax.bar_label(container)

            if metric_type == "regret":

                plt.ylabel("Cumulative regret each arm has")

            elif metric_type == "loss":

                plt.ylabel("Cumulative Cross-entropy")

            max_point = max(y["iteration : " + str(iterations[-1])])
            plt.ylim(0, max_point + 100)

            plt.title("Behavior : " + behav_name + " - Algorithm : " + name)
            plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)
            plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
            plt.show()

    @staticmethod
    def heatmap_arms_result_all(results_arms: dict, iteration: int, behav_name: str, segment: int, metric_type):

        fig = plt.figure(figsize=[14, 14])

        import seaborn as sns

        arms = [a for a in results_arms]
        algos = [a for a in results_arms[arms[0]]]
        _iteration = segment * iteration

        x = ["Bandits : " + str(a) for a in results_arms]

        y = {}

        for name in algos:

            if metric_type == "regret":
                y[DedicaterName.result_name(name)] = \
                    [results_arms[a][name]["cum_regret"][int(_iteration)-1] for a in arms]

            if metric_type == "loss":
                y[DedicaterName.result_name(name)] = \
                    [results_arms[a][name]["cum_cross_entropy"][int(_iteration)-1] for a in arms]

        df = pd.DataFrame(y, index=x)

        if metric_type == "regret":

            title_name = "Metric : Cumulative regret" + " - Iteration : " + \
                         str(_iteration) + " - Behavior type : " + DedicaterName.behavior_name(behav_name)

        elif metric_type == "loss":

            title_name = "Metric : Cumulative Cross entropy" + " - Iteration : " + \
                         str(_iteration) + " - Behavior type : " + DedicaterName.behavior_name(behav_name)


        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f",  linewidths=.5)

        plt.title(title_name)
        plt.yticks(rotation=30)
        plt.xticks(rotation=30)
        plt.xlabel("The name of the algorithm for each")
        plt.ylabel("The number of bandits")
        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)
        # plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")

        return fig

    @staticmethod
    def heatmap_behaviors_result_all(behaviors_results: dict, iteration: int, segment: int, metric_type):

        fig = plt.figure(figsize=[14, 14])

        import seaborn as sns

        behaviors = [b for b in behaviors_results]
        arms = [a for a in behaviors_results[behaviors[0]]]
        algos = [a for a in behaviors_results[behaviors[0]][arms[0]]]
        _iteration = iteration * segment

        x = [DedicaterName.behavior_name(b) for b in behaviors_results]

        y = {}

        for name in algos:

            if metric_type == "regret":

                y[DedicaterName.result_name(name)] = \
                    [behaviors_results[b][4][name]["cum_regret"][int(_iteration)-1] for b in behaviors]

            elif metric_type == "loss":

                y[DedicaterName.result_name(name)] = \
                    [behaviors_results[b][4][name]["cum_cross_entropy"][int(_iteration)-1] for b in behaviors]

        df = pd.DataFrame(y, index=x)

        if metric_type == "regret":

            title_name = "Metric : Cumulative regret" + " - Iteration : " + \
                         str(_iteration)

        elif metric_type == "loss":

            title_name = "Metric : Cumulative Cross entropy" + " - Iteration : " + \
                         str(_iteration)


        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f",  linewidths=.5)

        plt.title(title_name)
        plt.yticks(rotation=30)
        plt.xticks(rotation=30)
        plt.xlabel("The name of the algorithm for each")
        plt.ylabel("The name of the behavior for each")
        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)
        # plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")

        return fig

    @staticmethod
    def bar_behaviors_result(behaviors_results: dict, iteration: str, segmentation: list, metric_type):

        behaviors = [b for b in behaviors_results]
        arms = [a for a in behaviors_results[behaviors[0]]]
        algos = [a for a in behaviors_results[behaviors[0]][arms[0]]]
        iterations = [int(iteration * i) for i in segmentation]

        x = [str(b) for b in behaviors]

        y = {}

        for name in algos:

            for i in iterations:

                if metric_type == "regret":

                    y["iteration : " + str(i)] = [round(behaviors_results[b][4][name]["cum_regret"][i-1])for b in behaviors]

                elif metric_type == "loss":

                    y["iteration : " + str(i)] = [round(behaviors_results[b][4][name]["cum_cross_entropy"][i-1])for b in behaviors]

            df = pd.DataFrame(y, index=x)

            ax = df.plot(kind="bar", alpha=0.7, rot=30)

            for container in ax.containers:
                ax.bar_label(container)


            if metric_type == "regret":

                plt.ylabel("Cumulative regret each arm has")

            elif metric_type == "loss":

                plt.ylabel("Cumulative Cross-entropy")

            max_point = max(y["iteration : " + str(iterations[-1])])
            plt.ylim(0, max_point + 100)
            plt.title("Algorithm : " + name)
            plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)
            plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
            plt.show()

    @staticmethod
    def plot_final_gaussian_results(result: dict, violin: bool = True):

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        mu_iterations = result["mu_iterations"]
        std_iterations = result["std_iterations"]

        n_actions = len(mu_iterations)

        A = [i for i in range(n_actions)]
        mu = [mu_iterations[i][0] for i in range(n_actions)]
        std = [std_iterations[i][0] for i in range(n_actions)]

        A_bar = [i + 0.1 for i in range(n_actions)]
        mu_bar = result["mu_bar"]
        var_bar = result["var_bar"]
        std_bar = [np.sqrt(v) for v in var_bar]

        ax.set_xlabel("Action")
        ax.set_ylabel("Q")
        ax.set_xticks(np.arange(min(A), max(A) + 1, 1.0))

        if violin:
            data = [np.random.normal(mean, std, size=1000) for mean, std in zip(mu, std)]
            ax.violinplot(data, A, points=20, widths=0.3, showmeans=True)

            data_bar = [np.random.normal(mean, std, size=1000) for mean, std in zip(mu_bar, std_bar)]
            violin_parts = ax.violinplot(data_bar, A_bar, points=1000, widths=0.3, showmeans=True)

            for pc in violin_parts['bodies']:
                pc.set_facecolor('red')
                pc.set_edgecolor('red')

        for i in range(n_actions):

            if not violin:
                ax.errorbar(A[i] - 0.1, mu[i], yerr=std[i], color="b", marker="o", linestyle="none", capsize=6,
                             capthick=1)
                ax.errorbar(A[i] + 0.1, mu_bar[i], yerr=std_bar[i], color="r", marker="o", linestyle="none", capsize=6,
                             capthick=1)

            textstr = '\n'.join((r'$\mu=%.2f$' % (mu[i],), r'$\sigma^2=%.2f$' % (std[i],)))
            props = dict(boxstyle='round', facecolor='blue', alpha=0.0)
            ax.text(A[i] + 0.1, mu[i], textstr, fontsize=14, bbox=props, ha='center', va='center')

            textstr_bar = '\n'.join((r'$\mu=%.2f$' % (mu_bar[i],), r'$\sigma^2=%.2f$' % (std_bar[i],)))
            props = dict(boxstyle='round', facecolor='blue', alpha=0.0)
            ax.text(A[i] + 0.5, mu[i] - 1, textstr_bar, fontsize=14, color="r", bbox=props, ha='center', va='center')

        ax.legend()
        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def save_plot(fig, path, figure_name: str, close_figure: bool = True):

        fig.savefig(path + figure_name + ".png")
        print("Figure saved to: " + path + figure_name + ".png")

        if close_figure:
            plt.close(fig)

class AnimateSolver:

    @staticmethod
    def animate_gaussian_arm(result: dict, gif_name: str = "", path: str = "", is_unknown_variance=True,):

        n_episode = 1  # Just plot one episode (Too slow)

        mu = result["mu_iterations"]
        var = result["var_iterations"]
        std = [np.sqrt(v) for v in var]

        mu_bar = result["mu_bar"]
        var_bar = result["var_bar"]
        std_bar = [np.sqrt(v) for v in var_bar]

        var_bar_known = result["var_bar_known"]
        std_bar_known = [np.sqrt(v) for v in var_bar_known]

        Q = result["Q"]

        n_iteration = Q.shape[0]
        n_action = Q.shape[1]

        filenames = []
        gif_name = gif_name + "_" + str(1)

        fig, axes = plt.subplots(2, 5, figsize=(22.5, 9))
        axes = axes.reshape(-1)

        plt.subplots_adjust(left=0.2, bottom=0.2, right=1, top=0.90, wspace=0.2, hspace=0.2)

        for k in range(n_iteration):

            print("Plot Gaussian " + str(k) + "/" + str(n_iteration))

            filename = "temp_" + str(k) + "_" + str(1)
            filenames.append(filename)

            fig.suptitle("Iteration " + str(k))
            ymax = 0

            x = np.linspace(0.0, np.max(mu) + 3 * std[0], 200)

            for j in range(n_action):

                # Estimated PDF(probability density function) from known variance
                if not is_unknown_variance:

                    y = stats.norm.pdf(x, mu_bar[j], std_bar_known[j])
                    p = axes[j].plot(x, y, color="g", lw=2, label="Estimated arm " + str(j))
                    c = p[0].get_markeredgecolor()
                    axes[j].fill_between(x, y, 0, color="g", alpha=0.2)

                # Estimated PDF
                y = stats.norm.pdf(x, mu_bar[j], std_bar[j])
                p = axes[j].plot(x, y, color="r", lw=2, label="Estimated arm " + str(j))
                c = p[0].get_markeredgecolor()
                axes[j].fill_between(x, y, 0, color="r", alpha=0.2)

                # True PDF
                y = stats.norm.pdf(x, mu[j], std[j])
                p = axes[j].plot(x, y, color="b", lw=2, label="True arm " + str(j))
                c = p[0].get_markeredgecolor()
                axes[j].fill_between(x, y, 0, color="b", alpha=0.2)

                axes[j].legend(loc='upper left')
                axes[j].set_xlabel("Mean")
                axes[j].set_ylabel("Probability density")

                ymax = max(ymax, y[1:].max() * 1.05)

            plt.savefig(path + filename, dpi=150)

            for j in range(n_action):
                axes[j].cla()  # Clear all axis

        #build gif
        print("Creating Gaussian .gif (Take time)")

        with imageio.get_writer(path + gif_name + '.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(path + filename + ".png")
                writer.append_data(image)

        print("Saving Gaussian .gif to:" + str(path + gif_name + ".gif"))

        # Remove files
        for filename in set(filenames):
            os.remove(path + filename + ".png")

    @staticmethod
    def animate_oracle(oracle_data, path: str = "", gif_name: str = ""):

        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 1, 1)

        mu_known = oracle_data["mu_known"]
        action_known = oracle_data["action"]

        n_iteration = mu_known.shape[0]
        n_actions = mu_known.shape[1]

        colors = PlotSolver.get_colors_map(size=n_actions)

        mmax = np.zeros(shape=(n_iteration, n_actions))
        mmin = np.zeros(shape=(n_iteration, n_actions))

        filenames = []

        for i in tqdm(range(n_iteration)):

            filename = "temp_" + str(i) + "_" + str(1)
            filenames.append(filename)
            fig.suptitle("Iteration : " + str(i))

            min_points = []
            max_points = []

            for n in range(n_actions):
                ax.plot(mu_known[:i, n], color=colors[n], linestyle="--", alpha=0.7,
                        label="Oracle mean : Arm " + str(n))

                ax.set_xlabel("Iteration")
                ax.set_ylabel("Oracle mean")

                min_points.append(min(mu_known[:, n]))
                max_points.append(max(mu_known[:, n]))

            min_point = min(min_points)
            max_point = max(max_points)
            ax.set_ylim(min_point - 2, max_point + 1)
            ax.set_xlim(0, i + 1)

            # plt.suptitle(figure_title)

            # for k in range(i):
            #     ax.scatter(k, min_point - 1, color=colors[int(action_known[k])], s=40)

            plt.legend(loc=2, frameon=True, fancybox=False, ncol=3, framealpha=0.5, edgecolor="black")
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.2, hspace=0.2)

            plt.savefig(path + filename, dpi=150)

            ax.cla()

        print("Creating Q .gif (Take time)")

        with imageio.get_writer(path + gif_name + '.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(path + filename + ".png")
                writer.append_data(image)

        print("Saving Gaussian .gif to:" + str(path + gif_name + ".gif"))

        # Remove files
        for filename in set(filenames):
            os.remove(path + filename + ".png")

    @staticmethod
    def animate_Q_lower_upper_bound(result, oracle_data, plob: bool = True, path: str = "", gif_name: str = ""):

        ncols = 1
        nrows = 2

        fig = plt.figure(figsize=(22.5, 8))
        gs = gridspec.GridSpec(ncols, nrows, width_ratios=(3, 1))  # *1
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        mu = result["Q"]
        var = result["var_bar"]
        std = result["std_bar"]
        action = result["action"]
        mu_known = oracle_data["mu_known"]
        std_known = oracle_data["std_known"]

        n_iteration = mu.shape[0]
        n_actions = mu.shape[1]

        colors = PlotSolver.get_colors_map(size=n_actions)

        mmax = np.zeros(shape=(n_iteration, n_actions))
        mmin = np.zeros(shape=(n_iteration, n_actions))

        filenames = []

        xlim = [0, 100]

        for i in tqdm(range(n_iteration)):

            filename = "temp_" + str(i) + "_" + str(1)
            filenames.append(filename)
            fig.suptitle("Iteration : " + str(i))

            if i > xlim[1]:
                xlim[0] += 1
                xlim[1] += 1

            min_points = []
            max_points = []

            for n in range(n_actions):
                ax.plot(mu_known[:, n], color=colors[n], linestyle="--", alpha=0.7,
                        label="Oracle mean : Arm " + str(n))
                ax.plot(np.arange(i), mu[:i, n], label="Sample mean : Arm " + str(n), color=colors[n])

                # draw behavior of each variance
                mmax[:, n] = mu[:, n] + np.sqrt(var[:, n])
                mmin[:, n] = mu[:, n] - np.sqrt(var[:, n])

                ax.plot(np.arange(i), mmax[:i, n], alpha=0.3, color=colors[n])
                ax.plot(np.arange(i), mmin[:i, n], alpha=0.3, color=colors[n])
                ax.fill_between(np.arange(i), mmax[:i, n], mmin[:i, n], alpha=0.05, color=colors[n])

                ax.set_xlabel("Iteration")
                ax.set_ylabel("Oracle mean & Sample mean")

                x = np.linspace(0.0, np.max(mu_known[:, n]) + 3 * std_known[0, n], 200)
                x_e = np.linspace(0.0, np.max(mu[:, n]) + 3 * std[0, n] + 5, 400)

                # Estimated PDF
                if plob:
                    y = stats.norm.pdf(x_e, mu[i, n], std[i, n])
                    p = ax2.plot(y, x_e, color=colors[n], lw=2, alpha = 0.3, label="")
                    c = p[0].get_markeredgecolor()
                    ax2.fill_between(y, x_e, 0, color=colors[n], alpha=0.05)

                else:
                    ax2.axhline(mu[i, n], ls = "-", color = colors[n])

                # True PDF
                y = stats.norm.pdf(x, mu_known[i, n], std_known[i, n])
                p = ax2.plot(y, x, color=colors[n], lw=2, linestyle="--", alpha = 0.3, label="")
                c = p[0].get_markeredgecolor()
                ax2.fill_between(y, x, 0, color=colors[n], alpha=0.05)

                ax2.set_xlabel("Probability density")
                min_points.append(min(mu_known[:, n]))
                max_points.append(max(mu_known[:, n]))

            min_point = min(min_points)
            max_point = max(max_points)
            ax.set_ylim(min_point - 2, max_point + 2)
            ax.set_xlim(xlim[0], xlim[1])
            ax2.set_ylim(min_point - 2, max_point + 2)
            ax2.set_xlim(0.0, 0.6)

            # plt.suptitle(figure_title)

            for k in range(i):
                ax.scatter(k, min_point - 1, color=colors[int(action[k])], s=40)

            ax.legend(loc=2, frameon=True, fancybox=False, ncol=3, framealpha=0.5, edgecolor="black", fontsize=20)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0, hspace=0.2)

            plt.savefig(path + filename, dpi=150)

            ax.cla()
            ax2.cla()

        print("Creating Q .gif (Take time)")

        with imageio.get_writer(path + gif_name + '.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(path + filename + ".png")
                writer.append_data(image)

        print("Saving Gaussian .gif to:" + str(path + gif_name + ".gif"))

        # Remove files
        for filename in set(filenames):
            os.remove(path + filename + ".png")