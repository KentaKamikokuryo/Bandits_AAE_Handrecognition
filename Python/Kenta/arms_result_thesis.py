# the classes of Reinforcement learning
from Classes_RL.Info_RL import BanditInfo
from Classes_RL.BanditsFactory import BanditsFactory
from Kenta.Thesis_RL import Manager
import seaborn as sns
import matplotlib.pyplot as plt
from Classes_RL.Plot_RL import PlotSolver, PlotBehaviorNonStaticStrategy
import os

# plt.ioff()

sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.2)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# simple
cwd = os.getcwd()
# Get path one level above the root of the project
path_parent_project = os.path.abspath(os.path.join(cwd, os.pardir))
# Path to the data
path_test = path_parent_project + "\\Test\\"
if not (os.path.exists(path=path_test)):
    os.makedirs(path_test)

search_info = [False]
#
# evaluation_info = {"segmentation": [0.1, 0.5, 1],
#                    "metric_type": "loss"}

# search_info = [False]

evaluation_info = {"segmentation": [0.1, 0.5, 1],
                   "metric_type": "loss"}

bandit_info = BanditInfo()
behavior_parameters = bandit_info.behavior_parameters
behavior_parameters.pop("real")
behavior_parameters.pop("unsteady")

############################################################################
behavior_parameters = {}
behavior_parameters["sin"] = {"name": "sin", "amplitude_base": [1, 2], "frequency_range": [1, 3],
                                   "std_amplitude": 1}
############################################################################

results_behaviors = {}

for behavior_name in behavior_parameters:

    results_arms = {}

    for a in range(3, 4):

        RL_info = {"n_action": a,
                   "n_iteration": 100,
                   "n_episode": 30}

        print("running while behavior names : " + behavior_name)

        # get behavior parameter
        behavior_parameter = bandit_info.get_parameters(behavior_name)

        # create bandits at here, since the all bandits should be same to evaluate
        bandit_fac = BanditsFactory(parameters=behavior_parameter, n_iteration=RL_info["n_iteration"],
                                    n_action=RL_info["n_action"])

        bandits = bandit_fac.create()

        #
        # plotter = PlotBehaviorNonStaticStrategy()
        # fig = plotter.plot(bandits=bandits)
        # plt.legend(loc=4, frameon=True, fancybox=False, ncol=1, framealpha=0.7, edgecolor="black")
        # fig.savefig(path_test + behavior_name + ".png")

        for s in search_info:

            manager = Manager(bandits=bandits,
                              RL_info=RL_info,
                              search_info=s,
                              evaluation_info=evaluation_info)

            results, results_mean = manager.run()
            oracle = manager.oracle

            if s:

                pass

            else:

                results_arms[a] = results_mean

    results_behaviors[behavior_name] = results_arms

# PlotSolver.bar_iterations(results=results_behaviors,
#                           iteration=RL_info["n_iteration"],
#                           segmentation=evaluation_info["segmentation"],
#                           arm_n=4,
#                           metric_type=evaluation_info["metric_type"])

    for segment in evaluation_info["segmentation"]:

        fig = PlotSolver.heatmap_arms_result_all(results_arms=results_arms,
                                                 iteration=RL_info["n_iteration"],
                                                 behav_name=behavior_name,
                                                 segment=segment,
                                                 metric_type=evaluation_info["metric_type"])

        if evaluation_info["metric_type"] == "regret":

            fig_name = "Metric_CumulativeRegret" + "_Iteration_" + \
                         str(RL_info["n_iteration"]*segment) + "_BehaviorType_" + behavior_name

        elif evaluation_info["metric_type"] == "loss":

            fig_name = "Metric_CumulativeCrossEntropy" + "_Iteration_" + \
                         str(RL_info["n_iteration"]*segment) + "_Behavior type_" + behavior_name

        fig.savefig(path_test + fig_name + ".png")

        plt.clf()
        plt.close("all")

# for segment in evaluation_info["segmentation"]:
#
#     fig = PlotSolver.heatmap_behaviors_result_all(behaviors_results=results_behaviors,
#                                                   iteration=RL_info["n_iteration"],
#                                                   segment=segment,
#                                                   metric_type=evaluation_info["metric_type"])
#
#     if evaluation_info["metric_type"] == "regret":
#
#         fig_name = "Metric_CumulativeRegret" + "_Iteration_" + \
#                    str(RL_info["n_iteration"] * segment)
#
#     elif evaluation_info["metric_type"] == "loss":
#
#         fig_name = "Metric_CumulativeCrossEntropy" + "_Iteration_" + \
#                    str(RL_info["n_iteration"] * segment)
#
#     fig.savefig(path_test + fig_name + ".png")
#
#     plt.clf()
#     plt.close("all")
#
# PlotSolver.bar_arms_result(results_arms=results_arms,
#                            iteration=RL_info["n_iteration"],
#                            behav_name="RndWalk",
#                            segmentation=evaluation_info["segmentation"])