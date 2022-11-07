import matplotlib.pyplot as plt
# the classes of Reinforcement learning
from Classes_RL.Info_RL import BanditInfo
from Classes_RL.BanditsFactory import BanditsFactory

from Kenta.Thesis_RL import Manager, Saved_results

RL_info = {"n_action": 4,
           "n_iteration": 100,
           "n_episode": 5}

evaluation_info = {"segmentation": [0.1, 0.5, 1],
                   "metric_type": "loss"}

search_info = [True, False]
search_info = [True]

saved_search = Saved_results()
saved_train = Saved_results()

bandit_info = BanditInfo()
behavior_parameters = bandit_info.behavior_parameters
behavior_parameters.pop("real")

############################################################################
behavior_parameters = {}
behavior_parameters["RndWalk"] = {"name": "RndWalk", "range": 10, "std": 1}
############################################################################

for behavior_name in behavior_parameters:
    print("running while behavior names : " + behavior_name)

    # get behavior parameter
    behavior_parameter = bandit_info.get_parameters(behavior_name)

    # create bandits at here, since the all bandits should be same to evaluate
    bandit_fac = BanditsFactory(parameters=behavior_parameter, n_iteration=RL_info["n_iteration"],
                                n_action=RL_info["n_action"])

    bandits = bandit_fac.create()

    for s in search_info:

        manager = Manager(bandits=bandits,
                          RL_info=RL_info,
                          search_info=s,
                          evaluation_info=evaluation_info)

        results, results_mean, oracle = manager.run()

        if s:

            saved_search.add_results(behavior_name=behavior_name, results=results, results_mean=results_mean)
            saved_search.add_oracle(behavior=behavior_name, oracle=oracle)

        else:

            saved_train.add_results(behavior_name=behavior_name, results=results, results_mean=results_mean)
            saved_train.add_oracle(behavior=behavior_name, oracle=oracle)

import seaborn as sns

sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.2)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

results = saved_train.results_behavior
results_mean = saved_train.results_mean_behavior
oracles = saved_train.oracle_behavior

results = results["RndWalk"]
oracle = oracles["RndWalk"]

from Classes_RL.Plot_RL import PlotSolver

plotter = PlotSolver()
# axis = plotter.bar_iterations(results=results, behav_name="RndWalk", iteration=100, segmentation=[0.1, 0.5, 1])
# fig = plotter.plot_gaussian_results_iterations(result=result, oracle=oracle, iteration=100)
plt.legend(loc=2, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")




