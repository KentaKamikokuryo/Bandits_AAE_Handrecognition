from Classes_RL.BanditsFactory import BanditsFactory
from Classes_RL.Plot_RL import PlotBehaviorNonStaticStrategy
import matplotlib.pyplot as plt

# parameters = {"name": "static"}
parameters = {"name": "sin", "frequency": 2, "amplitude": 10, "std_amplitude": 5}
# parameters = {"name": "unsteady", "n_change": 5, "min_iteration_change": 10}
# parameters = {"name": "log", "range": 20}
# parameters = {"name": "exp", "range": 10, "range_exp": 5}
# parameters = {"name": "RndWalk", "range": 7, "std": 10}

n_iteration = 100
n_action = 5
n_episode = 1

argument = {"parameters": parameters,
            "n_iteration": n_iteration,
            "n_action": n_action}

fac = BanditsFactory(**argument)
bandits = fac.create()

import seaborn as sns
sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.5)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plotter = PlotBehaviorNonStaticStrategy()
fig = plotter.plot(bandits=bandits)
plt.legend(loc=4, frameon=True, fancybox=False, ncol=1, framealpha=0.7, edgecolor="black")