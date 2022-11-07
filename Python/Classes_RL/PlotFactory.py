from Classes_RL.Interfaces import IPlotBehavior
from typing import Tuple, List
from Classes_RL.Bandit import Bandit
from Classes_RL.Plot_RL import PlotResults
from Classes_RL.Plot_RL import PlotBehaviorStaticStrategy, PlotBehaviorNonStaticStrategy

# class PlotResultsFactory():
##
#     def create(self, solver_name):
#
#         if "boltzmann" in solver_name:
#
#             plot_results_strategy = PlotResults()
#
#         elif "bayesian" in solver_name or "kalman" in solver_name or "sliding" in solver_name:
#
#             plot_results_strategy = ProbabilisticPlotResults()
#
#         else:
#
#             plot_results_strategy = None
#
#         return plot_results_strategy

class PlotBehaviorFactory():

    plot_behavior_strategy: IPlotBehavior

    def create(self, parameters) -> IPlotBehavior:

        # Get plot behavior
        if parameters["name"] == "static":

            plot_behavior_strategy = PlotBehaviorStaticStrategy()

        elif parameters["name"] == "sin" or parameters["name"] == "unsteady" or parameters["name"] == "real":

            plot_behavior_strategy = PlotBehaviorNonStaticStrategy()

        elif parameters["name"] == "log" or parameters["name"] == "exp" or parameters["name"] == "RndWalk":

            plot_behavior_strategy = PlotBehaviorNonStaticStrategy()

        else:

            plot_behavior_strategy = None
            print("Nothing (Behavior) is being referenced.")

        return plot_behavior_strategy