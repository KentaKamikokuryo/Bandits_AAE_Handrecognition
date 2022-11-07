from Classes_RL.Bandit import Bandit
from typing import List
from Classes_RL.Bandit import BehaviorStaticStrategy, BehaviorSinStrategy, BehaviorUnsteadyStrategy, BehaviorLogStrategy, BehaviorExpStrategy, BehaviorRndWalkStrategy, BehaviorRealStrategy
import matplotlib.pyplot as plt

class BanditsFactory():

    def __init__(self, parameters: dict, n_iteration: int, n_action: int):

        """ Generate Bandit. Example of input
            parameters = {"name": "static"}
            parameters = {"name": "sin", "frequency": 2, "amplitude": 1, "std_mpplitude": 5}
            parameters = {"name": "unsteady", "n_change": 5, "min_iteration_change": 10}
            parameters = {"name": "real"}
            parameters = {"name": "log", "range": 20}
            parameters = {"name": "exp", "range": 10, "range_exp": 5}
            parameters = {"name": "RndWalk", "range": 10, "std": 1}
            parameters = {"name": "real, "mu_list": [], "std_list": []}
        """

        self.parameters = parameters
        self.name_behavior = parameters["name"]

        self.n_iteration = n_iteration
        self.n_action = n_action

    def create(self) -> List[Bandit]:

        # factories = {"static": PlotBehaviorStaticStrategy(),
        #              "sin": PlotBehaviorNonStaticStrategy(),
        #              "unsteady": PlotBehaviorNonStaticStrategy(),
        #              "log": PlotBehaviorNonStaticStrategy(),
        #              "exp": PlotBehaviorNonStaticStrategy(),
        #              "RndWalk": PlotBehaviorNonStaticStrategy(),
        #              "real": PlotBehaviorNonStaticStrategy()}
        #
        # self.plot_strategy = factories[self.name_behavior]

        bandits = []

        for i in range(self.n_action):

            if self.name_behavior == "static":

                behavior = BehaviorStaticStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)
                behavior.execute()

            elif self.name_behavior == "sin":

                behavior = BehaviorSinStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)
                behavior.execute()

            elif self.name_behavior == "unsteady":

                behavior = BehaviorUnsteadyStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)
                behavior.execute()

            elif self.name_behavior == "log":

                behavior = BehaviorLogStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)
                behavior.execute()

            elif self.name_behavior == "exp":

                behavior = BehaviorExpStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)
                behavior.execute()

            elif self.name_behavior == "RndWalk":

                behavior = BehaviorRndWalkStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)
                behavior.execute()

            elif self.name_behavior == "real":

                if "mu_list" in self.parameters and "std_list" in self.parameters:

                    mu_list = self.parameters["mu_list"][i, :]
                    std_list = self.parameters["std_list"][i, :]

                    behavior = BehaviorRealStrategy()
                    behavior.set_parameters(parameters=self.parameters, mu=list(mu_list), std=list(std_list))
                    behavior.execute()

                else:

                    print("Bandit from parameters: " + str(self.parameters) + " does not have mu_list or std_list")
                    print("Return None Behavior")
                    behavior = None

            else:

                print("Could not find corresponding Bandit from parameters: " + str(self.parameters))
                print("Return default Bandit (Static)")
                behavior = BehaviorStaticStrategy()
                behavior.set_parameters(parameters=self.parameters, n_iteration=self.n_iteration)

            bandit = Bandit(behavior)
            bandits.append(bandit)

        return bandits

