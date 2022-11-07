from Classes_RL.SolverFactory import Models


class BanditInfo():

    def __init__(self):

        self.behavior_parameters = {}
        self.behavior_parameters["static"] = {"name": "static"}
        self.behavior_parameters["sin"] = {"name": "sin", "amplitude_base": [1, 2], "frequency_range": [1, 3], "std_amplitude": 1}
        self.behavior_parameters["unsteady"] = {"name": "unsteady", "n_change": 2, "min_iteration_change": 50}
        self.behavior_parameters["real"] = {"name": "real"}
        self.behavior_parameters["log"] = {"name": "log", "range": 20}
        self.behavior_parameters["exp"] = {"name": "exp", "range": 10, "range_exp": 5}
        self.behavior_parameters["RndWalk"] = {"name": "RndWalk", "range": 10, "std": 1}

    def get_parameters(self, name: str = "static"):

        if name in self.behavior_parameters:
            return self.behavior_parameters[name]
        else:
            return None

class SolverInfo:

    def __init__(self, n_action: int, n_iteration: int, n_episode: int):

        self.solver_names = [Models.Boltzmann_stationary, Models.Boltzmann_non_stationary,
                             Models.Boltzmann_UCB_stationary, Models.Boltzmann_UCB_non_stationary,

                             Models.Bayesian_um_kv_softmax_stationary, Models.Bayesian_um_kv_softmax_non_stationary,
                             Models.Bayesian_um_uv_softmax_stationary, Models.Bayesian_um_uv_softmax_non_stationary,

                             Models.Bayesian_um_kv_softmax_sliding_non_stationary,
                             Models.Bayesian_um_uv_softmax_sliding_non_stationary,

                             Models.Kalman_greedy,
                             Models.Kalman_e_greedy,
                             Models.Kalman_UCB,
                             Models.Kalman_softmax,
                             Models.Kalman_UCB_softmax,
                             Models.Kalman_Thompson_greedy,
                             Models.Kalman_Thompson_softmax]

        self.solver_names_app = [Models.Boltzmann_stationary, Models.Boltzmann_non_stationary,
                                 Models.Boltzmann_UCB_stationary, Models.Boltzmann_UCB_non_stationary,

                                 Models.Kalman_greedy,
                                 Models.Kalman_e_greedy,
                                 Models.Kalman_UCB,
                                 Models.Kalman_softmax,
                                 Models.Kalman_UCB_softmax,
                                 Models.Kalman_Thompson_greedy,
                                 Models.Kalman_Thompson_softmax]

        self.solver_names = [Models.Boltzmann_UCB_stationary, Models.Boltzmann_UCB_non_stationary,

                             Models.Kalman_greedy,
                             Models.Kalman_e_greedy,
                             Models.Kalman_UCB,
                             Models.Kalman_softmax,
                             Models.Kalman_UCB_softmax,
                             Models.Kalman_Thompson_greedy,
                             Models.Kalman_Thompson_softmax]

        self.solver_names = [Models.Boltzmann_non_stationary,
                             Models.Boltzmann_UCB_non_stationary,

                             Models.Kalman_softmax,
                             Models.Kalman_UCB_softmax,
                             Models.Kalman_Thompson_softmax]

        self.n_action = n_action
        self.n_iteration = n_iteration
        self.n_episode = n_episode

    def get_parameter(self):

        return {"n_action": self.n_action, "n_iteration": self.n_iteration, "n_episode": self.n_episode}
