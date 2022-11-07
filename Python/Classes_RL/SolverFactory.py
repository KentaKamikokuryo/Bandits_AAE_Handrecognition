from Classes_RL.Boltzmann_solver import Boltzmann, BoltzmannStationary, BoltzmannNonStationary, BoltzmannUCBStationary, BoltzmannUCBNonStationary, BoltzmannSlidingWindowUCB
from Classes_RL.Bayesian_solver import Bayesian, BayesianStationaryUMUV, BayesianNonStationaryUMUV, BayesianStationaryUMKV, BayesianNonStationaryUMKV
from Classes_RL.Bayesian_sliding_solver import BayesianSlidingWindowUMKV, BayesianSlidingWindowUMUV
from Classes_RL.Kalman_filter_solver import Kalman, KalmanGreedy, KalmanEpsilonGreedy, KalmanUCB, KalmanSoftmax, KalmanUCBSoftmax, KalmanThompsonGreedy, KalmanThompsonSoftmax
from Classes_RL.Interfaces import Solver
from Classes_RL.Bandit import Bandit
from typing import List

class Models:

    Boltzmann_stationary = "boltzmann_stationary"  # Not coded
    Boltzmann_UCB_stationary = "boltzmann_UCB_stationary"  # Not coded
    Boltzmann_non_stationary = "boltzmann_non_stationary"  # Not coded
    Boltzmann_UCB_non_stationary = "boltzmann_UCB_non_stationary"  # Not coded

    Boltzmann_UCB_sliding = "boltzmann_UCB_sliding_non_stationary"

    Kalman_greedy = "kalman_greedy"  # Coded
    Kalman_e_greedy = "kalman_e_greedy"  # Coded
    Kalman_UCB = "kalman_UCB"  # Coded
    Kalman_softmax = "kalman_softmax"  # Coded
    Kalman_UCB_softmax = "kalman_UCB_softmax"  # Coded
    Kalman_Thompson_greedy = "kalman_filter_Thompson_argmax"  # Coded
    Kalman_Thompson_softmax = "kalman_filter_Thompson_softmax"  # Coded

    # Bayesian_um_kv_greedy_stationary = "bayesian_um_kv_greedy_stationary"  # Not coded
    # Bayesian_um_uv_greedy_stationary = "bayesian_um_uv_greedy_stationary"  # Not coded
    Bayesian_um_kv_softmax_stationary = "bayesian_um_kv_softmax_stationary"  # Coded
    Bayesian_um_uv_softmax_stationary = "bayesian_um_uv_softmax_stationary"  # Coded

    # Bayesian_um_kv_greedy_non_stationary = "bayesian_um_kv_greedy_non_stationary"  # Not coded
    # Bayesian_um_uv_greedy_non_stationary = "bayesian_um_uv_greedy_non_stationary"  # Not coded
    Bayesian_um_kv_softmax_non_stationary = "bayesian_um_kv_softmax_non_stationary"  # Coded
    Bayesian_um_uv_softmax_non_stationary = "bayesian_um_uv_softmax_non_stationary"  # Coded

    # Bayesian_um_kv_greedy_sliding_non_stationary = "bayesian_um_kv_greedy_sliding_non_stationary"  # Not coded
    # Bayesian_um_uv_greedy_sliding_non_stationary = "bayesian_um_uv_greedy_sliding_non_stationary"  # Not coded
    Bayesian_um_kv_softmax_sliding_non_stationary = "bayesian_um_kv_softmax_sliding_non_stationary"  # Coded
    Bayesian_um_uv_softmax_sliding_non_stationary = "bayesian_um_uv_softmax_sliding_non_stationary"  # Coded

# print(Models.Boltzmann_stationary, Models.Boltzmann_UCB_stationary, Models.Boltzmann_non_stationary, Models.Boltzmann_UCB_non_stationary,
#       Models.Kalman_greedy, Models.Kalman_e_greedy, Models.Kalman_UCB, Models.Kalman_softmax, Models.Kalman_UCB_softmax, Models.Kalman_Thompson_greedy, Models.Kalman_Thompson_softmax,
#       Models.Bayesian_um_kv_greedy_stationary, Models.Bayesian_um_uv_greedy_stationary, Models.Bayesian_um_kv_softmax_stationary, Models.Bayesian_um_uv_softmax_stationary,
#       Models.Bayesian_um_kv_greedy_non_stationary, Models.Bayesian_um_uv_greedy_non_stationary, Models.Bayesian_um_kv_softmax_non_stationary, Models.Bayesian_um_uv_softmax_non_stationary,
#       Models.Bayesian_um_kv_greedy_sliding_non_stationary, Models.Bayesian_um_uv_greedy_sliding_non_stationary, Models.Bayesian_um_kv_softmax_sliding_non_stationary, Models.Bayesian_um_uv_softmax_sliding_non_stationary)

class SolverFactory():

    def create(self, bandits: List[Bandit], oracle_data: dict, n_episode: int, hyper_model: dict) -> Solver:

        # print("SolverFactory: " + str(hyper_model))
        solver_name = hyper_model["solver_name"]
        # print(solver_name)

        # print(solver_name == Models.Boltzmann_non_stationary)

        arguments = {"bandits": bandits,
                     "oracle_data": oracle_data,
                     "n_iteration": bandits[0].get_n_iteration(),
                     "n_episode": n_episode}

        # ------- Boltzmann -------

        if solver_name == Models.Boltzmann_stationary:

            solver = BoltzmannStationary(**arguments)

        elif solver_name == Models.Boltzmann_UCB_stationary:

            solver = BoltzmannUCBStationary(**arguments)

        elif solver_name == Models.Boltzmann_non_stationary:

            solver = BoltzmannNonStationary(**arguments)

        elif solver_name == Models.Boltzmann_UCB_non_stationary:

            solver = BoltzmannUCBNonStationary(**arguments)

        # -------- Boltzmann - Sliding window -------

        elif solver_name == Models.Boltzmann_UCB_sliding:

            solver = BoltzmannSlidingWindowUCB(**arguments)

        # --------- Kalman -------

        elif solver_name == Models.Kalman_greedy:

            solver = KalmanGreedy(**arguments)

        elif solver_name == Models.Kalman_e_greedy:

            solver = KalmanEpsilonGreedy(**arguments)

        elif solver_name == Models.Kalman_UCB:

            solver = KalmanUCB(**arguments)

        elif solver_name == Models.Kalman_softmax:

            solver = KalmanSoftmax(**arguments)

        elif solver_name == Models.Kalman_UCB_softmax:

            solver = KalmanUCBSoftmax(**arguments)

        elif solver_name == Models.Kalman_Thompson_greedy:

            solver = KalmanThompsonGreedy(**arguments)

        elif solver_name == Models.Kalman_Thompson_softmax:

            solver = KalmanThompsonSoftmax(**arguments)

        # ------- Bayesian (Thompson Sampling) --------

        elif solver_name == Models.Bayesian_um_kv_softmax_stationary:

            solver = BayesianStationaryUMKV(**arguments)

        elif solver_name == Models.Bayesian_um_uv_softmax_stationary:

            solver = BayesianStationaryUMUV(**arguments)

        elif solver_name == Models.Bayesian_um_kv_softmax_non_stationary:

            solver = BayesianNonStationaryUMKV(**arguments)

        elif solver_name == Models.Bayesian_um_uv_softmax_non_stationary:

            solver = BayesianNonStationaryUMUV(**arguments)

        # ------- Bayesian - Sliding window -------

        elif solver_name == Models.Bayesian_um_kv_softmax_sliding_non_stationary:

            solver = BayesianSlidingWindowUMKV(**arguments)

        elif solver_name == Models.Bayesian_um_uv_softmax_sliding_non_stationary:

            solver = BayesianSlidingWindowUMUV(**arguments)

        else:

            solver = None

        solver.set_hyperparameters(hyper_model)

        return solver