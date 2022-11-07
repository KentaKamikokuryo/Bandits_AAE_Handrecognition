from Classes_RL.SolverFactory import Models


class Dedicater_name():

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
            name = "Kalman greedy"

        elif models_enum_name == Models.Kalman_softmax:
            name = "Kalman"

        elif models_enum_name == Models.Kalman_UCB:
            name = "Kalman greedy UCB"

        elif models_enum_name == Models.Kalman_UCB_softmax:
            name = "Kalman UCB"

        elif models_enum_name == Models.Kalman_Thompson_greedy:
            name = "Kalman greedy TS"

        elif models_enum_name == Models.Kalman_Thompson_softmax:
            name = "Kalman TS"

        elif models_enum_name == Models.Kalman_e_greedy:
            name = "Kalman e-greedy"

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