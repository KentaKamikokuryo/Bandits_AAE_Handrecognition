import numpy as np
from Classes_ML.Bandit import Bandit
from Classes_ML.Plot import Plot_bandit
from Classes_data.Info import Bandit_info
from Classes_data.Info import Path_info
from Classes_RL.Boltzmann_solver import Bayesian_exploration, Boltzmann_exploration
from Classes_ML.Hyperparameters import Hyperparamaters_bandits

# Path
path_info = Path_info()

# Bandit information
bandit_info = Bandit_info()

# Parameters
model_names = bandit_info.model_names[4:]
train_parameters = bandit_info.train_parameters
behavior_parameters = bandit_info.behavior_parameters[:2]

train_parameter = train_parameters[0]
behavior_parameter = behavior_parameters[0]
model_name = model_names[0]

# Create the bandits
bandits = [Bandit(n_iteration=train_parameter["n_iteration"]) for i in range(train_parameter["n_action"])]

for behavior_parameter in behavior_parameters:

    for bandit in bandits:
        bandit.set_behavior(parameters=behavior_parameter)

    # bandits[1].change_behavior_mu(new_mu=10)
    # bandits[1].change_behavior_std(new_std=2)

    # bandits[2].change_behavior_mu(new_mu=10)
    # bandits[2].change_behavior_std(new_std=1)

    for model_name in model_names:

        if "boltzmann" in model_name:

            solver = Boltzmann_exploration(bandits=bandits, train_parameter=train_parameter)

        elif "bayesian" in model_name:

            solver = Bayesian_exploration(bandits=bandits, train_parameter=train_parameter)

        Plot_bandit.plot_behavior(bandits=bandits, path_folder_figure=path_info.path_folder_figure_bandit_bayesian)
        Plot_bandit.plot_action_wanted(bandits=bandits, path_folder_figure=path_info.path_folder_figure_bandit_bayesian)

        hyperparameters_list = Hyperparamaters_bandits.generate_hyperparameters(model_name=model_name, display_info=True)
        hyperparameters = hyperparameters_list[0]

        dict_r_iteration_sm = {}
        dict_Q_sm = {}
        dict_Q_mean_sm = {}
        dict_action_sm = {}
        dict_regret_sm = {}

        dict_r_iteration_ts = {}
        dict_Q_ts = {}
        dict_Q_mean_ts = {}
        dict_action_ts = {}
        dict_regret_ts = {}

        dict_mu_bar = {}
        dict_tau_bar = {}
        dict_var_bar = {}
        dict_var_bar_known = {}

        for hyperparameters in hyperparameters_list:

            # TODO: Kenta - change bayesian_um_uv_gamma top simply bayesian_um_uv

            # TODO: it may be wrong to use something like self.Q[a] += self.alpha * (self.r - self.Q[a]) with Bayesian (I did it too don't worry haha)
            # TODO: indeed, here of course it will track the Q value properly if it is non-stationary, but at the same time we continue to improve the precision
            #  which contradict a bit the sampling. I am not sure if what I did is good or not.
            #  We will see more in detail later when you will start working on specific non-stationary Bayesian bandit (Maybe they use a way to update the precision differently if the Q value is changing)

            solver.set_hyperparameters(hyperparameters)

            if "boltzmann" in model_name:

                r_iteration_all, Q_all, Q_mean_all, action_all, regrets_all = solver.run_search()

                # TODO: Kenta - Add here the information that need to be kept for Boltzmann
                dict_r_iteration_sm[solver.name] = np.mean(r_iteration_all, axis=0)
                dict_Q_sm[solver.name] = np.mean(Q_all, axis=0)
                dict_Q_mean_sm[solver.name] = np.mean(Q_mean_all, axis=0)
                dict_action_sm[solver.name] = np.mean(action_all, axis=0) / np.sum(np.mean(action_all, axis=0))
                dict_regret_sm[solver.name] = np.mean(regrets_all, axis=0)

                Plot_bandit.plot_results(bandits=solver.bandits,
                                         dict_r_iteration=dict_r_iteration_sm, dict_Q_mean=dict_Q_mean_sm,
                                         dict_action=dict_action_sm, dict_regret=dict_regret_sm,
                                         model_name=model_name, path_folder_figure=path_info.path_folder_figure_bandit)

            elif "bayesian" in model_name:

                r_iteration_all, Q_all, Q_mean_all, action_all, regrets_all, mu_bar_all, tau_bar_all, var_bar_all, var_bar_known_all = solver.run_search()

                # Plot animation
                # if model_name == "bayesian_um_kv":
                #     Plot_bandit.plot_animation_gaussian_arm(bandits, mu_bar_all, var_bar_all, var_bar_known_all,
                #                                             is_unknown_variance=False, gif_name=behavior_parameter['behavior'] + "_" + model_name,
                #                                             path_folder_figure=path_info.path_folder_figure_test_bandit_bayesian_best)

                dict_r_iteration_ts[solver.name] = np.mean(r_iteration_all, axis=0)
                dict_Q_ts[solver.name] = np.mean(Q_all, axis=0)
                dict_Q_mean_ts[solver.name] = np.mean(Q_mean_all, axis=0)
                dict_action_ts[solver.name] = np.mean(action_all, axis=0) / np.sum(np.mean(action_all, axis=0))
                dict_regret_ts[solver.name] = np.mean(regrets_all, axis=0)

                dict_mu_bar[solver.name] = np.mean(mu_bar_all[:, -1, :], axis=0)
                dict_tau_bar[solver.name] = np.mean(tau_bar_all[:, -1, :], axis=0)
                dict_var_bar[solver.name] = np.mean(var_bar_all[:, -1, :], axis=0)
                dict_var_bar_known[solver.name] = np.mean(var_bar_known_all[:, -1, :], axis=0)

                Plot_bandit.plot_final_gaussian_results(bandits=bandits, mu_bar_all=mu_bar_all, var_bar_all=var_bar_all,
                                                        model_name=model_name,
                                                        path_folder_figure=path_info.path_folder_figure_bandit_bayesian)

                Plot_bandit.plot_results(bandits=solver.bandits,
                                         dict_r_iteration=dict_r_iteration_ts, dict_Q_mean=dict_Q_mean_ts,
                                         dict_action=dict_action_ts, dict_regret=dict_regret_ts,
                                         model_name=model_name, path_folder_figure=path_info.path_folder_figure_bandit_bayesian)
