import os, itertools
import pandas as pd
from tabulate import tabulate
import numpy as np

from Classes_ML.Model_ML import Model_ML
from Classes_ML.Model_DL import Model_DL
from Classes_ML.Hyperparameters import Hyperparameters_ML, Hyperparameters_DL
from Classes_ML.Prior import PriorDistribution
from Classes_ML.Distribution import Normal_distribution_2D
from Classes_ML.Latent_space import Latent_space, LatentSpaceUtilities

from Classes_data.Info import DB_info
from Classes_data.Data_ML import Data, Data_Z

from Classes_results.Ranked import Ranked
from Classes_results.Genetic import Genetic_algorithm

class Evaluate():

    def __init__(self, Z_transform_K: dict, y_transform_encoded: list, loss_dict_K: dict, prior: dict, metric_type: str, model_name: str):

        self.Z_transform_K = Z_transform_K
        self.y_transform_encoded = y_transform_encoded
        self.loss_dict_K = loss_dict_K
        self.prior = prior
        self.metric_type = metric_type
        self.model_name = model_name

        self.n_K_fold = len(list(Z_transform_K.keys()))

    def compute_metrics(self):

        if self.metric_type == "KL":

            try:

                KL_dict_K = {K: {} for K in self.Z_transform_K.keys()}

                for K in self.Z_transform_K.keys():

                    Q_Gaussian_2D_dict_K = Latent_space.compute_2D_Gaussian(z=self.Z_transform_K[K], y=self.y_transform_encoded)

                    if self.model_name == "UAAE":
                        Q_Gaussian_2D_dict_K = self.identify_cluster(Q_Gaussian_2D_dict_K)

                    for unique in range(self.prior["size"]):

                        variances = np.array([self.prior["x_std"] ** 2, self.prior["y_std"] ** 2])
                        theta = 2.0 * np.pi / float(self.prior["size"]) * float(unique)

                        P_mean_vector = PriorDistribution.get_centroid(n_labels=self.prior["size"], label=unique, shift=self.prior["shift"])
                        P_cov_matrix = PriorDistribution.compute_cov_from_eigen_theta(eigen_values=variances, theta=theta)
                        P_distribution = Normal_distribution_2D(mean_vector=P_mean_vector, cov_matrix=P_cov_matrix)

                        Q_mean_vector = Q_Gaussian_2D_dict_K[unique]["mean"]
                        Q_cov_matrix = Q_Gaussian_2D_dict_K[unique]["cov"]

                        Q_distribution = Normal_distribution_2D(mean_vector=Q_mean_vector, cov_matrix=Q_cov_matrix)

                        KL_dict_K[K][unique] = LatentSpaceUtilities.compute_KL_divergence(P_distribution=P_distribution, Q_distribution=Q_distribution)

                R_loss_K = {K: self.loss_dict_K[K]["AE_metric"] for K in self.loss_dict_K.keys()}

                self.metrics_transform_K = np.array([Evaluate.compute_AAE_metric(KL_dict=KL_dict_K[K], R_loss=R_loss_K[K], R_weight=100) for K in self.loss_dict_K.keys()])
                self.metric_transform_mean = np.mean(self.metrics_transform_K)
                self.metric_transform_sd = np.std(self.metrics_transform_K)

                # Add KL to the dictionary
                for i, K in enumerate(self.Z_transform_K.keys()):
                    self.loss_dict_K[K]["KL"] = np.array(list(KL_dict_K[K].values())).mean()

            except:

                self.metrics_transform_K = np.array([99999 for K in self.Z_transform_K.keys()])
                self.metric_transform_mean = np.mean(self.metrics_transform_K)
                self.metric_transform_sd = np.std(self.metrics_transform_K)

                # Add KL to the dictionary
                for i, K in enumerate(self.Z_transform_K.keys()):
                    self.loss_dict_K[K]["KL"] = 99999

        elif self.metric_type == "Recons":

            self.metrics_transform_K = np.array([self.loss_dict_K[K]["AE_metric"] for K in self.loss_dict_K.keys()])
            self.metric_transform_mean = np.mean(self.metrics_transform_K)
            self.metric_transform_sd = np.std(self.metrics_transform_K)

        else:

            self.metrics_transform_K = np.array([99999 for K in self.Z_transform_K.keys()])
            self.metric_transform_mean = np.mean(self.metrics_transform_K)
            self.metric_transform_sd = np.std(self.metrics_transform_K)

        # Create a dictionary with mean and std value for all loss
        loss_dict_mean = {}
        loss_keys = self.loss_dict_K[list(self.loss_dict_K.keys())[0]].keys()
        for k in loss_keys:
            mean = np.mean([self.loss_dict_K[K][k] for K in self.loss_dict_K.keys()])
            std = np.std([self.loss_dict_K[K][k] for K in self.loss_dict_K.keys()])
            loss_dict_mean[k + "_mean"] = mean
            loss_dict_mean[k + "_std"] = std

    @staticmethod
    def compute_AAE_metric(KL_dict: dict, R_loss, R_weight):

        KL_mean = np.array(list(KL_dict.values())).mean()
        metric = KL_mean + R_weight * R_loss

        return metric

    def identify_cluster(self, Q_Gaussian_2D_dict_K: dict):

        sequence = list(np.arange(self.prior["size"]))
        permutations = list(itertools.permutations(sequence))

        min_q_index = 0
        min_distance = float('inf')

        for i, permutation in enumerate(permutations):

            sum_distance = 0.0

            for p_index, q_index in enumerate(permutation):

                sum_distance += np.linalg.norm(Q_Gaussian_2D_dict_K[q_index]["mean"] - PriorDistribution.get_centroid(n_labels=self.prior["size"], label=p_index, shift=self.prior["shift"]))

            if sum_distance < min_distance:

                min_q_index = i
                min_distance = sum_distance

        min_permutation = permutations[min_q_index]

        new_Q_Gaussian_2D_dict_K = {}

        for i in range(self.prior["size"]):

            new_Q_Gaussian_2D_dict_K[min_permutation[i]] = Q_Gaussian_2D_dict_K[i]

        return new_Q_Gaussian_2D_dict_K

class Manager:

    def __init__(self, DB_N: int, model_name: str):

        self.DB_N = DB_N
        self.model_name = model_name

        self.model_ML_names = ["PCA", "T_SNE", "ICA", "ISO", "LDA", "Laplacian"]
        self.model_DL_names = ["AE", "VAE", "UAAE", "SSAAE"]

        if model_name in self.model_ML_names:
            self.is_DL = False
        else:
            self.is_DL = True

        self.db_info = DB_info()
        self.db_info.get_DB_info(DB_N=self.DB_N)

        self.data = Data(db_info=self.db_info)
        self.data.get_DB_N(is_DL=self.is_DL, reshape_to_1D=True)

        self.data_Z = Data_Z(self.data, model_name=self.model_name)
        
        self.data.n_K_fold = 1

        self.max_epoch_search = 100
        self.max_epoch_final = 100
        self.prior = {'size': self.data.n_label, 'x_std': 1.0, 'y_std': 0.6, 'shift': 4.0}

        # import matplotlib
        # matplotlib.use('Qt5Agg')
        # import matplotlib.pyplot as plt
        # z_real = PriorDistribution.generate_z_un_N(batch_size=10000, n_labels=self.prior["size"], x_std=self.prior["x_std"], y_std=self.prior["y_std"], shift=self.prior["shift"])
        # gif = plt.figure(figsize=(16, 10))
        # plt.scatter(z_real[:, 0], z_real[:, 1])

    def set_interface(self, interface_dict):

        self.hyper_model_search = interface_dict["hyper_model_search"]
        self.train_final_model = interface_dict["train_final_model"]
        self.perform_analysis = interface_dict["perform_analysis"]

        self.hyper_model_search_type = interface_dict["hyper_model_search_type"]
        self.metric_type = interface_dict["metric_type"]

    def set_hyper_model_search(self):

        if self.is_DL:

            self.hyperparameters = Hyperparameters_DL(model_name=self.model_name)
            self.hyperparameters_all_combination, self.hyperparameters_choices = self.hyperparameters.generate_hyperparameters(n_label=self.data.n_label, max_epoch=self.max_epoch_search, display_info=True)

            self.path_figure = self.db_info.path_search_models_DL_DB_N + self.model_name + "\\"
            self.path_search = self.db_info.path_search_models_DL_DB_N + self.model_name + "\\"

        else:

            self.hyperparameters = Hyperparameters_ML(model_name=self.model_name)
            self.hyperparameters_all_combination, self.hyperparameters_choices = self.hyperparameters.generate_hyperparameters(display_info=True)

            self.path_figure = self.db_info.path_search_models_ML_DB_N + self.model_name + "\\"
            self.path_search = self.db_info.path_search_models_ML_DB_N + self.model_name + "\\"

        if not (os.path.exists(self.path_figure)):
            os.makedirs(self.path_figure)
        if not (os.path.exists(self.path_search)):
            os.makedirs(self.path_search)

        print("Search results will be saved at " + self.path_search)
        print("Search figures will be saved at " + self.path_figure)

        # Ranked
        self.ranked = Ranked(model_name=self.model_name,
                             search_type=self.hyper_model_search_type,
                             path=self.path_search,
                             metric_order="descending",
                             metric_name="loss")

        if "AAE" in self.model_name:

            # Search with genetic algorithm
            self.n_agent_generation = 20
            self.n_generation = 1  # One generation mean random search

        else:

            self.n_agent_generation = 10
            self.n_generation = 1  # One generation mean random search

        self.GA = Genetic_algorithm(n_generation=self.n_generation, n_agent_generation=self.n_agent_generation, nn_param_choices=self.hyperparameters_choices, display_info=True)
        selection_parameters = dict(name="tournament", metric_order="descending", mutate_chance=0.05, n_agent_fight=2, keep=0.2, keep_best=0.05)
        self.GA.set_selection_methods(selection_parameters=selection_parameters)

    def set_hyper_model_final(self):

        if self.is_DL:

            self.path_figure = self.db_info.path_figure_DL_DB_N + self.model_name + "\\"
            self.path_search = self.db_info.path_search_models_DL_DB_N + self.model_name + "\\"
            self.path_results = self.db_info.path_results_DL_DB_N

        else:

            self.path_figure = self.db_info.path_figure_ML_DB_N + self.model_name + "\\"
            self.path_search = self.db_info.path_search_models_ML_DB_N + self.model_name + "\\"
            self.path_results = self.db_info.path_results_ML_DB_N

        if not (os.path.exists(self.path_figure)):
            os.makedirs(self.path_figure)
        if not (os.path.exists(self.path_search)):
            os.makedirs(self.path_search)

        print("Search results will be loaded from " + self.path_search)
        print("Results will be saved at " + self.path_results)
        print("Figures will be saved at " + self.path_figure)

        if self.model_name == "UAAE":

            self.hyper_model_best = {'input_dim': (64,),
                                     'latent_dim': [2],
                                     'n_neighbors_latent_space': 10,
                                     'encoder_activation': 'relu',
                                     'encoder_units': [128, 64],
                                     'discriminator_units': [512, 512],
                                     'discriminator_activation': 'sigmoid',
                                     'kernel_regularizer_name': 'none',
                                     'kernel_initializer_info': {'name': 'glorot_uniform'},
                                     'use_batch_normalization': False,
                                     'learning_rate_schedule': {'method': 'none', 'learning_rate_start': 0.0001},
                                     'prior': self.prior,
                                     'dropout_rate': 0.0,
                                     'optimizer_name': 'adam',
                                     'loss_function': {'name': 'AAE_loss'},
                                     'metric_function': {'name': 'rmse'},
                                     'batch_size': 64,
                                     'max_epoch': self.max_epoch_final,
                                     'model_name': self.model_name}

        elif self.model_name == "SSAAE":

            self.hyper_model_best = {'input_dim': (64,),
                                     'latent_dim': [2],
                                     'n_neighbors_latent_space': 10,
                                     'encoder_activation': 'relu',
                                     'encoder_units': [128, 64],
                                     'discriminator_units': [512, 512],
                                     'discriminator_activation': 'sigmoid',
                                     'kernel_regularizer_name': 'none',
                                     'kernel_initializer_info': {'name': 'glorot_uniform'},
                                     'use_batch_normalization': False,
                                     'learning_rate_schedule': {'method': 'none', 'learning_rate_start': 0.0001},
                                     'prior': self.prior,
                                     'dropout_rate': 0.0,
                                     'optimizer_name': 'adam',
                                     'loss_function': {'name': 'AAE_loss'},
                                     'metric_function': {'name': 'rmse'},
                                     'batch_size': 64,
                                     'max_epoch': self.max_epoch_final,
                                     'model_name': self.model_name}

        else:

            # Loaded results from search
            self.ranked = Ranked(model_name=self.model_name,
                                 search_type=self.hyper_model_search_type,
                                 path=self.path_search,
                                 metric_order="descending",
                                 metric_name="loss")

            self.ranked.load_ranked_metric()
            self.ranked.display_loaded_ranked_metric()
            self.ranked.load_best_hyperparameters()

            self.hyper_model_best = self.ranked.hyperparameter_best
            self.hyper_model_list_sorted = self.ranked.hyperparameters_list_sorted

            # # Plot to check relationship between KL and other metrics
            import matplotlib.pyplot as plt
            KL_metrics = self.hyper_model_list_sorted["KL"].values
            AE_metric = self.hyper_model_list_sorted["AE_metric"].values
            D_metric = self.hyper_model_list_sorted["D_metric"].values
            G_metric = self.hyper_model_list_sorted["G_metric"].values

            plt.scatter(KL_metrics, AE_metric, color="b")
            plt.scatter(KL_metrics, D_metric, color="r")
            plt.scatter(KL_metrics, G_metric, color="g")

    def create_model(self, hyper_model):

        # Generate the model class here from Model_ML or Model_DL class
        if self.is_DL:
            model_class = Model_DL(hyper_model)
        else:
            model_class = Model_ML(hyper_model)

        self.model_class = model_class
        self.model_class.show_config()

    def fit_transform(self, is_test: bool, save_model: bool):

        """
        Used to train and valid models. We do not save the models here. Only get the accuracy results to rank the models

        Input
        -------
        is_test: bool
            if True: train_valid and test
            if False: train and valid
        save_model: bool
            save the models or not

        Returns
        -------
        Z_fit_K: dict

        Z_transform_K: dict

        metric_transform_mean: float
            mean value of metrics for K fold cross validation
        metric_transform_sd: float
            standard deviation value of metrics for K fold cross validation
        metrics_transform_K: dict
             metrics_K with K values representing the metrics for K fold cross validation
        loss_dict: dict

        """

        Z_fit_K = {}
        Z_transform_K = {}
        loss_dict_K = {}

        # Loop n_K_fold
        for K in range(self.data.n_K_fold):

            self.model_class.show_config()

            X_fit, X_transform, _, _, y_fit_encoded, y_transform_encoded, _, _ = self.data.get_fit_transform_data(K=K, is_test=is_test)

            if self.model_name == "T_SNE" or self.model_name == "MDS" or self.model_name == "Laplacian":

                Z_fit_K[K], Z_transform_K[K], z = self.model_class.fit_transform(X_fit, y_fit_encoded, X_transform, y_transform_encoded)

            elif "AAE" in self.model_name:

                Z_fit_K[K] = self.model_class.fit(X_fit, y_fit_encoded, X_transform, y_transform_encoded)
                Z_transform_K[K] = self.model_class.transform(X_transform, y_transform_encoded)
                loss_dict_K[K] = self.model_class.loss_dict

            else:

                Z_fit_K[K] = self.model_class.fit(X_fit, y_fit_encoded, X_transform, y_transform_encoded)
                Z_transform_K[K] = self.model_class.transform(X_transform, y_transform_encoded)

            if save_model:
                self.model_class.save_model(K, path_model=self.data.path_model_DB_N)
                # self.model_class.save_best_nn(self.model_class.hyper_model["n_neighbors_latent_space"], K, path_model=self.data.path_model_DB_N)

        # Metrics
        if "AAE" in self.model_name:

            evaluate = Evaluate(Z_transform_K, y_transform_encoded, loss_dict_K, prior=self.prior, metric_type="KL", model_name=self.model_name)
            evaluate.compute_metrics()

            metric_transform_mean = evaluate.metric_transform_mean
            metric_transform_sd = evaluate.metric_transform_sd
            metrics_transform_K = evaluate.metrics_transform_K
            loss_dict_K = evaluate.loss_dict_K

        else:

            metrics_transform_K = self.data_Z.evaluate_latent_space(Z_fit_K=Z_fit_K, Z_transform_K=Z_transform_K, metric_type=self.metric_type, is_test=is_test)
            metric_transform_mean = np.array(list(metrics_transform_K.values())).mean()
            metric_transform_sd = np.array(list(metrics_transform_K.values())).std()
            print("metrics = " + str(metric_transform_mean) + "(" + str(metric_transform_sd) + ")")

        return Z_fit_K, Z_transform_K, metric_transform_mean, metric_transform_sd, metrics_transform_K, loss_dict_K

    def fit_plot(self, Z_fit_K: dict, Z_transform_K: dict, is_test: bool, index: str = ""):

        latent_space_K, Z_fit_trajectories_K, Z_transform_trajectories_K = self.data_Z.latent_analysis(Z_fit_K=Z_fit_K, Z_transform_K=Z_transform_K,
                                                                                                       is_test=is_test, do_standardized=False)

        latent_space_std_K, Z_fit_trajectories_std_K, Z_transform_trajectories_std_K = self.data_Z.latent_analysis(Z_fit_K=Z_fit_K, Z_transform_K=Z_transform_K,
                                                                                                                   is_test=is_test, do_standardized=True)

        # Plot latent space
        K_plot = 0

        if self.hyper_model_search:
            figure_name = str(index) + "_" + self.model_name
        elif self.train_final_model:
            figure_name = self.model_name
        else:
            figure_name = self.model_name

        figure_title = self.model_name

        self.data_Z.plot_latent_space(latent_space=latent_space_K[K_plot],
                                      path_figure=self.path_figure,
                                      figure_name=figure_name,
                                      figure_title=figure_title)

        if "AAE" not in self.model_name:

            # Plot standardized latent space (Each label has its own subplot)
            self.data_Z.plot_standardized_latent_gradient(latent_space=latent_space_std_K[K_plot],
                                                          K_plot=K_plot,
                                                          path_figure=self.path_figure,
                                                          figure_name=figure_name,
                                                          figure_title=figure_title)

        if self.perform_analysis:

            self.data_Z.plot_latent_gradient(latent_space=latent_space_K[K_plot],
                                             K_plot=K_plot,
                                             path_figure=self.path_figure,
                                             figure_name=figure_name,
                                             figure_title=figure_title)

            self.data_Z.plot_latent_trajectories(latent_space=latent_space_K[K_plot],
                                                 Z_transform_trajectories=Z_transform_trajectories_K[K_plot],
                                                 K_plot=K_plot,
                                                 path_figure=self.path_figure,
                                                 figure_name=figure_name,
                                                 figure_title=figure_title)

            self.data_Z.plot_standardized_latent_trajectories(latent_space=latent_space_std_K[K_plot],
                                                              Z_transform_trajectories=Z_transform_trajectories_std_K[K_plot],
                                                              K_plot=K_plot,
                                                              path_figure=self.path_figure,
                                                              figure_name=figure_name,
                                                              figure_title=figure_title)

    def run_search(self):

        self.set_hyper_model_search()

        hyperparameters_gen = self.GA.generate_first_population()

        id = 0

        for gen in range(self.GA.n_generation):

            metrics_mean_list, metrics_sd_list = [], []
            print("Current generation: DB_N " + str(self.DB_N) + " - " + str(self.GA.current_generation + 1) + "/" + str(self.GA.n_generation) + " generation - Model: " + self.model_name)

            for i, hyper_model in enumerate(hyperparameters_gen):

                print("Current search: DB_N " + str(self.DB_N) + " - " + str(self.GA.current_generation + 1) + "/" + str(self.GA.n_generation) + " - Agent " + str(i + 1) + "/" + str(self.GA.n_agent_generation))
                df = pd.DataFrame.from_dict([hyper_model])
                print(tabulate(df, headers='keys', tablefmt='psql'))

                self.create_model(hyper_model=hyper_model)

                Z_fit_K, Z_transform_K, metric_valid_mean, metric_valid_sd, metric_valid_K, loss_dict_K = self.fit_transform(is_test=False, save_model=False)

                # TODO: need to get a loss_dict with a mean values of K (Later)
                if "AAE" in self.model_name:
                    hyper_model.update(loss_dict_K[0])
                else:
                    pass

                self.ranked.add(hyper_model, metric_valid_mean, metric_valid_sd, count_params=0, id=str(gen) + "_" + str(i))

                metrics_mean_list.append(metric_valid_mean)
                metrics_sd_list.append(metric_valid_sd)

                # Plot latent space
                self.fit_plot(Z_fit_K=Z_fit_K, Z_transform_K=Z_transform_K, is_test=False, index=str(gen) + "_" + str(i))

                id += 1

            # Generate new generation
            self.GA.keep_generation_information(metrics_mean_list=metrics_mean_list, metrics_sd_list=metrics_sd_list)
            hyperparameters_gen = self.GA.generate_ith_generation(metrics_mean_list=metrics_mean_list, metrics_sd_list=metrics_sd_list)

        for gen in range(self.GA.n_generation):
            self.GA.display_generation_info(generation_n=gen)

        # Ranked metrics
        self.ranked.ranked()
        self.ranked.display_ranked_metric()
        self.ranked.save_ranked_metric()
        self.ranked.save_best_hyperparameter()

    def run_final(self):

        self.set_hyper_model_final()
        self.create_model(hyper_model=self.hyper_model_best)

        Z_fit_K, Z_transform_K, metric_test_mean, metric_test_sd, metric_test_K, loss_dict_K = self.fit_transform(is_test=True, save_model=True)

        self.data_Z.save_Z(Z_fit_K, Z_transform_K, metric_test_mean, metric_test_sd, loss_dict_K, path=self.path_results)

    def run_analysis(self):

        self.set_hyper_model_final()
        self.create_model(hyper_model=self.hyper_model_best)
        self.model_class.load_model(self.data.n_K_fold, path_model=self.data.path_model_DB_N)

        # Load data
        Z_fit, Z_transform, metric_valid_mean, metric_valid_sd, loss_dict = self.data_Z.load_Z(path=self.path_results)

        self.fit_plot(Z_fit_K=Z_fit, Z_transform_K=Z_transform, is_test=True)

        # Save final trajectories for RL algorythm
        latent_space, Z_fit_trajectories, Z_transform_trajectories = self.data_Z.latent_analysis(Z_fit_K=Z_fit, Z_transform_K=Z_transform, is_test=True, do_standardized=False)
        latent_space_std, Z_fit_trajectories_std, Z_transform_trajectories_std = self.data_Z.latent_analysis(Z_fit_K=Z_fit, Z_transform_K=Z_transform, is_test=True, do_standardized=True)

        # Combine dictionary
        self.data_Z.save_latent_analysis(latent_space=latent_space, Z_fit_trajectories=Z_fit_trajectories, Z_transform_trajectories=Z_transform_trajectories, path=self.path_results, is_standardized=False)
        self.data_Z.save_latent_analysis(latent_space=latent_space_std, Z_fit_trajectories=Z_fit_trajectories_std, Z_transform_trajectories=Z_transform_trajectories_std, path=self.path_results, is_standardized=True)

        Z_fit_trajectories, Z_transform_trajectories, Z_centroids_dict = self.data_Z.load_latent_analysis(path=self.path_results, is_standardized=False)
        Z_fit_trajectories_std, Z_transform_trajectories_std, Z_centroids_dict_std = self.data_Z.load_latent_analysis(path=self.path_results, is_standardized=True)

        self.data_Z.save_latent_df(Z_fit_trajectories, Z_transform_trajectories, Z_fit_trajectories_std, Z_transform_trajectories_std, K=0, path=self.path_results)
        df_fit, df_transform, image_fit, image_transform = self.data_Z.load_latent_df(path=self.path_results, K=0)

    def run(self):

        if self.hyper_model_search:

            self.run_search()

        elif self.train_final_model:

            self.run_final()

        elif self.perform_analysis:

            self.run_analysis()

code = 2

model_ML_names = ["PCA", "T_SNE", "ICA", "ISO", "LDA", "Laplacian"]
model_DL_names = ["AE", "VAE", "UAAE", "SSAAE"]

db_info = DB_info()

if code == 0:

    import Set_tf_0
    Is = [0]
    DB_Ns = [i for i in range(len(db_info.DB_N_configurations))]
    model_names = model_DL_names[2:4]

elif code == 1:

    import Set_tf_1
    Is = [1]
    DB_Ns = [i for i in range(len(db_info.DB_N_configurations))]
    model_names = model_DL_names[2:4]

elif code == 2:

    import Set_tf_1
    Is = [1, 2]
    DB_Ns = [i for i in range(len(db_info.DB_N_configurations))]
    model_names = model_DL_names[2:4]

for DB_N in DB_Ns:

    for I in Is:

        if I == 0:  # Hyper search

            interface_dict = {"hyper_model_search": True,
                              "train_final_model": False,
                              "perform_analysis": False,
                              "hyper_model_search_type": "genetic",
                              "metric_type": "KL"}

        elif I == 1:

            interface_dict = {"hyper_model_search": False,
                              "train_final_model": True,
                              "perform_analysis": False,
                              "hyper_model_search_type": "genetic",
                              "metric_type": "KL"}

        elif I == 2:

            interface_dict = {"hyper_model_search": False,
                              "train_final_model": False,
                              "perform_analysis": True,
                              "hyper_model_search_type": "genetic",
                              "metric_type": "KL"}

        else:

            interface_dict = {}

        for model_name in model_names:

            # Instantiate Manager class
            manager = Manager(DB_N=DB_N, model_name=model_name)
            manager.set_interface(interface_dict=interface_dict)
            manager.run()