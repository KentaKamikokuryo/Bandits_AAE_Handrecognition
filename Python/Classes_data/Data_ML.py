import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import norm

from Classes_data.DB import DB
from Classes_ML.Utilities_ML import Utilities_ML
from Classes_ML.Factory import Dataframe_factory
from Classes_ML.Latent_space import Latent_space, Standardize_latent_space
from Classes_ML.Interfaces import Latent
from Classes_data.Info import DB_info
from Classes_ML.Plot import Plot_ML

class Data:

    def __init__(self, db_info: DB_info):

        self.db_info = db_info
        self.DB_N = self.db_info.DB_N

    def get_DB_N(self, is_DL: bool= False, reshape_to_1D: bool = False):

        self.tvt_info = self.db_info.tvt_info

        self.n_K_fold = self.tvt_info['n_K_fold']
        self.n_outer_loop = self.tvt_info['n_outer_loop']

        if is_DL:
            self.path_figure_DB_N = self.db_info.path_figure_DL_DB_N
            self.path_model_DB_N = self.db_info.path_model_DL_DB_N
            self.path_search_models_DB_N = self.db_info.path_search_models_DL_DB_N
        else:
            self.path_figure_DB_N = self.db_info.path_figure_ML_DB_N
            self.path_model_DB_N = self.db_info.path_model_ML_DB_N
            self.path_search_models_DB_N = self.db_info.path_search_models_ML_DB_N

        # Get database
        db = DB(DB_N=self.DB_N)
        self.XY, self.X_image = db.get_DB_data(reshape_to_1D=reshape_to_1D)

        self.subjects_train_target = list(self.db_info.DB_N_configuration["subjects_train_target"])
        self.subjects_train_target = [s + "_T" for s in self.subjects_train_target]
        self.subjects_train_valid = list(self.db_info.DB_N_configuration["subjects_train_valid"])
        self.subjects_test = list(self.db_info.DB_N_configuration["subjects_test"])

        # Todo: this may lead to problem in the future
        self.behaviors_name = list(self.XY.keys())[5:]

        self.n_subjects_train_valid = len(self.subjects_train_valid)
        self.n_subjects_test = len(self.subjects_test)
        self.n_subjects = self.n_subjects_train_valid + self.n_subjects_test

        # generate_K_fold_index here
        self.samples_train_K, self.samples_valid_K = Utilities_ML.generate_K_fold_index(samples_train_valid=list(self.subjects_train_valid), tvt_info=self.tvt_info)
        self.samples_test_K = self.subjects_test

        # Add target
        for K in self.samples_train_K.keys():
            self.samples_train_K[K].extend(self.subjects_train_target)

        # Get labels
        labels = []
        for subject in self.XY["Y"].keys():
            labels.append([x for x in self.XY["Y"][subject].keys()])

        labels = np.array(labels).squeeze()
        self.labels = np.unique(labels)
        self.n_label = len(self.labels)

    def get_fit_transform_data(self, K: int, is_test: bool):

        if is_test:

            X_fit = np.concatenate([self.XY["X"][k][k2] for k in self.samples_train_K[K] + self.samples_valid_K[K] for k2 in self.XY["X"][k].keys()], axis=0)
            X_transform = np.concatenate([self.XY["X"][k][k2] for k in self.samples_test_K for k2 in self.XY["X"][k].keys()], axis=0)

            y_fit = np.concatenate([self.XY["Y"][k][k2] for k in self.samples_train_K[K] + self.samples_valid_K[K] for k2 in self.XY["Y"][k].keys()], axis=0)
            y_transform = np.concatenate([self.XY["Y"][k][k2] for k in self.samples_test_K for k2 in self.XY["Y"][k].keys()], axis=0)

            tuple_fit = [(k, k2) for k in self.samples_train_K[K] + self.samples_valid_K[K] for k2 in self.XY["X"][k].keys() for i in range(self.XY["X"][k][k2].shape[0])]
            tuple_transform = [(k, k2) for k in self.samples_test_K for k2 in self.XY["X"][k].keys() for i in range(self.XY["X"][k][k2].shape[0])]

        else:

            # Reorganize data for training
            X_fit = np.concatenate([self.XY["X"][k][k2] for k in self.samples_train_K[K] for k2 in self.XY["X"][k].keys()], axis=0)
            X_transform = np.concatenate([self.XY["X"][k][k2] for k in self.samples_valid_K[K] for k2 in self.XY["X"][k].keys()], axis=0)

            y_fit = np.concatenate([self.XY["Y"][k][k2] for k in self.samples_train_K[K] for k2 in self.XY["Y"][k].keys()], axis=0)
            y_transform = np.concatenate([self.XY["Y"][k][k2] for k in self.samples_valid_K[K] for k2 in self.XY["Y"][k].keys()], axis=0)

            tuple_fit = [(k, k2) for k in self.samples_train_K[K] for k2 in self.XY["X"][k].keys() for i in range(self.XY["X"][k][k2].shape[0])]
            tuple_transform = [(k, k2) for k in self.samples_valid_K[K] for k2 in self.XY["X"][k].keys() for i in range(self.XY["X"][k][k2].shape[0])]

        le = preprocessing.LabelEncoder()
        le.fit(y_fit)

        y_fit_encoded = le.transform(y_fit)
        y_transform_encoded = le.transform(y_transform)

        return X_fit, X_transform, y_fit, y_transform, y_fit_encoded, y_transform_encoded, tuple_fit, tuple_transform

    def get_behavior_data(self, behavior_name: str, K: int, is_test: bool):

        if is_test:

            behavior_fit = np.concatenate([self.XY[behavior_name][k][k2] for k in self.samples_train_K[K] + self.samples_valid_K[K] for k2 in self.XY[behavior_name][k].keys()], axis=0)
            behavior_transform = np.concatenate([self.XY[behavior_name][k][k2] for k in self.samples_test_K for k2 in self.XY[behavior_name][k].keys()], axis=0)

        else:

            behavior_fit = np.concatenate([self.XY[behavior_name][k][k2] for k in self.samples_train_K[K] for k2 in self.XY[behavior_name][k].keys()], axis=0)
            behavior_transform = np.concatenate([self.XY[behavior_name][k][k2] for k in self.samples_valid_K[K] for k2 in self.XY[behavior_name][k].keys()], axis=0)

        return behavior_fit, behavior_transform

    def get_centroids_index(self, K: int, is_test: bool):

        # Detect target subject to compute centroids
        index_centroids = []
        start = 0
        end = 0

        if is_test:

            for k in self.samples_train_K[K]:
                index = list(self.XY["X"][k].keys())
                for k2 in self.XY["X"][k].keys():
                    end += self.XY["X"][k][k2].shape[0]
                    if ("_T" in k):
                        index_centroids.extend([i for i in range(start, end)])
                    start = end
        else:
            for k in self.samples_test_K:
                index = list(self.XY["X"][k].keys())
                for k2 in self.XY["X"][k].keys():
                    end += self.XY["X"][k][k2].shape[0]
                    if ("_T" in k):
                        index_centroids.extend([i for i in range(start, end)])
                    start = end

        return index_centroids

class Data_Z:

    def __init__(self, data: Data, model_name: str):

        self.data = data
        self.model_name = model_name

        self.db_info = data.db_info

    def evaluate_latent_space(self, Z_fit_K, Z_transform_K,  metric_type: str, is_test: bool):

        self.metrics_transform_K = {}

        for K in Z_fit_K.keys():

            _, _, y_fit, y_transform, _, _, _, _ = self.data.get_fit_transform_data(K=K, is_test=is_test)
            index_centroids = self.data.get_centroids_index(K=K, is_test=is_test)

            # Fit latent space
            latent_space = Latent_space()
            latent_space.fit(Z=Z_fit_K[K], Y=y_fit, index_centroids=index_centroids)
            latent_space.transform(Z=Z_transform_K[K], Y=y_transform)

            # Fit standardized latent space
            standardize_latent_space = Standardize_latent_space()
            standardize_latent_space.fit(Z=Z_fit_K[K], Y=y_fit, index_centroids=index_centroids)
            standardize_latent_space.transform(Z=Z_transform_K[K], Y=y_transform)

            n_neighbors = 10

            # Evaluate the model
            if metric_type == "accuracy":
                # metrics = LatentSpaceUtilities.evaluate_accuracy(Z=Z_transform_K[K], y=y_transform, n_neighbors=n_neighbors)
                pass

            elif metric_type == "regressor_knn":

                metrics = []

                for behavior_name in self.data.behaviors_name:

                    behavior_fit, behavior_transform = self.data.get_behavior_data(behavior_name=behavior_name, K=K, is_test=is_test)

                    # Test NN regressor
                    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
                    knn.fit(Z_fit_K[K], behavior_fit)

                    behavior_transform_ = knn.predict(X=Z_transform_K[K])
                    metric = np.mean(behavior_transform - behavior_transform_)

                    metrics.append(np.abs(metric))

                metrics = np.mean(metrics)

            # behavior_name = "ratio_X_change"
            #
            # if is_test:
            #     subjects = self.samples_test_K
            #     plot_analysis = True
            #     save_figure = True
            #     behavior_transform = np.concatenate([self.XY[behavior_name][k][k2] for k in subjects for k2 in self.XY[behavior_name][k].keys()], axis=0)
            # else:
            #     subjects = self.samples_valid_K[K]
            #     plot_analysis = False
            #     save_figure = False
            #     behavior_transform = np.concatenate([self.XY[behavior_name][k][k2] for k in subjects for k2 in self.XY[behavior_name][k].keys()], axis=0)
            #
            # behavior_change_dict, _ = Dictionary_factory.split_classes(z=np.array(behavior_transform), y=y_transform, unique_class=np.unique(y_fit))
            # d_standardized_valid_dict = standardize_latent_space.d_test_classes
            #
            # figure_suffix = "K_" + str(K) + "_" + behavior_name
            # metrics = LatentSpaceUtilities.behavior_regression(behavior_change_dict=behavior_change_dict,
            #                                                    d_standardized_valid_dict=d_standardized_valid_dict,
            #                                                    figure_suffix=figure_suffix,
            #                                                    deg=2,
            #                                                    model_name=self.model_name,
            #                                                    plot_analysis=plot_analysis,
            #                                                    save_figure=save_figure,
            #                                                    path_figure=self.path_figure)

            else:
                metrics = np.array([0])

            self.metrics_transform_K[K] = np.mean(metrics)

        return self.metrics_transform_K

    def latent_analysis(self, Z_fit_K, Z_transform_K, do_standardized: bool,  is_test: bool):

        Z_fit_trajectories_K = {}
        Z_transform_trajectories_K = {}
        latent_space_K = {}

        for K in Z_fit_K.keys():

            X_fit, X_transform, y_fit, y_transform, y_fit_encoded, y_transform_encoded, tuple_fit, tuple_transform = self.data.get_fit_transform_data(K=K, is_test=is_test)

            index_centroids = self.data.get_centroids_index(K=K, is_test=is_test)

            if do_standardized:
                latent_space = Standardize_latent_space()
            else:
                latent_space = Latent_space()

            latent_space.fit(Z=Z_fit_K[K], Y=y_fit, index_centroids=index_centroids)
            latent_space.transform(Z=Z_transform_K[K], Y=y_transform)

            Z_fit_trajectories_K[K] = self.to_subjects_trajectories(latent_space=latent_space,
                                                                    Z_transform=Z_fit_K[K], y_transform=y_fit,
                                                                    tuple_subject_session=tuple_fit, is_standardized=do_standardized)

            Z_transform_trajectories_K[K] = self.to_subjects_trajectories(latent_space=latent_space,
                                                                          Z_transform=Z_transform_K[K], y_transform=y_transform,
                                                                          tuple_subject_session=tuple_transform, is_standardized=do_standardized)

            latent_space_K[K] = latent_space

        return latent_space_K, Z_fit_trajectories_K, Z_transform_trajectories_K

    def plot_latent_space(self, latent_space: Latent, path_figure: str = None, figure_name: str = "", figure_title: str = ""):

        if latent_space.__class__.__name__ == 'Latent_space':

            display = False

            if display:
                plt.ion()
            else:
                plt.ioff()

            z_train = latent_space.Z_train
            y_train = latent_space.Y_train
            z_test = latent_space.Z_test
            y_test = latent_space.Y_test

            figure_title_spec = figure_title
            figure_name_spec = figure_name

            # Plot latent space
            fig, ax = Plot_ML.plot_latent_space(z_train=z_train, y_train=y_train, z_test=z_test, y_test=y_test,
                                                gradient_train=None, gradient_test=None,
                                                figure_title=figure_title_spec)
            Plot_ML.plot_confidence_ellipse(ellipse_dict=latent_space.ellipse_dict, fig=fig, ax=ax)
            Plot_ML.plot_centroids(c=latent_space.c_train, fig=fig, ax=ax)

            Plot_ML.savefig(path_figure=path_figure, figure_name=figure_name_spec, fig=fig)

        else:

            print("latent_space should be of type Latent_space")
            return None

    def plot_latent_gradient(self, latent_space: Latent, K_plot:int=0, path_figure: str = None, figure_name: str = "", figure_title: str = ""):

        if latent_space.__class__.__name__ == 'Latent_space':
            pass
        else:
            print("latent_space should be of type Latent_space")
            return None

        display = False

        if display:
            plt.ion()
        else:
            plt.ioff()

        for behavior_name in self.data.behaviors_name:

            behavior_fit, behavior_transform = self.data.get_behavior_data(behavior_name=behavior_name, K=K_plot, is_test=True)

            z_train = latent_space.Z_train
            y_train = latent_space.Y_train
            z_test = latent_space.Z_test
            y_test = latent_space.Y_test
            c = latent_space.c_train
            gradient_train = behavior_fit
            gradient_test = behavior_transform

            figure_title_spec = figure_title + "_" + behavior_name + "_" + "grad"
            figure_name_spec = figure_name + "_" + behavior_name + "_" + "grad"

            # Plot latent space with gradient of color based on artificial behavior information
            fig, ax = Plot_ML.plot_latent_space(z_train=z_train, y_train=y_train, z_test=z_test, y_test=y_test,
                                                gradient_train=gradient_train, gradient_test=gradient_test,
                                                figure_title=figure_title_spec)

            Plot_ML.plot_confidence_ellipse(ellipse_dict=latent_space.ellipse_dict, fig=fig, ax=ax)
            Plot_ML.plot_centroids(c=c, fig=fig, ax=ax)


            Plot_ML.savefig(path_figure=path_figure, figure_name=figure_name_spec, fig=fig)

    def plot_latent_trajectories(self, latent_space: Latent, Z_transform_trajectories: dict, K_plot:int =0, path_figure: str = None, figure_name: str = "", figure_title: str = ""):

        if latent_space.__class__.__name__ == 'Latent_space':
            pass
        else:
            print("latent_space should be of type Latent_space")
            return None

        display = False

        if display:
            plt.ion()
        else:
            plt.ioff()

        for behavior_name in self.data.behaviors_name:

            behavior_fit, behavior_transform = self.data.get_behavior_data(behavior_name=behavior_name, K=K_plot, is_test=True)

            z_train = latent_space.Z_train
            y_train = latent_space.Y_train
            z_test = latent_space.Z_test
            y_test = latent_space.Y_test
            c = latent_space.c_train
            gradient_train = behavior_fit
            gradient_test = behavior_transform

            for subject in Z_transform_trajectories["Z"].keys():

                figure_title_spec = figure_title + "_" + behavior_name + "_" + "grad_traj_" + str(subject)
                figure_name_spec = figure_name + "_" + behavior_name + "_" + "grad_traj_" + str(subject)

                z_test_dict = {k: Z_transform_trajectories["Z"][subject][k] for k in Z_transform_trajectories["Z"][subject].keys()}

                fig, ax = Plot_ML.plot_latent_space(z_train=z_train, y_train=y_train, z_test=z_test, y_test=y_test,
                                                    gradient_train=gradient_train, gradient_test=gradient_test,
                                                    figure_title=figure_title_spec)

                Plot_ML.plot_trajectories(z_test_dict=z_test_dict, fig=fig, ax=ax)

                Plot_ML.savefig(path_figure=path_figure, figure_name=figure_name_spec, fig=fig)

    def plot_standardized_latent_gradient(self, latent_space: Latent, K_plot:int=0, path_figure: str = None, figure_name: str = "", figure_title: str = ""):

        if latent_space.__class__.__name__ == 'Standardize_latent_space':
            pass
        else:
            print("latent_space should be of type Standardize_latent_space")
            return None

        for behavior_name in self.data.behaviors_name:

            behavior_fit, behavior_transform = self.data.get_behavior_data(behavior_name=behavior_name, K=K_plot, is_test=True)

            display = False

            if display:
                plt.ion()
            else:
                plt.ioff()

            z_train = latent_space.Z_train
            y_train = latent_space.Y_train
            z_test = latent_space.Z_test
            y_test = latent_space.Y_test
            c = latent_space.c_train
            gradient_train = behavior_fit
            gradient_test = behavior_transform

            figure_title_spec = figure_title + "_" + behavior_name + "_" + "grad_std"
            figure_name_spec = figure_name + "_" + behavior_name + "_" + "grad_std"

            # Plot individual space with gradient of color based on artificial behavior information
            fig, axes = Plot_ML.plot_latent_space_subplot(z_train=z_train, y_train=y_train, z_test=z_test, y_test=y_test,
                                                        gradient_train=gradient_train, gradient_test=gradient_test,
                                                        figure_title=figure_title_spec)

            Plot_ML.plot_confidence_ellipse_subplot(ellipse_dict=latent_space.ellipse_dict, fig=fig, axes=axes)
            Plot_ML.savefig(path_figure=path_figure, figure_name=figure_name_spec, fig=fig)

    def plot_standardized_latent_trajectories(self, latent_space: Latent, Z_transform_trajectories: dict, K_plot:int =0, path_figure: str = None, figure_name: str = "", figure_title: str = ""):

        if latent_space.__class__.__name__ == 'Standardize_latent_space':
            pass
        else:
            print("latent_space should be of type Standardize_latent_space")
            return None

        display = False

        if display:
            plt.ion()
        else:
            plt.ioff()

        for behavior_name in self.data.behaviors_name:

            behavior_fit, behavior_transform = self.data.get_behavior_data(behavior_name=behavior_name, K=K_plot, is_test=True)

            z_train = latent_space.Z_train
            y_train = latent_space.Y_train
            z_test = latent_space.Z_test
            y_test = latent_space.Y_test

            gradient_train = behavior_fit
            gradient_test = behavior_transform

            for subject in Z_transform_trajectories["Z_std"].keys():

                figure_title_spec = figure_title + "_" + behavior_name + "_" + "grad_traj_" + str(subject) + "_std"
                figure_name_spec = figure_name + "_" + behavior_name + "_" + "grad_traj_" + str(subject) + "_std"

                z_test_dict = {k: Z_transform_trajectories["Z_std"][subject][k] for k in Z_transform_trajectories["Z_std"][subject].keys()}

                # Plot individual space with gradient of color based on artificial behavior information
                fig, axes = Plot_ML.plot_latent_space_subplot(z_train=z_train, y_train=y_train, z_test=z_test, y_test=y_test,
                                                              gradient_train=gradient_train, gradient_test=gradient_test,
                                                              figure_title=figure_title_spec)
                Plot_ML.plot_trajectories_subplot(z_test_dict=z_test_dict, fig=fig, axes=axes)
                Plot_ML.savefig(path_figure=path_figure, figure_name=figure_name_spec, fig=fig)

    def save_Z(self, Z_fit, Z_transform, metric_mean, metric_sd, losses: dict, path: str):

        # Save results
        name = self.model_name + "_Z_fit.npy"
        np.save(path + name, Z_fit, allow_pickle=True)

        name = self.model_name + "_Z_transform.npy"
        np.save(path + name, Z_transform, allow_pickle=True)

        name = self.model_name + "_metric_mean.npy"
        np.save(path + name, metric_mean, allow_pickle=True)

        name = self.model_name + "_metric_sd.npy"
        np.save(path + name, metric_sd, allow_pickle=True)

        name = self.model_name + "_losses.npy"
        np.save(path + name, losses, allow_pickle=True)

        print("Z_fit, Z_transform, metric_mean, metric_sd, losses saved at: " + path)

    def load_Z(self, path):

        name = self.model_name + "_Z_fit.npy"
        Z_fit = np.load(path + name, allow_pickle=True).item()

        name = self.model_name + "_Z_transform.npy"
        Z_transform = np.load(path + name, allow_pickle=True).item()

        name = self.model_name + "_metric_mean.npy"
        metric_valid_mean = np.load(path + name, allow_pickle=True).item()

        name = self.model_name + "_metric_sd.npy"
        metric_valid_sd = np.load(path + name, allow_pickle=True).item()

        name = self.model_name + "_losses.npy"
        loss_dict = np.load(path + name, allow_pickle=True).item()

        print("Z_fit, Z_transform, metric_mean, metric_sd, losses loaded from: " + path)

        return Z_fit, Z_transform, metric_valid_mean, metric_valid_sd, loss_dict

    def save_latent_analysis(self, latent_space: Latent, Z_fit_trajectories: dict, Z_transform_trajectories: dict, path: str, is_standardized: bool):

        """ Save latent space information at the path of the DB_N

        Parameters
        ----------
        latent_space: Latent
            latent_space or Standardize_latent_space
        Z_fit_trajectories: dict
            Here is an example of how the dictionary is organized
            First:
                trajectories_dict.keys() will give:
                dict_keys(['Subject01', 'Subject02', 'Subject03', 'Subject04', 'Subject05', 'Subject06', 'Subject07'])
                It represents each subjects
            Then:
                trajectories_dict['Subject01'].keys() will give:
                dict_keys([0, 1, 2])
                It represents each label for the key subject:
            Then:
                trajectories_dict['Subject01'][0].keys() will give:
                dict_keys(['Z', 'Z_std', 'd', 'd_std'])
                d, d_std represent the distance in the latent space between the sample and the centroids of its corresponding cluster.
            Then:
                trajectories_dict['Subject01'][0]["d"].shape will give:
                (20)
                Meaning there is 20 sessions with each value for each session.
        Z_centroids_dict: dict
            the dictionary is organized as Label keys
        path: str
            path
        """

        if is_standardized:
            suffix = "_standardized"
        else:
            suffix = ""

        Z_centroids_dict = {}

        for K in latent_space.keys():
            Z_centroids_dict[K] = latent_space[K].c_train

        name = self.model_name + "_Z_fit_trajectories" + suffix + ".npy"
        np.save(path + name, Z_fit_trajectories, allow_pickle=True)

        name = self.model_name + "_Z_transform_trajectories" + suffix + ".npy"
        np.save(path + name, Z_transform_trajectories, allow_pickle=True)

        name = self.model_name + "_Z_centroids_dict" + suffix + ".npy"
        np.save(path + name, Z_centroids_dict, allow_pickle=True)

        print("Z_fit_trajectories, Z_transform_trajectories, Z_centroids_dict saved at: " + path)

    def load_latent_analysis(self, path: str, is_standardized: bool):

        """ load latent space information at the path of the DB_N

        Parameters
        ----------
        model_name: str
            None

        Returns
        -------
        Z_fit_dict: dict
            Here is an example of how the dictionary is organized
            First:
                Z_fit_dict.keys() will give:
                dict_keys(['Subject01', 'Subject02', 'Subject03', 'Subject04', 'Subject05', 'Subject06', 'Subject07'])
                It represents each subjects
            Then:
                Z_fit_dict['Subject01'].keys() will give:
                dict_keys(['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020'])
                It represents each session for the key subject:
            Then:
                Z_fit_dict['Subject01']['0001'].keys() will give:
                dict_keys(['Y', 'Z', 'Z_std', 'd', 'd_std', 'ratio_X_change'])
                d, d_mean, d_std represent the distance in the latent space between the sample and the centroids of its corresponding cluster.
            Then:
                Z_fit_dict['Subject01']['0001']["Y"].shape will be the same as the number of various movement done on one session

        Z_transform_dict: dict
            Same as Z_fit_dict
        Z_centroids_dict: dict
            Same as Z_fit_dict
        """

        if is_standardized:
            suffix = "_standardized"
        else:
            suffix = ""

        name = self.model_name + "_Z_fit_trajectories" + suffix + ".npy"
        Z_fit_trajectories = np.load(path + name, allow_pickle=True).item()

        name = self.model_name + "_Z_transform_trajectories" + suffix + ".npy"
        Z_transform_trajectories = np.load(path + name, allow_pickle=True).item()

        name = self.model_name + "_Z_centroids_dict" + suffix + ".npy"
        Z_centroids_dict = np.load(path + name, allow_pickle=True).item()

        print("Z_fit_trajectories, Z_transform_trajectories, Z_centroids_dict loaded from: " + path)

        return Z_fit_trajectories, Z_transform_trajectories, Z_centroids_dict

    def save_latent_df(self, Z_fit_trajectories, Z_transform_trajectories, Z_fit_trajectories_std, Z_transform_trajectories_std, K: int, path: str):

        """ Save latent space information as pandas dataframe (.csv file) at the path of the DB_N
        """

        Z_fit = {**Z_fit_trajectories[K], **Z_fit_trajectories_std[K]}
        Z_transform = {**Z_transform_trajectories[K], **Z_transform_trajectories_std[K]}

        # Save dataframe
        df_fit, image_fit = Data_Z.Z_as_dataframe(XY=self.data.XY, Z=Z_fit,
                                                  data_image=self.data.XY["X_img"],
                                                  subject_to_keep=self.data.subjects_train_valid)

        df_transform, image_transform = Data_Z.Z_as_dataframe(XY=self.data.XY, Z=Z_transform,
                                                              data_image=self.data.XY["X_img"],
                                                              subject_to_keep=self.data.subjects_test)

        name = self.model_name + "_df_fit_" + str(K) + ".csv"
        df_fit.to_csv(path + name, index=False)

        name = self.model_name + "_df_transform_" + str(K) + ".csv"
        df_fit.to_csv(path + name, index=False)

        name = self.model_name + "_image_fit_" + str(K) + ".npy"
        np.save(path + name, image_fit)

        name = self.model_name + "_image_transform_" + str(K) + ".npy"
        np.save(path + name, image_transform)

        print("df_fit, df_transform, image_fit, image_transform saved at: " + path)

    def load_latent_df(self, K: int, path: str):

        name = self.model_name + "_df_fit_" + str(K) + ".csv"
        df_fit = pd.read_csv(path + name)

        name = self.model_name + "_df_transform_" + str(K) + ".csv"
        df_transform = pd.read_csv(path + name)

        name = self.model_name + "_image_fit_" + str(K) + ".npy"
        image_fit = np.load(path + name)

        name = self.model_name + "_image_transform_" + str(K) + ".npy"
        image_transform = np.load(path + name)

        df_fit['Session_name'] = df_fit['Session_name'].astype(str).str.zfill(4)
        df_transform['Session_name'] = df_transform['Session_name'].astype(str).str.zfill(4)

        print("df_fit, df_transform, image_fit, image_transform loaded from: " + path)

        return df_fit, df_transform, image_fit, image_transform

    @staticmethod
    def to_subjects_trajectories(latent_space: Latent_space, Z_transform, y_transform, tuple_subject_session, is_standardized: bool):

        Z = {}
        Z["Z"] = {}
        Z["d"] = {}

        subjects = np.unique([t[0] for t in tuple_subject_session])
        labels = np.unique([t[1] for t in tuple_subject_session])

        import copy
        latent_space_temp = copy.deepcopy(latent_space)

        # Get latent space for each subject and session
        for k in subjects:

            Z["Z"][k] = {}
            Z["d"][k] = {}

            session_count = 0

            for k2 in labels:

                Z["Z"][k][k2] = {}
                Z["d"][k][k2] = {}

                indices = [i for i, x in enumerate(tuple_subject_session) if x == (k, k2)]

                latent_space_temp.transform(Z=Z_transform[indices], Y=y_transform[indices])

                Z["Z"][k][k2] = latent_space_temp.Z_test
                Z["d"][k][k2] = latent_space_temp.d_test

                session_count += 1

        if is_standardized:

            Z_std = {}
            Z_std["Z_std"] = Z["Z"]
            Z_std["d_std"] = Z["d"]

            return Z_std

        else:

            return Z

    @staticmethod
    def Z_as_dataframe(XY, Z, data_image, subject_to_keep: list):

        # Get train data/information
        df_temp = []
        image = []

        for k in subject_to_keep:
            if "_T" in k:  # Ignore target is they exist
                break

            for k2 in XY["df"][k].keys():  # Label

                df = XY["df"][k][k2]

                # Add data to dataframe
                df["Z_0"] = Z["Z"][k][k2][:, 0]
                df["Z_1"] = Z["Z"][k][k2][:, 1]

                df["Z_std_0"] = Z["Z_std"][k][k2][:, 0]
                df["Z_std_1"] = Z["Z_std"][k][k2][:, 1]

                df["d"] = Z["d"][k][k2]
                df["d_std"] = Z["d_std"][k][k2]

                df["ratio_X_change"] = XY["ratio_X_change"][k][k2]
                df["ratio_Y_change"] = XY["ratio_Y_change"][k][k2]
                df["rotation_change"] = XY["rotation_change"][k][k2]

                df_temp.append(df)

        #         image.extend([data_image[k][k2]])
        #
        # image = np.concatenate([img for img in image], axis=0)
        df = pd.concat(df_temp, axis=0)

        df = Dataframe_factory.add_index_columns(df, new_columns="Image_index")
        df = df.reset_index(drop=True)
        df = Dataframe_factory.add_subject_session_index_to_df(df=df, columns=["Session_name", "Subject_name"], new_columns=["Index_session", "Index_subject"])
        df["Index_label"] = df["index"]
        df = df.drop("index", 1)

        print(tabulate(df.head(n=50), headers='keys', tablefmt='psql'))

        return df, image