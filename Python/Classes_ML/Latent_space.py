import math
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from numpy.polynomial.polynomial import polyfit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import shapiro, ttest_ind
from statsmodels.formula.api import ols
from pingouin import pairwise_ttests
import statsmodels.api as sm
from tabulate import tabulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm
import imageio
import matplotlib.patches as pat
import itertools
import copy
from collections import OrderedDict
from Classes_ML.Factory import Dictionary_factory
from Classes_ML.Interfaces import Latent
from Classes_ML.Distribution import Normal_distribution, Normal_distribution_2D, IDistribution
from Classes_ML.Regression_analysis import Regression_analysis
from Classes_ML.Plot import Plot_ML
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class LatentSpaceUtilities:

    @staticmethod
    def evaluate_accuracy(Z, y: [int], n_neighbors: int):

        """
        Evaluate accuracy

        Input
        -------
        Z : ndarray(float)
            2D array of data that is dimensionally reduced dim(num_train_data, 2)
        Y : list(str)
            list of train labels (Must be list of int)

        Returns
        -------
        metrics: float
            accuracy
        """

        knn_model = KNeighborsClassifier(n_neighbors)
        knn_model.fit(X=Z, y=y)
        metrics = knn_model.score(X=Z, y=y)
        print("Metrics = " + str(metrics))

        return metrics

    @staticmethod
    def behavior_regression(behavior_change_dict, d_standardized_valid_dict, deg: int, figure_suffix: str, model_name: str, plot_analysis=False, save_figure=False, path_figure=""):

        """ Compute metrics to evaluate the quality of the latent space

        Parameters
        ----------
        deg: int
            This value is used to define the order of the polyfit function. Value can be 1, 2 or 3.
            1: Linear regression
            2: second order polyfit
            3: third order polyfit

        Returns
        -------
        metrics: float
            R2 value
        """
        metrics = []

        # Plot
        for i, k in enumerate(behavior_change_dict.keys()):

            x = behavior_change_dict[k]
            y = d_standardized_valid_dict[k]

            if deg == 1:  # Linear regression

                regression_analysis = Regression_analysis(x, y, deg=deg, model_name=model_name)
                analysis = regression_analysis.analysis()

            elif deg == 2:  # Second order polyfit

                regression_analysis = Regression_analysis(x, y, deg=deg, model_name=model_name)
                analysis = regression_analysis.analysis()

            elif deg == 3:  # Third order polyfit

                regression_analysis = Regression_analysis(x, y, deg=deg, model_name=model_name)
                analysis = regression_analysis.analysis()

            else:  # Linear regression

                regression_analysis = Regression_analysis(x, y, deg=deg, model_name=model_name)
                analysis = regression_analysis.analysis()

            metrics.append(analysis["r_squared"])

            if plot_analysis:
                fig, axes = plt.subplots(1, 3, figsize=(18, 10))
                fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.15, hspace=0.15)

                axes[0].plot(x)
                axes[0].set_title(model_name + "_" + figure_suffix + "_" + k)
                axes[0].set_xlabel("Sessions")
                axes[0].set_ylabel("Behaviors")

                regression_analysis.plot_analysis(fig=fig, axes=axes[1:3],
                                                  figure_suffix=figure_suffix + "_" + k,
                                                  path_figure=path_figure,
                                                  save_figure=save_figure)

        return metrics

    @staticmethod
    def compute_behavior_metric(d_standardized_dict, gradient_dict, standardized_std):

        r2_normal_dict = {}

        for unique in d_standardized_dict.keys():  # keys: labels

            min_gradient = np.min(gradient_dict[unique])
            max_gradient = np.max(gradient_dict[unique])
            gradient_range = max_gradient - min_gradient
            true_gradient = [(g - min_gradient) / gradient_range for g in gradient_dict[unique]]

            d = d_standardized_dict[unique]

            true_cdf = []
            pred_cdf = []

            for i, d_session in enumerate(d):
                true_cdf.append(true_gradient[i])
                pred_cdf.append((0.5 - norm.sf(x=d_session, loc=0.0, scale=standardized_std)) * 2.0)

            r2_normal_dict[unique] = r2_score(y_true=true_cdf, y_pred=pred_cdf)

        return r2_normal_dict

    @staticmethod
    def compute_KL_divergence(P_distribution: IDistribution, Q_distribution: IDistribution):

        KL = -1.0

        if (type(P_distribution) is Normal_distribution) and (type(Q_distribution) is Normal_distribution):

            mu_p = P_distribution.mean
            mu_q = Q_distribution.mean
            sigma_p = P_distribution.variance
            sigma_q = Q_distribution.variance

            KL = np.log(sigma_q / sigma_p) + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2) - 0.5

        elif (type(P_distribution) is Normal_distribution_2D) and (type(Q_distribution) is Normal_distribution_2D):

            mu_p = P_distribution.mean_vector
            mu_q = Q_distribution.mean_vector
            sigma_p = P_distribution.cov_matrix
            sigma_q = Q_distribution.cov_matrix

            KL = 0.5 * (np.log(np.linalg.det(sigma_q) / np.linalg.det(sigma_p)) - 2.0
                        + np.trace(np.dot(np.linalg.inv(sigma_q), sigma_p))
                        + ((mu_q - mu_p).T @ np.linalg.inv(sigma_q) @ (mu_q - mu_p)))

        return KL

class Latent_space(Latent):

    def __init__(self):

        self.is_fit = False

    def fit(self, Z, Y, index_centroids=None):

        """ Class use to compute various metrics and perform transformation on the latent space.
        Fit the model with train/valid data

        Parameters
        ----------
        Z : ndarray(float)
            2D array of train data that is dimensionally reduced dim(num_train_data, 2)
        Y : list(str)
            list of train labels
        index_centroids : list(int)
            list of index of latent space use to compute the centroids. If None, th centroids will be computed with all fit data
        """

        if not index_centroids:
            index_centroids = [i for i in range(Z.shape[0])]

        self.Z_train = Z
        self.Y_train = Y
        self.unique_class = np.unique(self.Y_train)
        self.c_train = self.compute_centroids(z=self.Z_train[index_centroids, :], y=self.Y_train[index_centroids])
        self.d_train = self.compute_distance_with_centroids(z=self.Z_train, c=self.c_train, y=self.Y_train)

        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(np.unique(self.Y_train))))
        self.colors = dict(zip(np.unique(self.Y_train), colors))
        colors_centroid_thesis = Plot_ML.make_colors(name_list=['tab:blue', 'tab:red', 'tab:pink', 'tab:purple', 'tab:olive', 'tab:green'])
        self.colors = dict(zip(np.unique(self.Y_train), colors_centroid_thesis))
        # self.colors = dict(zip(np.unique(self.Y_train), colors))

        # Compute confidence ellipse
        self.ellipse_dict, self.ellipse_vertices_dict = Latent_space.compute_ellipse(z=self.Z_train, y=self.Y_train,
                                                                                     nstd=1.0, colors=self.colors)

        self.is_fit = True

    def transform(self, Z, Y):

        """ Class use to compute various metrics and perform transformation on the latent space
        Transform the model based ont he model created with train/valid data

        Parameters
        ----------
        Z : ndarray(float)
            2D array of train data that is dimensionally reduced dim(num_train_data, 2)
        Y : list(str)
            list of train labels
        """

        if self.is_fit:

            self.Z_test = Z
            self.Y_test = Y

            self.c_test = self.compute_centroids(z=self.Z_test, y=self.Y_test)
            self.d_test = self.compute_distance_with_centroids(z=self.Z_test,
                                                               c=self.c_train,
                                                               y=self.Y_test).squeeze()
            self.z_test_classes, self.num_test_data_classes = Dictionary_factory.split_classes(z=self.Z_test, y=self.Y_test,
                                                                                 unique_class=self.unique_class)

            self.d_test_classes, self.num_test_data_classes = Dictionary_factory.split_classes(z=self.d_test, y=self.Y_test,
                                                                                 unique_class=self.unique_class)
        else:

            print("Model was not fit")

    @staticmethod
    def compute_centroids(z, y):

        """Compute the centroids of each cluster in the latent space based on their label

        Parameters
        ----------
        z : ndarray(float)
            2D array of latent space data dim(num_test_data, 2)
        y : list(str)
            list of labels

        Returns
        -------
        c : dict
            dictionary of centroids. Each keys contain an ndarray representing the 2D centroids point of each unique label in y.
        """

        c = {}
        unique_class = np.unique(y)

        for i, unique in enumerate(unique_class):

            indices = [i for i, x in enumerate(y) if x == unique]

            if not indices:
                pass
            else:

                c[unique] = np.mean(z[indices, :], axis=0)

        return c

    @staticmethod
    def compute_distance_with_centroids(z, c, y):

        """Compute the distance between the points (z) with their respective centroids

        Parameters
        ----------
        z : ndarray(float)
            2D array of latent space data dim(num_test_data, 2)
        c : dict
            dictionary of centroids. Each keys contain an ndarray representing the 2D centroids point of each unique label in y.
        y : list(str)
            list of labels

        Returns
        -------
        d : ndarray(float)
            ndarray representing the distance of each point from z with their respective centroids contain in c.
        """
        d = []

        for i, label in enumerate(y):
            d.append(np.linalg.norm(c[label] - z[i], axis=-1))

        d = np.array(d)[np.newaxis, :]

        return d

    @staticmethod
    def compute_ellipse(z, y, nstd=1.0, colors=None):

        z = np.nan_to_num(z) # Sometimes when ML fail, sklearn return Nan

        unique_class = np.unique(y)
        ellipse_dict = {}
        ellipse_vertices_dict = {}

        for i, unique in enumerate(unique_class):

            ellipse_dict[unique] = {}

            indices = [i for i, x in enumerate(y) if x == unique]

            # Compute ellipse for plot
            Z_mean = np.nanmean(z[indices], axis=0)

            # Centered the data around each centroid
            Z_c = z[indices] - Z_mean

            # SVD on covariance
            cov = np.cov(Z_c.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
            theta = np.arctan2(vy, vx)

            width, height = 2 * nstd * np.sqrt(np.abs(eigvals))

            ellipse = Ellipse(xy=Z_mean, width=width, height=height, angle=math.degrees(theta),
                              fc=colors[unique], label=Plot_ML.remove_underbar(str(unique)+"_confidence_ellipse"), ec=(0, 0, 0, 0.5))
            # ellipse = Ellipse(xy=Z_mean, width=width, height=height, angle=math.degrees(theta),
            #                   fc=colors[unique], label=unique, ec=(0, 0, 0, 0.5))
            ellipse.set_alpha(alpha=0.5)
            ellipse_dict[unique] = ellipse

            # Get ellipse coordinates
            path = ellipse.get_path()
            vertices = path.vertices.copy()
            ellipse_vertices_dict[unique] = ellipse.get_patch_transform().transform(vertices)

            '''
            cov = np.cov(Z_c)
            U, S, Vt = np.linalg.svd(cov)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

            width, height = 2 * np.sqrt(S)
            '''

        return ellipse_dict, ellipse_vertices_dict

    @staticmethod
    def compute_2D_Gaussian(z, y):

        z = np.nan_to_num(z)  # Sometimes when ML fail, sklearn return Nan

        unique_class = np.unique(y)
        Gaussian_dict = {}

        for i, unique in enumerate(unique_class):

            Gaussian_dict[unique] = {}
            indices = [i for i, x in enumerate(y) if x == unique]

            # Center the data
            Z_mean = np.nanmean(z[indices], axis=0)
            Z_c = z[indices] - Z_mean

            cov = np.cov(Z_c.T)

            # Decorrelation of the data.
                # Eigen decomposition of the covariance matrix
                # SVD of the data matrix (n_dimension * n_vector)
                # SVD of the data covariance matrix

            # U, s, V = np.linalg.svd(Z_c.T)
            # Z = np.dot(U, Z_c.T).T
            #
            # cov = np.cov(Z.T)

            Gaussian_dict[unique]["mean"] = Z_mean.reshape([2, 1])
            Gaussian_dict[unique]["cov"] = cov

        return Gaussian_dict

class Standardize_latent_space(Latent):

    def __init__(self, CI=95):

        """ Class use to transform data points to create standardized latent space

        Parameters
        ----------
        CI : int
            confidence interval
        """

        self.CI = CI
        self.is_fit = False

    def fit(self, Z, Y, index_centroids=None):

        """ Class use to compute various metrics and perform transformation on the latent space.
        Fit the model with train/valid data

        Parameters
        ----------
        Z : ndarray(float)
            2D array of train data that is dimensionally reduced dim(num_train_data, 2)
        Y : list(str)
            list of train labels
        index_centroids : list(int)
            list of index of latent space use to compute the centroids. If None, th centroids will be computed with all fit data
        """

        if not index_centroids:
            index_centroids = [i for i in range(Z.shape[0])]

        self.Z_train = Z
        self.Y_train = Y
        self.unique_class = np.unique(self.Y_train)

        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(np.unique(self.Y_train))))
        self.colors = dict(zip(np.unique(self.Y_train), colors))
        colors_centroid_thesis = Plot_ML.make_colors(name_list=['tab:blue', 'tab:red', 'tab:pink', 'tab:purple', 'tab:olive', 'tab:green'])
        self.colors = dict(zip(np.unique(self.Y_train), colors_centroid_thesis))
        # self.colors = dict(zip(np.unique(self.Y_train), colors))

        self.c = np.sqrt(chi2.ppf(self.CI * 0.01, 2))

        # Compute centroids
        self.c_train = Latent_space.compute_centroids(z=self.Z_train[index_centroids, :], y=self.Y_train[index_centroids])

        self.z_train_classes, self.num_train_data_classes = Dictionary_factory.split_classes(z=self.Z_train, y=self.Y_train, unique_class=self.unique_class)
        self.__fit(z=self.Z_train, y=self.Y_train)
        self.Z_train = self.__transform(z=self.Z_train, y=self.Y_train)

        # Compute centroids
        self.c_train = Latent_space.compute_centroids(z=self.Z_train[index_centroids, :], y=self.Y_train[index_centroids])

        # Compute distance with centroids
        self.d_train = Latent_space.compute_distance_with_centroids(z=self.Z_train, c=self.c_train, y=self.Y_train)

        # Compute confidence ellipse
        self.ellipse_dict, self.ellipse_vertices_dict = Latent_space.compute_ellipse(z=self.Z_train, y=self.Y_train,
                                                                                     nstd=1.0, colors=self.colors)

        self.is_fit = True

    def transform(self, Z, Y):

        """ Class use to compute various metrics and perform transformation on the latent space
        Transform the model based ont he model created with train/valid data

        Parameters
        ----------
        Z : ndarray(float)
            2D array of train data that is dimensionally reduced dim(num_train_data, 2)
        Y : list(str)
            list of train labels
        """

        if self.is_fit:

            self.Z_test = Z
            self.Y_test = Y

            self.c_test = Latent_space.compute_centroids(z=self.Z_test, y=self.Y_test)

            self.z_test_classes, self.num_test_data_classes = Dictionary_factory.split_classes(z=self.Z_test, y=self.Y_test,
                                                                                 unique_class=self.unique_class)
            self.Z_test = self.__transform(z=self.Z_test, y=self.Y_test)
            self.c_test = Latent_space.compute_centroids(z=self.Z_test, y=self.Y_test)

            self.d_test = Latent_space.compute_distance_with_centroids(z=self.Z_test, c=self.c_train, y=self.Y_test).squeeze()
            self.d_test_classes, self.num_test_data_classes = Dictionary_factory.split_classes(z=self.d_test, y=self.Y_test,
                                                                                 unique_class=self.unique_class)
        else:

            print("Model was not fit")

    def __fit(self, z, y):

        """Compute Z_test data into classes
           Please execute before transform_train() and transform_test() function
        """

        z_classes, num_data_classes = Dictionary_factory.split_classes(z, y, unique_class=self.unique_class)

        self.eigenvalues_z_classes = {}
        self.eigenvectors_z_classes = {}
        self.theta_classes = {}
        self.a_classes = {}
        self.b_classes = {}
        self.R_classes = {}
        self.T_classes = {}
        self.standardization_matrix_classes = {}

        for i, unique in enumerate(self.unique_class):

            # Test part
            X = z_classes[unique]
            X = X - np.mean(X, axis=0)
            cov = np.cov(X.T)

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.abs(eigvals)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
            theta = np.arctan2(vy, vx)

            self.eigenvalues_z_classes[unique] = eigvals
            self.eigenvectors_z_classes[unique] = eigvecs

            self.theta_classes[unique] = theta

            nstd = 3  # square root of percent point of chi2

            self.a_classes[unique] = 2 * nstd * np.sqrt(eigvals[0])
            self.b_classes[unique] = 2 * nstd * np.sqrt(eigvals[1])

            self.R_classes[unique] = np.array([[np.cos(self.theta_classes[unique]), -np.sin(self.theta_classes[unique])],
                                               [np.sin(self.theta_classes[unique]), np.cos(self.theta_classes[unique])]])

            self.T_classes[unique] = np.vstack([np.concatenate([self.R_classes[unique], self.c_train[unique].reshape(2, 1)], axis=1),
                                                np.array([0, 0, 1])])

            # print("Shape of covariance matrix = ", cov.shape)
            # print("Shape of eigen vectors of " + str(unique) + " = ", self.eigenvectors_z_classes[unique].shape)
            # print("Updated shape of eigen vectors of " + str(unique) + " = ", self.eigenvectors_z_classes[unique].shape)
            # print("Eigen values of " + str(unique) + " = ", self.eigenvalues_z_classes[unique])

        # self.r = self.c / np.sqrt(len(self.unique_class)) * np.sqrt(sum([np.sqrt(self.eigenvalues_z_classes[y].prod()) for y in self.unique_class]))
        self.r = 1.0  # radius

        self.standardization_matrix_classes = {y: np.array([[(self.r*2) / self.a_classes[y], 0],
                                                            [0, (self.r*2) / self.b_classes[y]]]) for y in self.unique_class}

        self.standardized_std = np.sqrt(self.r/nstd)  # standard deviation of normal distribution of standardized circle
        #self.chi2_cdf = chi2.cdf(x=nstd**2, df=2)  # cumulative distribution function of chi2 (= one of normal distribution)
        #self.norm_ppf = norm.ppf(a=self.chi2_cdf, loc=0, scale=self.standardized_std)  # percent point function of normal distribution
        #self.normal_scale = self.r / self.norm_ppf

    def __transform(self, z, y):

        """Transform latent space into standardized latent space. Please use fit first

        Returns
        -------
        z_standardized : ndarray
            2D array of standardized Z_train data dim(num_train_data, 2)
        """

        unique_class = np.unique(y)
        z_standardized = z.copy()

        for i, unique in enumerate(unique_class):

            indices = [i for i, x in enumerate(y) if x == unique]

            z_1_class = np.vstack([z[indices].T, np.ones(z[indices].shape[0])])  # shape:(3, n)
            z_ellipse_1_class = np.dot(np.linalg.inv(self.T_classes[unique]), z_1_class)  # shape:(3, n)
            z_ellipse_class = z_ellipse_1_class[:2, :].T  # shape:(n, 2)
            z_circle_class = np.dot(self.standardization_matrix_classes[unique], z_ellipse_class.T).T  # shape:(n, 2)
            z_circle_1_class = np.vstack([z_circle_class.T, np.ones(z[indices].shape[0])])  # shape:(3, n)
            z_standardized_1_class = z_circle_1_class[:2, :].T  # shape:(n, 2)

            z_standardized[indices] = z_standardized_1_class

        # z_classes, num_data_classes = Dictionary_factory.split_classes(z, y, unique_class=self.unique_class)
        #
        # z_circle_classes = {}
        # z_standardized_classes = {}
        #
        # for i, unique in enumerate(self.unique_class):
        #
        #     z_1_class = np.vstack([z_classes[unique].T, np.ones(num_data_classes[unique])])  # can't do with np.concatenate because dimension of inputs is different, shape:(3, n)
        #
        #     z_ellipse_1_class = np.dot(np.linalg.inv(self.T_classes[unique]), z_1_class)  # shape:(3, n)
        #     z_ellipse_class = z_ellipse_1_class[:2, :].T  # shape:(n, 2)
        #
        #     # z_circle_class = np.dot(self.standardization_matrix_classes[y], z_ellipse_class.T).T  # shape:(n, 2)
        #     z_circle_classes[unique] = np.dot(self.standardization_matrix_classes[unique], z_ellipse_class.T).T  # shape:(n, 2)
        #     # z_circle_1_class = np.vstack([z_circle_class.T, np.ones(self.num_train_data_classes[y])])  # shape:(3, n)
        #     z_circle_1_class = np.vstack([z_circle_classes[unique].T, np.ones(num_data_classes[unique])])  # shape:(3, n)
        #
        #     z_standardized_1_class = np.dot(self.T_classes[unique], z_circle_1_class)  # shape:(3, n)
        #     z_standardized_classes[unique] = z_standardized_1_class[:2, :].T  # shape:(n, 2)
        #
        # # self.z_train_standardized = np.concatenate([z_standardized_classes[y] for y in self.unique_class], axis=0)
        # z_standardized = np.concatenate([z_circle_classes[y] for y in self.unique_class], axis=0)

        return z_standardized

class Statistics_distance():

    def __init__(self, d_test, Y_test):

        self.d_test = d_test
        self.Y_test = Y_test

        self.mean, self.std = self.compute_mean_std(d=self.d_test, Y_class=self.Y_test)

    @staticmethod
    def compute_mean_std(d, Y_class):

        mean = {}
        std = {}

        unique_class = np.unique(Y_class)

        for i, unique in enumerate(unique_class):

            indices = [i for i, x in enumerate(Y_class) if x == unique]

            if not indices:
                pass
            else:

                mean[unique] = np.nanmean(d[indices], axis=0)
                std[unique] = np.nanstd(d[indices], axis=0)

        return mean, std

    @staticmethod
    def normality_test(d, Y_class, display_results=False):

        alpha = 0.05

        normality_check = {}

        unique_class = np.unique(Y_class)

        for i, unique in enumerate(unique_class):

            indices = [i for i, x in enumerate(Y_class) if x == unique]

            if not indices:
                pass
            else:

                stat, p = shapiro(d[indices])
                if display_results:
                    print('Statistics=%.3f, p=%.3f' % (stat, p))
                if p > alpha:
                    normality_check[unique] = True
                    if display_results:
                        print('Sample looks Gaussian (fail to reject H0) for ' + unique)
                else:
                    normality_check[unique] = False
                    if display_results:
                        print('Sample does not look Gaussian (reject H0) for ' + unique)

        return normality_check

    @staticmethod
    def combine_array_2_string(mean, std):

        array_str = np.empty(shape=mean.shape, dtype='object')

        line = 0
        for a, b in zip(mean, std):
            row = 0
            for a2, b2 in zip(a, b):
                array_str[line, row] = str(a2) + " (" + str(b2) + ")"
                row += 1
            line += 1

        return array_str

    @staticmethod
    def get_ANOVA_table(df_test, subject, display_results=False, plot_box_plot=False):

        print(tabulate(df_test, headers='keys', tablefmt='psql'))

        # ANOVA test
        sessions = np.unique(df_test["Session_name"].values)

        unique_class = np.unique(df_test["True"].values)
        index_dictionary = dict(enumerate(sessions))
        index_dictionary_invert = {v: k for k, v in index_dictionary.items()}

        results = {}

        for i, unique in enumerate(unique_class):

            results[unique] = {}

            df_temp = df_test[(df_test['Subject_name'] == subject)
                              & (df_test['True'] == unique)]

            df_anova = pd.DataFrame(df_temp, columns=['d', 'Session_name'])
            df_anova = df_anova.rename(columns={'Session_name': 'group'})
            df_anova = df_anova.rename(columns={'d': 'metrics'})

            model = ols('metrics ~ C(group)', data=df_anova).fit()
            anova_result = sm.stats.anova_lm(model, typ=2)

            p_value = anova_result['PR(>F)']['C(group)']

            if plot_box_plot:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig = df_anova.boxplot('metrics', 'group', ax=ax, grid=False)
                fig.set_title(unique)

            if p_value < 0.05:

                m_comp = pairwise_ttests(data=df_anova, dv='metrics', between='group', padjust='bonf')
                p_uncorr = m_comp['p-unc']

                A = m_comp['A']
                B = m_comp['B']
                df_cond = pd.concat([A, B, p_uncorr], axis=1)
                df_cond = df_cond.rename(columns={'p-unc': 'p'})

                if "p-corr" in m_comp.columns:
                    p_corr = m_comp['p-corr']
                    A = m_comp['A']
                    B = m_comp['B']
                    df_cond1 = pd.concat([A, B, p_corr], axis=1)
                    df_cond2 = pd.concat([B, A, p_corr], axis=1)
                    df_cond2 = df_cond2.rename(columns={'A': 'B', 'B': 'A'})

                    df_cond1 = df_cond1.rename(columns={'p-corr': 'p'})
                    df_cond2 = df_cond2.rename(columns={'p-corr': 'p'})

                    df_cond = pd.concat([df_cond1, df_cond2], axis=0)

                for session in sessions:
                    results[unique][session] = df_cond['B'][(df_cond['A'] == session)].values.tolist()

                if display_results:
                    print(" -----------------------------" + unique + " (significant ANOVA) ------------------------------------------------ ")
                    print(model.summary())
                    print(tabulate(anova_result, headers='keys', tablefmt='psql'))

                    print(tabulate(m_comp, headers='keys', tablefmt='psql'))
                    print()
                    print(" --- Uncorrected ---")

                if display_results:
                    for p, p_val in enumerate(p_uncorr):
                        if p_val < 0.05:
                            print("     - Significant different - " + str(unique) + " between session " + str(A[p]) + " and session " + str(B[p]) + " --------------")
                    print(" ----------------------------------------------------------------------------- ")
                    print()

                if display_results:
                    if "p-corr" in m_comp.columns:
                        print(" --- Bonferoni ---")
                        for p, p_val in enumerate(p_corr):
                            if p_val < 0.05:
                                print("     - Significant different - " + str(unique) + " between session " + str(A[p]) + " and session " + str(B[p]) + " --------------")
                        print(" ----------------------------------------------------------------------------- ")
                        print()
                    else:
                        print(" --- Bonferoni not possible given the number of condition ---")
                        print(" ----------------------------------------------------------------------------- ")

            else:

                results[unique] = None
                print(" -----------------------------" + unique + " (non-significant ANOVA) ------------------------------------------------ ")
                print(model.summary())

        return results