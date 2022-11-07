import numpy as np
import pandas as pd
import math, itertools, os
from tabulate import tabulate

np.set_printoptions(precision=4, suppress=True)


class Hyperparameters_ML():

    def __init__(self, model_name):

        self.model_name = model_name

    def generate_hyperparameters(self, display_info=False):

        if self.model_name == "PCA":

            hyperparameters_choices = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid', 'cosine'],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        elif self.model_name == "T_SNE":

            hyperparameters_choices = dict(perplexity=[5, 10, 20, 50, 100],
                                           learning_rate=[100],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        elif self.model_name == "ICA":

            hyperparameters_choices = dict(algorithm=['parallel', 'deflation'],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        elif self.model_name == "MDS":

            hyperparameters_choices = dict(metric=[True, False],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        elif self.model_name == "ISO":

            hyperparameters_choices = dict(n_neighbors=[5, 10, 20, 50, 100],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        elif self.model_name == "LLE":

            hyperparameters_choices = dict(n_neighbors=[5, 10, 20, 50, 100],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        elif self.model_name == "Laplacian":

            hyperparameters_choices = dict(affinity=['nearest_neighbors', 'rbf'],
                                           n_neighbors=[5, 10, 20, 50, 100],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])
        elif self.model_name == "LDA":

            hyperparameters_choices = dict(solver=['svd'],
                                           n_components=[2],
                                           n_neighbors_latent_space=[10],
                                           model_name=[self.model_name])

        else:

            hyperparameters_choices = dict()

        keys, values = zip(*hyperparameters_choices.items())
        hyperparameters_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(hyperparameters_all_combination)

        if display_info:
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return hyperparameters_all_combination, hyperparameters_choices


class Hyperparameters_DL():

    def __init__(self, model_name):

        self.model_name = model_name

    def generate_hyperparameters(self, n_label: int = 3, max_epoch: int = 10, display_info:bool = False):

        if self.model_name == "AE":

            hyperparameters_choices = dict(input_dim=[(64,)],
                                           latent_dim=[[2]],
                                           n_neighbors_latent_space=[10],

                                           encoder_units=[[32, 16, 8]],
                                           activation=['relu'],
                                           use_batch_normalization=[True],
                                           learning_rate_schedule=[dict(method='none', learning_rate_start=0.01),
                                                                   dict(method='none', learning_rate_start=0.05)],

                                           dropout_rate=[0.2],
                                           optimizer_name=['adam'],
                                           metric_function=[dict(name='rmse')],
                                           loss_function=[dict(name='rmse')],

                                           batch_size=[32],
                                           max_epoch=[max_epoch],
                                           model_name=[self.model_name])

        elif self.model_name == "VAE":

            hyperparameters_choices = dict(input_dim=[(64,)],
                                           latent_dim=[[2]],
                                           n_neighbors_latent_space=[10],

                                           encoder_units=[[32, 16, 8], [64, 32, 16]],
                                           activation=['relu', 'sigmoid', 'tanh'],

                                           use_batch_normalization=[True],

                                           learning_rate_schedule=[dict(method='none', learning_rate_start=0.01),
                                                                   dict(method='none', learning_rate_start=0.001)],

                                           dropout_rate=[0.2, 0.4],
                                           optimizer_name=['adam'],
                                           loss_function=[dict(name='VAE_loss', loss_lambda=0.01),
                                                          dict(name='VAE_loss', loss_lambda=100)],
                                           metric_function=[dict(name='rmse')],

                                           batch_size=[32],
                                           max_epoch=[max_epoch],
                                           model_name=[self.model_name])

        elif self.model_name == "UAAE" or self.model_name == "SSAAE":

            # TODO: add learning rate decay methods

            hyperparameters_choices = dict(input_dim=[(64,)],
                                           latent_dim=[[2]],
                                           n_neighbors_latent_space=[10],

                                           encoder_activation=['sigmoid', 'tanh', 'relu'],

                                           encoder_units=[[8, 8], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256],
                                                          [16, 8], [32, 16], [64, 32], [128, 64], [256, 128], [512, 256],
                                                          [8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256],
                                                          [16, 8, 4], [32, 16, 8], [64, 32, 16], [128, 64, 32], [256, 128, 64], [512, 256, 128],
                                                          [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512]],

                                           discriminator_units=[[512, 512]],
                                           discriminator_activation=['sigmoid',  'relu'],

                                           kernel_regularizer_name=['none'],
                                           kernel_initializer_info=[dict(name='glorot_uniform')],

                                           use_batch_normalization=[False],
                                           learning_rate_schedule=[dict(method='none', learning_rate_start=0.01),
                                                                   dict(method='none', learning_rate_start=0.005),
                                                                   dict(method='none', learning_rate_start=0.001),
                                                                   dict(method='none', learning_rate_start=0.0005),
                                                                   dict(method='none', learning_rate_start=0.0001)],

                                           prior=[dict(size=n_label, x_std=1.0, y_std=0.5, shift=4.0)],

                                           dropout_rate=[0.0, 0.1, 0.2, 0.3, 0.4],
                                           optimizer_name=['adam'],
                                           loss_function=[dict(name='AAE_loss')],
                                           metric_function=[dict(name='rmse')],

                                           batch_size=[64],
                                           max_epoch=[max_epoch],
                                           model_name=[self.model_name])

        else:

            hyperparameters_choices = dict()

        keys, values = zip(*hyperparameters_choices.items())
        hyperparameters_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(hyperparameters_all_combination)

        if display_info:
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return hyperparameters_all_combination, hyperparameters_choices
