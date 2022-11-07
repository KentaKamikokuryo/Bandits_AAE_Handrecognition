import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from abc import ABC, ABCMeta, abstractmethod, abstractproperty

from Classes_ML.Interfaces import IModelML, IModel
from Classes_ML.Interfaces import Factory

class PCA(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.kernel = self.hyper["kernel"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("PCA")
        print("model_name: " + str(self.model_name))
        print("kernel: " + str(self.kernel))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = decomposition.KernelPCA(n_components=self.n_components, kernel=self.kernel)

        return model

class T_SNE(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.perplexity = self.hyper["perplexity"]
        self.learning_rate = self.hyper["learning_rate"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("TSNE")
        print("model_name: " + str(self.model_name))
        print("perplexity: " + str(self.perplexity))
        print("learning_rate: " + str(self.learning_rate))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = TSNE(n_components=self.n_components, perplexity=self.perplexity,
                     learning_rate=self.learning_rate, random_state=0, n_iter=300, verbose=1)

        return model

class ICA(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.algorithm = self.hyper["algorithm"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("ICA")
        print("model_name: " + str(self.model_name))
        print("algorithm: " + str(self.algorithm))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = FastICA(n_components=self.n_components, algorithm=self.algorithm, random_state=0)

        return model

class MuldiDimSscaling(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.metric = self.hyper["metric"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("MDS")
        print("metric: " + str(self.metric))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = MDS(n_components=self.n_components, metric=self.metric)

        return model

class ISO(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.n_neighbors = self.hyper["n_neighbors"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("ISO")
        print("n_neighbors: " + str(self.n_neighbors))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = Isomap(n_components=self.n_components, n_neighbors=self.n_neighbors)

        return model

class LLE(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.n_neighbors = self.hyper["n_neighbors"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("LLE")
        print("n_neighbors: " + str(self.n_neighbors))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components,
                                       method='modified', eigen_solver='dense')

        return model

class Laplacian(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.affinity = self.hyper["affinity"]
        self.n_neighbors = self.hyper["n_neighbors"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("Laplacian")
        print("affinity: " + str(self.affinity))
        print("n_neighbors: " + str(self.n_neighbors))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = SpectralEmbedding(affinity=self.affinity, n_neighbors=self.n_neighbors,
                                  n_components=self.n_components)

        return model

class LDA(IModelML):

    def __init__(self, hyper):

        self.hyper = hyper
        self.solver = self.hyper["solver"]
        self.n_components = self.hyper["n_components"]
        self.n_neighbors_latent_space = self.hyper["n_neighbors_latent_space"]
        self.model_name = self.hyper["model_name"]

        print("LDA")
        print("solver: " + str(self.solver))
        print("n_components: " + str(self.n_components))
        print("n_neighbors_latent_space: " + str(self.n_neighbors_latent_space))

    def create(self):

        model = LinearDiscriminantAnalysis(solver=self.solver, n_components=self.n_components)

        return model

class ModelFactoryML(Factory):

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model

    def create(self):

        if self.hyper_model['model_name'] == "PCA":
            model_obj = PCA(self.hyper_model)
        elif self.hyper_model['model_name'] == "T_SNE":
            model_obj = T_SNE(self.hyper_model)
        elif self.hyper_model['model_name'] == "ICA":
            model_obj = ICA(self.hyper_model)
        elif self.hyper_model['model_name'] == "MDS":
            model_obj = MuldiDimSscaling(self.hyper_model)
        elif self.hyper_model['model_name'] == "ISO":
            model_obj = ISO(self.hyper_model)
        elif self.hyper_model['model_name'] == "LLE":
            model_obj = LLE(self.hyper_model)
        elif self.hyper_model['model_name'] == "Laplacian":
            model_obj = Laplacian(self.hyper_model)
        elif self.hyper_model['model_name'] == "LDA":
            model_obj = LDA(self.hyper_model)
        else:
            model_obj = None

        model = model_obj.create()

        return model

class Model_ML(IModel):

    def __init__(self, hyper_model: dict):

        """
        Create Machine Learning model corresponding to the input dictionary hyper_model hyperparameters

        Input
        -------
        hyper_model : dict
            2D array of data that is dimensionally reduced dim(num_train_data, 2)

        Returns
        -------
        None
        """

        self.hyper_model = hyper_model
        self.model_name = self.hyper_model['model_name']

        self.create_model()

    def create_model(self):

        self.modelFactory = ModelFactoryML(hyper_model=self.hyper_model)
        self.model = self.modelFactory.create()

    def fit(self, X_fit, y_fit, X_transform, y_transform):

        self.model.fit(X_fit, y_fit)
        z_train = self.model.transform(X_fit)

        return z_train

    def transform(self, X_transform, y_transform):

        z_test = self.model.transform(X_transform)

        return z_test

    def fit_transform(self, X_fit, y_fit, X_transform, y_transform):

        self.create_model()

        cut_index = X_fit.shape[0]

        X = np.concatenate([X_fit, X_transform], axis=0)

        z = self.model.fit_transform(X)

        z_train = z[:cut_index]
        z_test = z[cut_index:]

        return z_train, z_test, z

    def transform_models_dict(self, X_transform, y_transform, K):

        if self.models_dict[K]:
            z_test = self.models_dict[K].transform(X_transform)
        else:
            z_test = None

        return z_test

    def fit_transform_models_dict(self, X_fit, y_fit, X_transform, y_transform, K):

        if self.models_dict[K]:

            cut_index = X_fit.shape[0]

            X = np.concatenate([X_fit, X_transform], axis=0)

            z = self.models_dict[K].fit_transform(X)

            z_train = z[:cut_index]
            z_test = z[cut_index:]

        else:
            z_test = None
            z_train = None
            z = None

        return z_train, z_test, z

    def save_model(self, K, path_model=""):

        name = str(self.model_name) + "_K_" + str(K) + ".pkl"
        joblib.dump(self.model, path_model + name)
        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + name)

    def load_model(self, n_K_fold, path_model=""):

        self.models_dict = {}

        for K in range(0, n_K_fold):

            name = str(self.model_name) + "_K_" + str(K) + ".pkl"
            self.models_dict[K] = joblib.load(path_model + name)

            print('Model ' + self.hyper_model['model_name'] + " load from " + path_model + name)

    def save_best_nn(self, best_nn, K, path_model=""):

        name = str(self.model_name) + "_K_" + str(K) + "_best_nn.npy"
        np.save(path_model + name, best_nn)

    def load_best_nn(self, n_K_fold, path_model=""):

        self.best_nn_dict = {}

        for K in range(0, n_K_fold):

            # Load best nn for unsupervised
            name = str(self.model_name) + "_K_" + str(K) + "_best_nn.npy"
            self.best_nn_dict[K] = np.load(path_model + name)

    def show_config(self):

        df = pd.DataFrame.from_dict([self.hyper_model])
        print(tabulate(df, headers='keys', tablefmt='psql'))
