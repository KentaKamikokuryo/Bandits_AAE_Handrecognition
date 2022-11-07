from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import List
import numpy as np
from tensorflow.keras.models import Model

class Factory(ABC):  # Declare Factory Interface

    @abstractmethod
    def create(self):
        pass

class Latent(ABC):

    Z_train: np.ndarray
    Y_train: np.ndarray
    Z_test: np.ndarray
    Y_test: np.ndarray
    c_train: dict
    ellipse_dict: dict

    @abstractmethod
    def fit(self, Z, Y):
        pass

    @abstractmethod
    def transform(self, Z, Y):
        pass

class IModelML(ABC):

    @abstractmethod
    def create(self):
        pass

class IModelDL(ABC):  # Declare Behavior Interface

    encoder_model: Model
    decoder_model: Model
    autoencoder_model: Model

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, validation_data=None, shuffle=True):
        pass

    @abstractmethod
    def save_model(self, K: int, path_model: str):

        pass

    @abstractmethod
    def load_model(self, n_K_fold: int, path_model: str):

        pass

class IModel(ABC):

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def fit(self, X_fit: np.ndarray, y_fit: np.ndarray, X_transform: np.ndarray, y_transform: np.ndarray):
        pass

    @abstractmethod
    def transform(self, X_transform: np.ndarray, y_transform: np.ndarray):
        pass

    @abstractmethod
    def fit_transform(self, X_fit: np.ndarray, y_fit: np.ndarray, X_transform: np.ndarray, y_transform: np.ndarray):
        pass

    @abstractmethod
    def transform_models_dict(self, X_transform: np.ndarray, y_transform: np.ndarray, K: int):
        pass

    @abstractmethod
    def fit_transform_models_dict(self, X_fit: np.ndarray, y_fit: np.ndarray, X_transform: np.ndarray, y_transform: np.ndarray, K: int):
        pass

    @abstractmethod
    def save_model(self, K: int, path_model: str):
        pass

    @abstractmethod
    def load_model(self, n_K_fold: int, path_model: str):
        pass

    @abstractmethod
    def save_best_nn(self, best_nn: int, K: int, path_model: str):
        pass

    @abstractmethod
    def load_best_nn(self, n_K_fold: int, path_model: str):
        pass

    @abstractmethod
    def show_config(self):
        pass


class IScope(ABC):

    loss: object
    metrics_list: List[object]
    optimizer: object

    @abstractmethod
    def create(self):
        pass
