from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import List

class IBehavior(ABC):  # Declare Behavior Interface

    mu_iterations: List
    std_iterations: List
    var_iterations: List
    tau_iterations: List

    parameters: dict
    n_iteration: int
    name: str

    @abstractmethod
    def set_parameters(self, *args, **kwargs):
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

class Solver(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def pull(self, k: int):
        pass

    @abstractmethod
    def update_Q(self, r: float, a: int):
        pass

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def set_hyperparameters(self, hyperparameters: dict):
        pass

class IPlotBehavior(ABC):

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass