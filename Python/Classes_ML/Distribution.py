import numpy as np

class IDistribution():

    n_dim: int

    def __init__(self, mean: float=None, variance: float=None, mean_vector=None, cov_matrix=None):

        self.mean = mean
        self.variance = variance
        self.mean_vector = mean_vector
        self.cov_matrix = cov_matrix

        self.init_n()

    def init_n(self):

        pass

class Normal_distribution(IDistribution):

    def init_n(self):

        self.n_dim = 1

        if self.mean is None:
            self.mean = 0.0

        if self.variance is None:
            self.variance = 1.0

class Normal_distribution_2D(IDistribution):

    def init_n(self):

        self.n_dim = 2

        if self.mean_vector is None:
            self.mean_vector = np.array([[0.0],
                                         [0.0]])

        if self.cov_matrix is None:
            self.cov_matrix = np.array([[1.0, 0.0],
                                        [0.0, 1.0]])

        if self.mean_vector.shape != (2, 1):
            raise Exception("Shape of mean vector must be (2, 1).")

        if self.cov_matrix.shape != (2, 2):
            raise Exception("Shape of covariance matrix must be (2, 2).")
