import numpy as np
from scipy.stats import norm
import math


class PriorDistribution():

    @staticmethod
    def gaussian(batch_size, n_dim: int = 2, mean: float = 0.0, var: float = 1.0):

        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)

        return z

    @staticmethod
    def gaussian_N(batch_size, n_dim: int = 2, n_labels: int = 3, x_std: float = 0.5, y_std: float = 0.1, shift: float = 1.0, label_indices=None):

        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(x, y, shift, label, n_labels):

            r = 2.0 * np.pi / float(n_labels) * float(label)
            new_x = x * math.cos(r) - y * math.sin(r)
            new_y = x * math.sin(r) + y * math.cos(r)
            new_x += shift * math.cos(r)
            new_y += shift * math.sin(r)

            return np.array([new_x, new_y]).reshape((2,))

        x = np.random.normal(0, x_std, (batch_size, (int)(n_dim / 2)))
        y = np.random.normal(0, y_std, (batch_size, (int)(n_dim / 2)))
        z = np.empty((batch_size, n_dim), dtype=np.float32)

        for batch in range(batch_size):

            for zi in range((int)(n_dim / 2)):

                if label_indices is not None:
                    z[batch, zi * 2:zi * 2 + 2] = sample(x=x[batch, zi], y=y[batch, zi], shift=shift, label=label_indices[batch], n_labels=n_labels)
                else:
                    z[batch, zi * 2:zi * 2 + 2] = sample(x=x[batch, zi], y=y[batch, zi], shift=shift, label=np.random.randint(0, n_labels), n_labels=n_labels)

        return z

    @staticmethod
    def get_centroid(n_labels: int = 3, label: int = 0, shift: float = 1.0):

        r = 2.0 * np.pi / float(n_labels) * float(label)

        centroid_x = shift * math.cos(r)
        centroid_y = shift * math.sin(r)

        centroid = np.array([[centroid_x],
                             [centroid_y]])

        return centroid

    @staticmethod
    def compute_cov_from_eigen_theta(eigen_values, theta):

        n_dim = 2

        if len(eigen_values) != 2:
            raise Exception("The numbers of eigen values must be 2.")

        D = np.zeros(shape=(n_dim, n_dim))

        for i, value in enumerate(eigen_values):
            D[i, i] = value

        eigen_vector1 = np.array([[np.cos(theta)],
                                  [np.sin(theta)]])
        eigen_vector2 = np.array([[-np.sin(theta)],
                                  [np.cos(theta)]])

        P = np.concatenate([eigen_vector1, eigen_vector2], axis=1)

        cov = P @ D @ P.T  # In fact, P @ D @ np.linalg.inv(P)

        return cov

    @staticmethod
    def generate_z_un(batch_size, n_dim: int = 2, mean: float = 0, var: float = 1):

        z_real_dist = PriorDistribution.gaussian(batch_size, n_dim=n_dim, mean=mean, var=var)

        return z_real_dist

    @staticmethod
    def generate_z_un_N(batch_size, n_dim: int = 2, n_labels: int = 3, x_std: float = 0.5, y_std: float = 0.1, shift: float = 1.0):

        z_real_dist = PriorDistribution.gaussian_N(batch_size, n_dim=n_dim, n_labels=n_labels,
                                                   x_std=x_std, y_std=y_std, shift=shift,
                                                   label_indices=None)

        return z_real_dist

    @staticmethod
    def generate_z_ss_N(batch_size, y, n_dim: int = 2, n_labels: int = 3, x_std: float = 0.5, y_std: float = 0.1, shift: float = 1.0):

        z_real_dist = PriorDistribution.gaussian_N(batch_size, n_dim=n_dim, n_labels=n_labels,
                                                   x_std=x_std, y_std=y_std, shift=shift,
                                                   label_indices=y)

        z_id = np.zeros((batch_size, n_labels))
        z_id[np.arange(batch_size), y] = 1

        return z_real_dist, z_id

# z_real_dist = PriorDistribution.generate_z_un(batch_size=1000, n_dim=2, mean=0, var=1)
# plt.figure()
# plt.scatter(z_real_dist[:, 0], z_real_dist[:, 1])
#
# z_real_dist = PriorDistribution.generate_z_un_N(batch_size=1000, n_labels=3)
# plt.figure()
# plt.scatter(z_real_dist[:, 0], z_real_dist[:, 1])
#
# z_real_dist, _ = PriorDistribution.generate_z_ss(batch_size=1000, n_labels=3)
# plt.figure()
# plt.scatter(z_real_dist[:, 0], z_real_dist[:, 1])

# class PriorDistribution():
#
#     @staticmethod
#     def Gaussian_2D(batch_size, mean=None, cov=None):
#
#         if mean is None:
#             mean = np.array([0.0, 0.0])
#
#         if cov is None:
#             cov = np.array([[1.0, 0.0],
#                             [0.0, 1.0]])
#
#         prior_distribution = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)
#
#         return prior_distribution
#
#     @staticmethod
#     def N_Gaussian_2D(batch_size, N=1, mean=None, cov=None, shift=None, interval=0.95, y_train=None):
#
#         if mean is None:
#             mean = np.array([0.0, 0.0])
#
#         if cov is None:
#             cov = np.array([[1.0, 0.0],
#                             [0.0, 0.1]])
#
#         if shift is None:
#             _, shift = norm.interval(alpha=interval, loc=mean[0], scale=cov[0, 0])
#
#         if y_train is None:
#             y_train = np.random.randint(low=0, high=N, size=batch_size)
#
#         theta_n = 2 * np.pi / N * y_train
#
#         prior_distribution = PriorDistribution.Gaussian_2D(batch_size=batch_size, mean=mean, cov=cov)
#         x = prior_distribution[:, 0]
#         y = prior_distribution[:, 1]
#
#         new_x = x * np.cos(theta_n) - y * np.sin(theta_n)
#         new_y = x * np.sin(theta_n) + y * np.cos(theta_n)
#         new_x += shift * np.cos(theta_n)
#         new_y += shift * np.sin(theta_n)
#
#         prior_distribution[:, 0] = new_x
#         prior_distribution[:, 1] = new_y
#
#         labels = np.zeros((len(y_train), N))
#         labels[range(len(y_train)), np.array(y_train).astype(int)] = 1
#
#         return prior_distribution, labels
