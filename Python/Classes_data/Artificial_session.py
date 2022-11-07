from numpy import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import string
import copy

class Utilities_augmentation():

    @staticmethod
    def generate_simple_curve(size, loc=1, sigma=0.1, knot=1, beg_zero=True, end_zero=True):

        xx = (np.ones((1, 1)) * (np.arange(0, size, (size - 1) / (knot + 1)))).transpose()

        for i, x in enumerate(xx):
            xx[i] = xx[i] + np.random.uniform(low=x-size/10, high=x+size/10, size=1)
            if xx[i] < 0:
                xx[i] = 0

        yy = np.random.uniform(low=loc-sigma, high=loc+sigma, size=(knot + 2, 1))

        if beg_zero:
            yy[0] = 0
        if end_zero:
            yy[-1] = 0

        x_range = np.arange(size)

        tt = CubicSpline(xx[:, 0], yy, axis=0)(x_range)
        tt = tt.squeeze()

        return tt

    @staticmethod
    def linear_2D_distortion(XY_orig, ratio_X, ratio_Y):

        X = XY_orig[:, 0]
        Y = XY_orig[:, 1]

        min_X = np.min(X)
        max_X = np.max(X)
        mid_X = (max_X + min_X) / 2

        min_Y = np.min(Y)
        max_Y = np.max(Y)
        mid_Y = (min_Y + max_Y) / 2

        # Retarget curve
        xx_X = np.array([min_X, mid_X, max_X])
        xx_Y = np.array([min_Y, mid_Y, max_Y])

        yy_X = np.array([min_X, mid_X, max_X]) * ratio_X
        yy_Y = np.array([min_Y, mid_Y, max_Y]) * ratio_Y

        for i, x in enumerate(yy_X):
            yy_X[i] = yy_X[i] + np.random.uniform(low=x - 0.01, high=x + 0.01, size=1)

        for i, x in enumerate(yy_Y):
            yy_Y[i] = yy_Y[i] + np.random.uniform(low=x - 0.01, high=x + 0.01, size=1)

        X_new = CubicSpline(xx_X, yy_X, axis=0)(X)
        Y_new = CubicSpline(xx_Y, yy_Y, axis=0)(Y)

        XY_new = np.concatenate([X_new[:, np.newaxis], Y_new[:, np.newaxis]], axis=1)

        return XY_new

    @staticmethod
    def non_linear_2D_distortion(XY_orig, sigma, knot=1):

        X = XY_orig[:, 0]
        Y = XY_orig[:, 1]

        min_X = np.min(X)
        max_X = np.max(X)

        min_Y = np.min(Y)
        max_Y = np.max(Y)

        # Retarget curve
        xx_X = np.linspace(min_X, max_X, knot + 2)
        yy_X = np.linspace(min_X, max_X, knot + 2)

        xx_Y = np.linspace(min_Y, max_Y, knot + 2)
        yy_Y = np.linspace(min_Y, max_Y, knot + 2)

        amplitude_X = np.max(xx_X, axis=0) - np.min(xx_X, axis=0)
        amplitude_Y = np.max(xx_Y, axis=0) - np.min(xx_Y, axis=0)

        for i in range(len(yy_X)):
            yy_X[i] = yy_X[i] + np.random.uniform(low=-sigma * amplitude_X, high=sigma * amplitude_X, size=1)
            yy_Y[i] = yy_Y[i] + np.random.uniform(low=-sigma * amplitude_Y, high=sigma * amplitude_Y, size=1)

        X_new = CubicSpline(xx_X, yy_X, axis=0)(X)
        Y_new = CubicSpline(xx_Y, yy_Y, axis=0)(Y)

        XY_new = np.concatenate([X_new[:, np.newaxis], Y_new[:, np.newaxis]], axis=1)

        return XY_new

    @staticmethod
    def add_random_noise(XY_orig, sigma):

        height = XY_orig.shape[0]
        width = XY_orig.shape[1]

        amplitude_current_signals = np.max(XY_orig, axis=0) - np.min(XY_orig, axis=0)
        noise = np.random.normal(loc=0, scale=sigma, size=(height, width))

        # XY
        XY_aug = XY_orig + noise * amplitude_current_signals

        return XY_aug

    @staticmethod
    def rotation_2D(XY_orig, rotation):

        theta = np.radians(rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        XY_new = np.dot(R, XY_orig.T).T

        return XY_new

    @staticmethod
    def linear_change_factor(beg, end, n_session):

        a = (end - beg) / (n_session - 1)
        b = beg

        # Fit a linear function
        xx = [i for i in range(n_session)]
        yy = [a * i + b for i in range(n_session)]

        func = CubicSpline(xx, yy, axis=0)  # This is useless but can be a good habit to keep to learn to use fit function with scipy
        ratio_change = func(xx)

        return ratio_change

    @staticmethod
    def stairs_change_factor(beg, end, step, gamma, n_session):

        a = (end - beg) / (n_session - 1)
        b = beg

        # Fit a linear function
        xx = [i for i in range(n_session)]
        yy = [a * i + b for i in range(n_session)]

        func = CubicSpline(xx, yy, axis=0)

        ratio_change = []

        for i in range(n_session):
            if i > 0:
                if i % (step) == 0:

                    r = np.random.uniform(low=0, high=1)  # chance to improve

                    if r > gamma:  # Continue to improve

                        r = np.random.randint(low=1, high=2)
                        ratio_change.append(func(i + r))

                    else:  # no improvement (or even decrease)

                        r = np.random.randint(low=1, high=n_session // 2)

                        if (func(i - r)) < b:  # Never go below the b value
                            ratio_change.append(b)
                        else:
                            ratio_change.append(func(i - r))  # Roll back to previous values r step behind
                else:
                    ratio_change.append(ratio_change[-1])  # Take the last value if not change
            else:
                ratio_change.append(a * i + b)

        return np.array(ratio_change)

class Create_artificial_session():

    def __init__(self, XY, hyper_DA_global):

        """ Class to create artificial 2D hand movement from existing data.
        After instantiating the class use the class methods self.run()

        Parameters
        ----------
        XY: ndarray (n, 2)
            2D array representing X and Y coordinates over n samples
        hyper_DA_global: dict
            dictionary representing hyperparameters use to create new data

        Examples
        --------
        hyper_DA_global: {'n_session': 20,
                  'n_movement_per_session': 3,
                  'ch_type': {'distortion_type': 'linear',
                  'b_ratio_X': 1.0,
                  'b_ratio_Y': 0.5,
                  'rotation_type': 'linear',
                  'rotation': 45},
                  'session': {'sigma_rn': 0.01,
                  'sigma_aw': 0.01,
                  'knot_aw': 2,
                  'rot_range': [-10, 10],
                  'augmentation_types': ['rn', 'rot', 'aw']},
                  'name': 'Subject05'}
        """

        self.XY_orig = copy.deepcopy(XY)

        self.XY_aug = copy.deepcopy(self.XY_orig[np.newaxis])

        self.hyper_DA_global = hyper_DA_global

        self.n_session = self.hyper_DA_global['n_session']
        self.n_movement_per_session = self.hyper_DA_global['n_movement_per_session']

        self.augmentation_types = self.hyper_DA_global["session"]['augmentation_types']

        self.height = XY.shape[0]
        self.width = XY.shape[1]

        self.sessions_data_new_dict = {}

        for i in range(self.n_session):
            self.sessions_data_new_dict[i] = {}
            for j in range(self.n_movement_per_session):
                self.sessions_data_new_dict[i][j] = copy.deepcopy(self.XY_orig)

    def generate_change_factor(self):

        # Generate progression equation (Session sensitive)
        # Then for each movement in each session, we will apply other distortion (Random noise, warping)) later in the code

        # TODO: add random walk behavior
        # TODO: check here "end" variable in Utilities_augmentation function
        # TODO:

        if self.hyper_DA_global['ch_type']["distortion_type"] == "linear":

            ratio_X_change = Utilities_augmentation.linear_change_factor(beg=self.hyper_DA_global['ch_type']['b_ratio_X'], end=1., n_session=self.n_session)
            ratio_Y_change = Utilities_augmentation.linear_change_factor(beg=self.hyper_DA_global['ch_type']['b_ratio_Y'], end=1., n_session=self.n_session)

        elif self.hyper_DA_global['ch_type']["distortion_type"] == "linear_stair":

            ratio_X_change = Utilities_augmentation.stairs_change_factor(beg=self.hyper_DA_global['ch_type']['b_ratio_X'], end=1., n_session=self.n_session, step=2, gamma=0.5)
            ratio_Y_change = Utilities_augmentation.stairs_change_factor(beg=self.hyper_DA_global['ch_type']['b_ratio_Y'], end=1., n_session=self.n_session, step=2, gamma=0.5)

        else:

            ratio_X_change = np.ones(shape=self.n_session)
            ratio_Y_change = np.ones(shape=self.n_session)

        if self.hyper_DA_global['ch_type']["rotation_type"] == "linear":

            rotation_change = Utilities_augmentation.linear_change_factor(beg=self.hyper_DA_global['ch_type']['rotation'], end=0, n_session=self.n_session)

        elif self.hyper_DA_global['ch_type']["distortion_type"] == "linear_stair":

            rotation_change = Utilities_augmentation.stairs_change_factor(beg=self.hyper_DA_global['ch_type']['rotation'], end=0, n_session=self.n_session, step=4, gamma=0.5)

        else:

            rotation_change = np.zeros(shape=self.n_session)

        self.ratio_X_change = [ratio_X_change[j] for j in range(ratio_X_change.shape[0]) for i in range(self.n_movement_per_session)]
        self.ratio_Y_change = [ratio_Y_change[j] for j in range(ratio_Y_change.shape[0]) for i in range(self.n_movement_per_session)]
        self.rotation_change = [rotation_change[j]for j in range(rotation_change.shape[0]) for i in range(self.n_movement_per_session)]

    def run(self):

        """
        Return
        ----------
        sessions_data_new_dict: dict
            dictionary that contain n_session keys for each session created.
            Moreover, each key contain a dictionary with n_movement_per_session keys with (n, 2) 2D array each.
            This one only consider distortion and rotation
        sessions_data_new_aug_dict: dict
            dictionary that contain n_session keys for each session created.
            Moreover, each key contain a dictionary with n_movement_per_session keys with (n, 2) 2D array each.
            This one consider distortion, rotation, random noise and amplitude warping
        """

        self.X_aug = None
        self.Y_aug = None

        # Change on X, Y axis (linear or staircase)
        for i in range(self.n_session):
            for j in range(self.n_movement_per_session):
                XY_new = Utilities_augmentation.linear_2D_distortion(XY_orig=self.sessions_data_new_dict[i][j],
                                                                     ratio_X=self.ratio_X_change[i * self.n_movement_per_session + j],
                                                                     ratio_Y=self.ratio_Y_change[i * self.n_movement_per_session + j])
                self.sessions_data_new_dict[i][j] = XY_new

        # Rotation
        for i in range(self.n_session):
            for j in range(self.n_movement_per_session):
                XY_new = Utilities_augmentation.rotation_2D(XY_orig=self.sessions_data_new_dict[i][j],
                                                            rotation=self.rotation_change[i * self.n_movement_per_session + j])
                self.sessions_data_new_dict[i][j] = XY_new

        self.sessions_data_new_aug_dict = copy.deepcopy(self.sessions_data_new_dict)

        for augmentation_type in self.augmentation_types:

            if augmentation_type == "rn":
                for i in range(self.n_session):
                    for j in range(self.n_movement_per_session):
                        XY_new = Utilities_augmentation.add_random_noise(XY_orig=self.sessions_data_new_aug_dict[i][j], sigma=self.hyper_DA_global["session"]['sigma_rn'])
                        self.sessions_data_new_aug_dict[i][j] = XY_new

            if augmentation_type == "aw":
                for i in range(self.n_session):
                    for j in range(self.n_movement_per_session):
                        XY_new = Utilities_augmentation.non_linear_2D_distortion(XY_orig=self.sessions_data_new_aug_dict[i][j],
                                                                                 sigma=self.hyper_DA_global["session"]['sigma_aw'],
                                                                                 knot=self.hyper_DA_global["session"]['knot_aw'])
                        self.sessions_data_new_aug_dict[i][j] = XY_new

            if augmentation_type == "rot":
                for i in range(self.n_session):
                    for j in range(self.n_movement_per_session):
                        r = np.random.uniform(low=self.hyper_DA_global["session"]['rot_range'][0], high=self.hyper_DA_global["session"]['rot_range'][1])
                        XY_new = Utilities_augmentation.rotation_2D(XY_orig=self.sessions_data_new_aug_dict[i][j],
                                                                    rotation=r)
                        self.sessions_data_new_aug_dict[i][j] = XY_new

        return self.sessions_data_new_dict, self.sessions_data_new_aug_dict
