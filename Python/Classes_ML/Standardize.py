import copy
import numpy as np

class Standardize():

    @staticmethod
    def generate_standardized_index(data_shape, standardization_type="all"):

        if standardization_type == 'all':
            inds = [[i for i in range(data_shape)]]
        elif standardization_type == 'individual':
            inds = [[i] for i in range(data_shape)]
        else:
            inds = [[i for i in range(data_shape)]]

        return inds

    @staticmethod
    def standardize_3D(data, inds):

        data_out = copy.deepcopy(data)

        std = np.zeros(shape=(data.shape[2]))
        mean = np.zeros(shape=(data.shape[2]))

        for ind in inds:

            mean_temp = np.nanmean(data[:, :, ind])
            std_temp = np.nanstd(data[:, :, ind], ddof=1)

            std[ind] = std_temp
            mean[ind] = mean_temp

        std = std.tolist()
        mean = mean.tolist()

        for i in range(data_out.shape[2]):
            data_out[:, :, i] = (data[:, :, i] - mean[i]) / std[i]

        data_out[np.isnan(data_out)] = 0

        return data_out, mean, std

    @staticmethod
    def standardize_3D_with_mean_and_sd(data, mean, std):

        data_out = copy.deepcopy(data)

        for i in range(data_out.shape[2]):
            data_out[:, :, i] = (data[:, :, i] - mean[i]) / std[i]

        data_out[np.isnan(data_out)] = 0

        return data_out

    @staticmethod
    def inverse_standardize_3D_with_mean_and_sd(data, mean, std):

        data_out = copy.deepcopy(data)

        for i in range(data_out.shape[2]):
            data_out[:, :, i] = (data[:, :, i] * std[i]) + mean[i]

        return data_out
