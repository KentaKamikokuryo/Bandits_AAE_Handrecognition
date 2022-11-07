from sklearn.model_selection import train_test_split

class Utilities_ML:

    @staticmethod
    def generate_K_fold_index(samples_train_valid, tvt_info):

        n_K_fold = tvt_info["n_K_fold"]
        tvt_ratio = tvt_info["tvt_ratio"]

        samples_train_K = {}
        samples_valid_K = {}

        for K in range(n_K_fold):
            samples_train_K[K], samples_valid_K[K] = train_test_split(samples_train_valid, test_size=tvt_ratio[1],
                                                                      shuffle=True)

        return samples_train_K, samples_valid_K

    @staticmethod
    def generate_K_fold_test_index(samples, tvt_info):

        n_K_fold = tvt_info["n_K_fold"]
        tvt_ratio = tvt_info["tvt_ratio"]

        samples_train_valid, samples_test = train_test_split(samples, test_size=tvt_ratio[2], shuffle=True)

        samples_train_K = {}
        samples_valid_K = {}

        for K in range(n_K_fold):
            samples_train_K[K], samples_valid_K[K] = train_test_split(samples_train_valid, test_size=tvt_ratio[1],
                                                                      shuffle=True)

        return samples_train_K, samples_valid_K, samples_test
