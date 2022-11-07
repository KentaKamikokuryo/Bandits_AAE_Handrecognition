from Classes_data.Info import DB_info
import numpy as np

class DB():

    '''
    Load database data
    '''

    def __init__(self, DB_N: int):

        self.DB_N = DB_N
        self.db_info = DB_info()
        self.db_info.get_DB_info(DB_N=self.DB_N)

    def load_behaviors(self):

        """ Get the behavior of artificial data associated with DB data

        Parameters
        ----------
        None

        Returns
        -------
        behaviors: dict
            Here is an example of how the dictionary is organized
            First:
                behaviors.keys() will give:
                dict_keys(['Subject01', 'Subject02', 'Subject03', 'Subject04', 'Subject05', 'Subject06', 'Subject07'])
                It represents each subjects
            Then:
                behaviors['Subject01'].keys() will give:
                dict_keys(['Circle', 'Rectangle', 'Square'])
                It represents each label for the key subject:
            Then:
                behaviors['Subject01']['Circle'].keys() will give:
                dict_keys(['ratio_X_change', 'ratio_Y_change', 'rotation_change', 'hyper_DA_global'])
                ratio_X_change, ratio_Y_change, rotation_change represent the artificial behavior.
                    hyper_DA_global represent the hyperparameters used to create the artificial behavior.
            Then:
                behaviors['Subject01']['Circle']["ratio_X_change"].shape will give (60)
       """

        name = "behaviors.npy"
        behaviors = np.load(self.db_info.path_DB_N + name, allow_pickle=True).item()

        return behaviors

    def get_DB_data(self, reshape_to_1D: bool = True):

        """ Get the DB data

        Parameters
        ----------
        reshape_to_1D: bool
            if True, the data are reshaped to 1D (Used for ML or MLP)
            if False, the data keep their original 2D shape (Used for CNN and RNN layer as first layers)

        Returns
        -------
        XY: dict
            Here is an example of how the dictionary is organized
            First:
                XY_train_valid.keys() will give:
                dict_keys(['X', 'Y', 'df', 'hyper_DA_global', 'ratio_X_change', 'ratio_Y_change', 'rotation_change'])
                It represents each subjects
            First:
                XY_train_valid['X'].keys() will give:
                dict_keys(['Subject01', 'Subject02' ...])
                It represents each subjects
            Then:
                XY_fit['Subject01'].keys() will give:
                dict_keys(['Circle', 'Rectangle', 'Square'])
                It represents each session for the key subject:
            Then:
                XY_train_valid["X"]["Subject01"]["Circle"].shape will give (60, 64)
                Meaning there is 60 movements. 64 represent the X and Y dimension of the hand movement data concatenate vertically.
        X_image: dict
            XY_transform is organized the same way as XY_fit
        """

        # Get train_valid data
        name = "XY.npy"
        XY = np.load(self.db_info.path_DB_N + name, allow_pickle=True).item()
        # name = "X_image.npy"
        # X_image = np.load(self.db_info.path_DB_N + name, allow_pickle=True).item()

        name = "X_image.npy"
        X_image = None

        # Reorganize the information data
        for subject in XY["X"].keys():
            for unique in XY["X"][subject].keys():
                X = XY["X"][subject][unique]
                if reshape_to_1D:
                    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
                XY["X"][subject][unique] = X

        return XY, X_image