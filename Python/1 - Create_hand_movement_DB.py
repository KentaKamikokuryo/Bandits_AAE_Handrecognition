import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
from Classes_ML.Standardize import Standardize
from Classes_data.Info import DB_info
from tabulate import tabulate
from Classes_data.XML import ImageXML, SessionXML
import numpy as np
from Classes_data.Info import Path_info
from Classes_data.DB import DB

class Create_artifical_DB():

    '''
    This class may only be used only here for artificial hand data.
    We will make a new one in the future with real data (So we will have several database with various index)
    '''

    def __init__(self, DB_N: int):

        self.DB_N = DB_N
        self.db_info = DB_info()
        self.db_info.get_DB_info(DB_N=self.DB_N)

        self.path_info = Path_info()
        self.path_info_target = Path_info()

        self.subjects_info = self.path_info.set_experiment(experiment_name=self.db_info.experiment_name)
        self.subjects_info_target = self.path_info_target.set_experiment(experiment_name=self.db_info.experiment_name_target)

        self.db_info.display_DBs_info()
        self.db_info.display_DB_info()

    def create_database(self):

        self.subjects_info_train_valid = {k: self.subjects_info[k] for k in self.db_info.subjects_train_valid_DB}
        self.subjects_info_test = {k: self.subjects_info[k] for k in self.db_info.subjects_test_DB}

        self.data_X_train_valid, self.data_Y_train_valid, self.df_scores_train_valid, self.image_train_valid = Create_artifical_DB.get_data(path_info=self.path_info,
                                                                                                                                                 subjects_info=self.subjects_info_train_valid,
                                                                                                                                                 suffix="")

        self.data_X_test, self.data_Y_test, self.df_scores_test, self.image_test = Create_artifical_DB.get_data(path_info=self.path_info,
                                                                                                                     subjects_info=self.subjects_info_test,
                                                                                                                     suffix="")

        self.data_target_X_train, self.data_target_Y_train, self.df_target_scores_train, self.target_image_train = Create_artifical_DB.get_data(path_info=self.path_info_target,
                                                                                                                                                     subjects_info=self.subjects_info_target,
                                                                                                                                                     suffix="_T")

        # Compute global mean and std for train set (dataset)
        X_train_valid = np.concatenate([self.data_X_train_valid[k][k2] for k in self.data_X_train_valid.keys() for k2 in self.data_X_train_valid[k].keys()], axis=0)
        y_train_valid = np.concatenate([self.data_Y_train_valid[k][k2] for k in self.data_Y_train_valid.keys() for k2 in self.data_Y_train_valid[k].keys()]).squeeze()

        # Standardize X data (2D vector)
        inds_X = Standardize.generate_standardized_index(data_shape=X_train_valid.shape[-1], standardization_type='all')
        X_train_valid_std, mean, std = Standardize.standardize_3D(data=X_train_valid, inds=inds_X)

        unique_class = np.unique(y_train_valid)
        self.behaviors = Create_artifical_DB.get_behavior(path_info=self.path_info,
                                                          subjects_info=self.subjects_info,
                                                          unique_class=unique_class, suffix="")
        self.behaviors_target = Create_artifical_DB.get_behavior(path_info=self.path_info_target,
                                                                 subjects_info=self.subjects_info_target,
                                                                 unique_class=unique_class,
                                                                 suffix="_T")

        # Combine
        df_list = [self.df_scores_train_valid[k][k2] for k in self.df_scores_train_valid.keys() for k2 in self.df_scores_train_valid[k].keys()]
        df_list.extend([self.df_scores_test[k][k2] for k in self.df_scores_test.keys() for k2 in self.df_scores_test[k].keys()])
        df_list.extend([self.df_target_scores_train[k][k2] for k in self.df_target_scores_train.keys() for k2 in self.df_target_scores_train[k].keys()])
        df_all = pd.concat(df_list, axis=0)
        df_all = df_all.reset_index()

        self.data_X = {**self.data_X_train_valid, **self.data_X_test, **self.data_target_X_train}
        self.data_Y = {**self.data_Y_train_valid, **self.data_Y_test, **self.data_target_Y_train}
        self.data_image = {**self.image_train_valid, **self.image_test, **self.target_image_train}

        self.subjects_info_all = {**self.subjects_info}
        for k in self.subjects_info_target.keys():
            self.subjects_info_all[k + "_T"] = self.subjects_info_target[k]

        self.behaviors_all = {}
        for k in self.behaviors.keys():
            self.behaviors_all[k] = {**self.behaviors[k], **self.behaviors_target[k]}

        self.data_XY = {}

        keys = ["X", "X_img", "Y", "df"]
        keys.extend(list(self.behaviors.keys()))

        for k in keys:
            self.data_XY[k] = {}

        for subject in self.data_X.keys():

            for k in keys:
                self.data_XY[k][subject] = {}

            for unique in unique_class:

                self.data_XY["X"][subject][unique] = []
                self.data_XY["X_img"][subject][unique] = []

                for session_name in self.subjects_info_all[subject]:
                    index = np.where(self.data_Y[subject][session_name] == unique)  # Index for the current label (unique)
                    self.data_XY["X"][subject][unique].append(Standardize.standardize_3D_with_mean_and_sd(self.data_X[subject][session_name][index], mean=mean, std=std))
                    self.data_XY["X_img"][subject][unique].append(self.data_image[subject][session_name][index])

                # self.data_XY["X_img"][subject][unique] = np.concatenate([x[np.newaxis] for x in self.data_X_image[subject][unique]], axis=0) / 255.0
                self.data_XY["X_img"][subject][unique] = np.concatenate([x for x in self.data_XY["X_img"][subject][unique]], axis=0) / 255.0  # Normalize image

                self.data_XY["X"][subject][unique] = np.concatenate([x for x in self.data_XY["X"][subject][unique]], axis=0)
                self.data_XY["Y"][subject][unique] = np.array([unique for i in range(self.data_XY["X"][subject][unique].shape[0])])
                self.data_XY["df"][subject][unique] = df_all[(df_all['Subject_name'] == subject) & (df_all['True'] == unique)]

                for behavior in self.behaviors_all.keys():
                    self.data_XY[behavior][subject][unique] = self.behaviors_all[behavior][subject][unique]

        # Display to check
        print(self.data_XY.keys())
        print(self.data_XY["X"].keys())
        print("Shape of X data: " + str(self.data_XY["X"]["Subject01"]["Cube"].shape))
        print("Shape of Y data: " + str(self.data_XY["Y"]["Subject01"]["Cube"].shape))
        print("Shape of ratio data: " + str(len(self.data_XY["ratio_X_change"]["Subject01"]["Cube"])))
        print("Shape of image data: " + str(self.data_XY["X_img"]["Subject01"]["Cube"].shape))

        name = "X_mean.npy"
        np.save(self.db_info.path_DB_N + name, mean)
        name = "X_sd.npy"
        np.save(self.db_info.path_DB_N + name, std)
        name = "XY.npy"
        np.save(self.db_info.path_DB_N + "" + name, self.data_XY)

    @staticmethod
    def get_data(path_info: Path_info, subjects_info: dict, suffix: str, plot_it=False):

        # Get target data
        data_X = {}
        data_Y = {}
        df_scores = {}
        data_image = {}

        for subject in subjects_info.keys():

            data_X[subject + suffix] = {}
            data_Y[subject + suffix] = {}
            df_scores[subject + suffix] = {}
            data_image[subject + suffix] = {}

            for session_name in subjects_info[subject]:

                path_info.set_subject_information(subject_name=subject, session_name=session_name)

                path_to_npy = path_info.path_subject_session + "\\NpyDominant\\"
                path_to_xml = path_info.path_subject_session + "\\XMLGesturesDominant\\"

                session_data_dict, session_scaled_dict, session_image_dict, df_session = SessionXML.load_session_npy(
                    path_to_npy=path_to_npy, path_to_xml=path_to_xml,
                    subject_name=subject, session_name=session_name)

                df_session['Subject_name'] = df_session['Subject_name'].astype(str) + suffix
                # print(tabulate(df_session, headers='keys', tablefmt='psql'))

                data_X[subject + suffix][session_name], _ = Create_artifical_DB.convert_dict_to_ML_data(d=session_data_dict)
                data_Y[subject + suffix][session_name] = df_session["True"].values
                df_scores[subject + suffix][session_name] = df_session
                data_image[subject + suffix][session_name] = np.concatenate([session_image_dict[k][np.newaxis, :] for k in df_session["Names"].values], axis=0)

                if plot_it:
                    ImageXML.plot_dict_image(dict=session_image_dict, title="Dataset - " + subject + suffix)

        return data_X, data_Y, df_scores, data_image

    @staticmethod
    def get_behavior(path_info: Path_info, subjects_info: dict, unique_class: [str], suffix=""):

        # Load behavior information
        behaviors = {}
        behaviors_list = []

        for subject in subjects_info.keys():
            # Load behavior information
            behaviors[subject + suffix] = {}
            for i, unique in enumerate(unique_class):
                path = path_info.path_data_experiment + subject + "\\"
                behavior_all = np.load(path + unique + "_behavior" + ".npy", allow_pickle=True).item()
                behaviors[subject + suffix][unique] = behavior_all
                behaviors_list.extend(list(behavior_all.keys()))

        behaviors_new = {}

        # Change organization
        for behavior in np.unique(behaviors_list):
            behaviors_new[behavior] = {}
            for subject in subjects_info.keys():
                behaviors_new[behavior][subject + suffix] = {}
                for i, unique in enumerate(unique_class):
                    behaviors_new[behavior][subject + suffix][unique] = behaviors[subject + suffix][unique][behavior]

        return behaviors_new

    @staticmethod
    def convert_dict_to_ML_data(d):

        """ Convert dictionary of hand movement to a 3D numpy array

        Parameters
        ----------
        d: dict
            dictionary of point data with keys representing the movement name. The number of keys represent n_sample

        Returns
        -------
        X: ndarray
            Array of hand movement with shape: n_sample x height x 2
        """

        X = []
        Y = []

        # Get data
        for k in d.keys():
            X.append(d[k])
            Y.append(k)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

db_info = DB_info()

DB_Ns = [i for i in range(len(db_info.DB_N_configurations))]

for DB_N in DB_Ns:

    # Create DB
    ca_DB = Create_artifical_DB(DB_N=DB_N)
    ca_DB.create_database()

    # Get DB (test)
    db = DB(DB_N=DB_N)
    XY, X_image = db.get_DB_data()
