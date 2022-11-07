from tabulate import tabulate
import pandas as pd
import os, itertools
from distutils.dir_util import copy_tree

class Path_info:

    def __init__(self):

        """ Class use to get all information relative to path (Data, DB, figure ...)
        """

        self.cwd = os.getcwd()

        # Get path one level above the root of the project
        self.path_parent_project = os.path.abspath(os.path.join(self.cwd, os.pardir))

        # Path to the data
        self.path_data = self.path_parent_project + "\\Data\\"

        print("cwd: " + self.cwd)
        print("path_parent_project: " + self.path_parent_project)
        print("path_data: " + self.path_data)

        # Figure
        # self.path_folder_figure_bandit = self.path_parent_project + "\\Figures_bandit\\"
        # self.path_folder_figure_bandit_bayesian = self.path_parent_project + "\\Figures_bandit_bayesian\\"
        #
        # self.path_folder_figure_test_bandit_best = self.path_parent_project + "\\Figures_bandit_best\\"
        # self.path_folder_figure_test_bandit_bayesian_best = self.path_parent_project + "\\Figures_bandit_bayesian_best\\"
        # self.path_folder_figure_test = self.path_parent_project + "\\Figures_bandit_test\\"

        # if not (os.path.exists(self.path_folder_figure_bandit)):
        #     os.makedirs(self.path_folder_figure_bandit)
        # if not (os.path.exists(self.path_folder_figure_bandit_bayesian)):
        #     os.makedirs(self.path_folder_figure_bandit_bayesian)
        # if not (os.path.exists(self.path_folder_figure_test_bandit_best)):
        #     os.makedirs(self.path_folder_figure_test_bandit_best)
        # if not (os.path.exists(self.path_folder_figure_test_bandit_bayesian_best)):
        #     os.makedirs(self.path_folder_figure_test_bandit_bayesian_best)
        # if not (os.path.exists(self.path_folder_figure_test)):
        #     os.makedirs(self.path_folder_figure_test)

    def set_experiment(self, experiment_name: str):

        """ Set path information to the experiment

        Parameters
        ----------
        experiment_name: str
            name of the experiment
        session_train_name: str
            name of the session used as reference

        Returns
        -------
        dictionary
            dictionary of subject and sessions
        """

        self.experiment_name = experiment_name
        self.path_data_experiment = self.path_data + self.experiment_name + "\\"

        print("path_data_experiment: " + self.path_data_experiment)

        if not (os.path.exists(self.path_data_experiment)):
            os.makedirs(self.path_data_experiment)

        subjects_info = {}
        subject_folder = [d for d in os.listdir(self.path_data_experiment) if os.path.isdir(os.path.join(self.path_data_experiment, d))]

        for s in subject_folder:
            path_subject = self.path_data_experiment + s + "\\"
            sessions = [d for d in os.listdir(path_subject) if os.path.isdir(os.path.join(path_subject, d))]
            subjects_info[s] = []
            for session in sessions:
                if 'ReferenceDominant' in session or 'ReferenceNonDominant' in session or 'Reference' in session:
                    pass
                else:
                    subjects_info[s].append(session)

        return subjects_info

    def set_subject_information(self, subject_name: str, session_name: str):

        """ Set path information to  subject and session

        Parameters
        ----------
        subject_name: str
            name of the subject
        session_name: str
            name of the session used as reference
        """

        self.subject_name = subject_name
        self.path_subject = self.path_data_experiment + self.subject_name + "\\"

        self.session_name = session_name
        self.path_subject_session = self.path_subject + str(self.session_name)
        self.path_subject_session_xml = self.path_subject_session + '\\XMLGesturesDominant\\'
        self.path_subject_session_textures = self.path_subject_session + '\\TexturesDominant\\'

        print("")
        print("path_subject: " + self.path_subject)
        print("path_subject_session: " + self.path_subject_session)
        print("path_subject_session_xml: " + self.path_subject_session_xml)
        print("path_subject_session_textures: " + self.path_subject_session_textures)

    def make_session_dirs(self, subject_name, session_name):

        self.set_subject_information(subject_name=subject_name, session_name=session_name)
        self.test_path_session = self.path_subject + self.session_name + "\\"  # not in Path() class

        self.test_path_session_npy = self.test_path_session + "NpyDominant\\"
        self.test_path_session_XML = self.test_path_session + "XMLGesturesDominant\\"  # to avoid error
        self.test_path_session_textures = self.test_path_session + "TexturesDominant\\"

        if not os.path.exists(self.test_path_session):
            os.makedirs(self.test_path_session)
            os.makedirs(self.test_path_session_npy)
            os.makedirs(self.test_path_session_XML)  # to avoid error
            os.makedirs(self.test_path_session_textures)

class DB_info():

    def __init__(self):

        path_info = Path_info()
        self.path_parent_project = path_info.path_parent_project

        # Path to the DB (Outside of the github since it may become too big)
        self.path_DB = self.path_parent_project + '\\Database\\'

        train_parameters = {'experiment_name': ["Artificial_X", "Artificial_Y", "Artificial_R",
                                                "Artificial_XY", "Artificial_XR", "Artificial_YR",
                                                "Artificial_XYR"],
                            'experiment_name_target': ["Artificial_T"],
                            'subjects_train_target': [[]],
                            'subjects_train_valid': [[]],
                            'subjects_test': [[]]}

        keys, values = zip(*train_parameters.items())
        train_configurations_0 = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(train_configurations_0)
        print(tabulate(df, headers='keys', tablefmt='psql'))

        for i, t in enumerate(train_configurations_0):

            experiment_name = train_configurations_0[i]["experiment_name"]
            experiment_name_target = train_configurations_0[i]["experiment_name_target"]

            subjects_info = path_info.set_experiment(experiment_name=experiment_name)
            subjects_info_target = path_info.set_experiment(experiment_name=experiment_name_target)

            self.subjects_name = list(subjects_info.keys())
            self.n_subjects = len(self.subjects_name)

            self.subjects_name_target = list(subjects_info_target.keys())
            self.n_subjects_target = len(self.subjects_name_target)

            self.tvt_info = dict(tvt_ratio=[0.7, 0.15, 0.15], n_K_fold=5, n_outer_loop=0, type='sample')

            self.n_subjects_train_valid = int(self.n_subjects * self.tvt_info["tvt_ratio"][0] + self.tvt_info["tvt_ratio"][1])
            self.n_subjects_test = self.n_subjects - self.n_subjects_train_valid

            train_configurations_0[i]["subjects_train_target"] = ["Subject" + str(i + 1).zfill(2) for i in range(0, self.n_subjects_target)]
            train_configurations_0[i]["subjects_train_valid"] = ["Subject" + str(i + 1).zfill(2) for i in range(0, self.n_subjects_train_valid)]
            train_configurations_0[i]["subjects_test"] = ["Subject" + str(i + 1).zfill(2) for i in range(self.n_subjects_train_valid, self.n_subjects)]

            train_configurations_0[i]["Description"] = "Artificial"
            train_configurations_0[i]["DB_N"] = i + 000

        self.DB_N_configurations = train_configurations_0

        df = pd.DataFrame.from_dict(self.DB_N_configurations)
        print(tabulate(df, headers='keys', tablefmt='psql'))

        self.path_paper = self.path_parent_project + "\\Paper\\"

        if not os.path.exists(self.path_paper):
            os.makedirs(self.path_paper)

    def get_DB_info(self, DB_N):

        self.DB_N = DB_N

        self.path_DB_N = self.path_DB + "DB_" + str(DB_N) + "\\"

        if not os.path.exists(self.path_DB_N):
            os.makedirs(self.path_DB_N)

        self.DB_N_configuration = self.DB_N_configurations[DB_N]

        self.subjects_train_valid_DB = self.DB_N_configuration["subjects_train_valid"]
        self.subjects_test_DB = self.DB_N_configuration["subjects_test"]
        self.experiment_name = self.DB_N_configuration["experiment_name"]
        self.experiment_name_target = self.DB_N_configuration["experiment_name_target"]

        self.path_ML = self.path_parent_project + "\\ML\\"
        self.path_DL = self.path_parent_project + "\\DL\\"
        self.path_RL = self.path_parent_project + "\\RL\\"

        self.path_figure_ML_DB_N = self.path_ML + "Figures\\DB_" + str(DB_N) + "\\"
        self.path_figure_DL_DB_N = self.path_DL + "Figures\\DB_" + str(DB_N) + "\\"
        self.path_figure_RL_DB_N = self.path_RL + "Figures\\DB_" + str(DB_N) + "\\"

        self.path_results_ML_DB_N = self.path_ML + "Results\\DB_" + str(DB_N) + "\\"
        self.path_results_DL_DB_N = self.path_DL + "Results\\DB_" + str(DB_N) + "\\"
        self.path_results_RL_DB_N = self.path_RL + "Results\\DB_" + str(DB_N) + "\\"

        self.path_model_ML_DB_N = self.path_ML + "Models\\DB_" + str(DB_N) + "\\"
        self.path_model_DL_DB_N = self.path_DL + "Models\\DB_" + str(DB_N) + "\\"
        self.path_model_RL_DB_N = self.path_RL + "Models\\DB_" + str(DB_N) + "\\"

        self.path_search_models_ML_DB_N = self.path_ML + "Search_models\\DB_" + str(DB_N) + "\\"
        self.path_search_models_DL_DB_N = self.path_DL + "Search_models\\DB_" + str(DB_N) + "\\"
        self.path_search_models_RL_DB_N = self.path_RL + "Search_models\\DB_" + str(DB_N) + "\\"

        if not os.path.exists(self.path_figure_ML_DB_N):
            os.makedirs(self.path_figure_ML_DB_N)
        if not os.path.exists(self.path_figure_DL_DB_N):
            os.makedirs(self.path_figure_DL_DB_N)
        if not os.path.exists(self.path_figure_RL_DB_N):
            os.makedirs(self.path_figure_RL_DB_N)

        if not os.path.exists(self.path_results_ML_DB_N):
            os.makedirs(self.path_results_ML_DB_N)
        if not os.path.exists(self.path_results_DL_DB_N):
            os.makedirs(self.path_results_DL_DB_N)
        if not os.path.exists(self.path_results_RL_DB_N):
            os.makedirs(self.path_results_RL_DB_N)

        if not os.path.exists(self.path_model_ML_DB_N):
            os.makedirs(self.path_model_ML_DB_N)
        if not os.path.exists(self.path_model_DL_DB_N):
            os.makedirs(self.path_model_DL_DB_N)
        if not os.path.exists(self.path_model_RL_DB_N):
            os.makedirs(self.path_model_RL_DB_N)

        if not os.path.exists(self.path_search_models_ML_DB_N):
            os.makedirs(self.path_search_models_ML_DB_N)
        if not os.path.exists(self.path_search_models_DL_DB_N):
            os.makedirs(self.path_search_models_DL_DB_N)
        if not os.path.exists(self.path_search_models_RL_DB_N):
            os.makedirs(self.path_search_models_RL_DB_N)

    def get_DB_RL_info(self, bandit_behavior):

        self.bandit_behavior = bandit_behavior

        self.path_ML = self.path_parent_project + "\\ML\\"
        self.path_DL = self.path_parent_project + "\\DL\\"
        self.path_RL = self.path_parent_project + "\\RL\\"

        self.path_figure_RL_DB_N = self.path_RL + "Figures\\DB_" + str(bandit_behavior) + "\\"
        self.path_results_RL_DB_N = self.path_RL + "Results\\DB_" + str(bandit_behavior) + "\\"
        self.path_model_RL_DB_N = self.path_RL + "Models\\DB_" + str(bandit_behavior) + "\\"
        self.path_search_models_RL_DB_N = self.path_RL + "Search_models\\DB_" + str(bandit_behavior) + "\\"

        if not os.path.exists(self.path_figure_RL_DB_N):
            os.makedirs(self.path_figure_RL_DB_N)
        if not os.path.exists(self.path_results_RL_DB_N):
            os.makedirs(self.path_results_RL_DB_N)
        if not os.path.exists(self.path_model_RL_DB_N):
            os.makedirs(self.path_model_RL_DB_N)
        if not os.path.exists(self.path_search_models_RL_DB_N):
            os.makedirs(self.path_search_models_RL_DB_N)

    def get_DB_info_as_dataframe(self):

        """ Get DB information as pandas df

        Parameters
        ----------
        DB: int
            index of the database

        Returns
        -------
        pandas df
            DB information as pandas df
        """

        df = pd.DataFrame.from_dict(self.DB_N_configurations)

        return df

    def display_DB_info(self):

        """ Display all DB information with tabulate

        Parameters
        ----------
        none

        Returns
        -------
        none
        """

        df = pd.DataFrame.from_dict([self.DB_N_configurations[self.DB_N]])
        print(tabulate(df, headers='keys', tablefmt='psql'))

    def display_DBs_info(self):

        """ Display all DB information with tabulate

        Parameters
        ----------
        none

        Returns
        -------
        none
        """

        df = pd.DataFrame.from_dict(self.DB_N_configurations)
        print(tabulate(df, headers='keys', tablefmt='psql'))
