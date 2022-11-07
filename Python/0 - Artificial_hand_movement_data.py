import random
import matplotlib.pyplot as plt
from Classes_data.Info import Path_info
import os, itertools, shutil
from Classes_data.Artificial_session import Create_artificial_session
import math
from tabulate import tabulate
from Classes_data.XML import ImageXML, SessionXML, XML
import numpy as np
import pandas as pd
import cProfile, pstats
from Classes_results.PlotPaper import PlotPaper
from Classes_data.Info import DB_info

class Manager():

    def __init__(self, behavior, experiment_name: str = "", n_subject=5):

        """ Manager class to create artificial 2D hand movement from existing data.
        After instantiating the class use the class methods self.run()

        Parameters
        ----------
        DB_N: str
            database number the data will be used for
        experiment_name: str
            name of the experiment
        n_subject: int
            number of artificial subject to create from these data
        """

        self.behavior = behavior
        self.experiment_name = experiment_name + "_" + ''.join(map(str, behavior))
        self.n_subject = n_subject

        if experiment_name == "Artificial":
            self.labels = ["Cube", "Cylinder", "Heart", "Infinite", "Sphere", "Triangle"]
            self.n_label = 3  # Number of label for each of them

        # Information about the subject that will be used to generate artificial data
        if "T" in self.experiment_name:
            self.is_target = True
        else:
            self.is_target = False

        self.path_info = Path_info()
        self.path_info.set_experiment(experiment_name=self.experiment_name)
        self.path_data_experiment = self.path_info.path_data_experiment

        self.delete_previous_data()

    def get_reference_data(self):

        experiment_reference_name = "References"
        subject_reference = "Subject9999"
        session_name = "9999"

        experiment_reference_name = "References"
        session_name = "9999"

        subjects_reference = ["Subject9999", "Subject9998", "Subject9997"]

        # Pick a subject randomly
        p = np.random.randint(0, len(subjects_reference))
        subject_reference = subjects_reference[p]

        # Get reference information
        path_info_reference = Path_info()
        path_info_reference.set_experiment(experiment_name=experiment_reference_name)
        path_info_reference.set_subject_information(subject_name=subject_reference, session_name=session_name)

        # Get reference data
        session_data_dict, session_scaled_dict, session_image_dict, df_session = SessionXML.load_session(path_to_xml=path_info_reference.path_subject_session_xml,
                                                                                                         subject_name=subject_reference,
                                                                                                         session_name=session_name)

        print(tabulate(df_session, headers='keys', tablefmt='psql'))
        # ImageXML.plot_dict_image(dict=session_image_dict, title="Dataset test - " + subject_reference)

        return session_data_dict, session_scaled_dict, session_image_dict, df_session

    def run(self):

        labels_reference = ['Cube_0', 'Cube_1', 'Cube_2',
                            'Cylinder_0', 'Cylinder_1', 'Cylinder_2',
                            'Heart_0', 'Heart_1', 'Heart_2',
                            'Infinite_0', 'Infinite_1', 'Infinite_2',
                            'Sphere_0', 'Sphere_1', 'Sphere_2',
                            'Triangle_0', 'Triangle_1', 'Triangle_2']

        # Start subject index. Since there is three labels for each class let's create n_subject for each of them
        start_indices = []

        for label in self.labels:
            start_indices.extend([self.n_subject * i for i in range(self.n_label)])

        index = 0
        start_index = start_indices[index]

        for start_index, k in zip(start_indices, labels_reference):

            self.session_data_dict, self.session_scaled_dict, self.session_image_dict, self.df_session = self.get_reference_data()

            print("Label: " + k)

            hyperparameter_DA_globals = Manager.generate_artificial_behavior(n_subject=self.n_subject, start_index=start_index, behaviors_name=self.behavior, is_target=self.is_target)
            self.label = k.split(sep="_")[0]
            self.generate(XY=self.session_data_dict[k], hyperparameter_DA_globals=hyperparameter_DA_globals)

        ##############
        # Plot paper #
        db_info = DB_info()
        path_paper = db_info.path_paper
        names = ["Cube_0", "Cylinder_0", "Heart_0", "Infinite_0", "Sphere_0", "Triangle_0"]
        reference_image = {}
        for name in names:
            reference_image[name] = self.session_image_dict[name].copy()
        PlotPaper.plot_reference_image(reference_image=reference_image, path=path_paper, figure_name="Reference")
        ##############

    def delete_previous_data(self):

        path = self.path_data_experiment
        print(path + " folder removed")
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)

    @staticmethod
    def generate_artificial_behavior(n_subject=10, start_index=0, behaviors_name=[], is_target=False):

        # For now, let's keep the global linear behavior. We will add exponential and log global behavior in the future
        # Here, we are going to create n_subject for each label

        # ch: what is the change
        # rn: random noise
        # rot: rotation
        # aw: amplitude warping (random distortion with wave)

        hyperparameter_DA_globals_ch_type_choices = dict(distortion_type=["linear", "linear_stair"],
                                                         b_ratio_X=[0.9, 1.0],
                                                         b_ratio_Y=[0.9, 1.0],
                                                         rotation_type=["linear"],
                                                         rotation=[-5, 5])

        hyperparameter_DA_session_choices = dict(sigma_rn=[0.001, 0.005, 0.01],
                                                 sigma_aw=[0.001, 0.005, 0.01],
                                                 knot_aw=[2, 3],
                                                 rot_range=[[-5, 5]],
                                                 augmentation_types=[['rn', 'rot', 'aw']])

        if is_target:

            hyperparameter_DA_globals_ch_type_choices = dict(distortion_type=["linear"],
                                                             b_ratio_X=[0.9, 1.0],
                                                             b_ratio_Y=[0.9, 1.0],
                                                             rotation_type=["linear"],
                                                             rotation=[-5, 5])

            hyperparameter_DA_session_choices = dict(sigma_rn=[0.001],
                                                     sigma_aw=[0.001],
                                                     knot_aw=[2],
                                                     rot_range=[[-5, 5]],
                                                     augmentation_types=[['rn', 'rot', 'aw']])

        else:

            for behavior in behaviors_name:

                if behavior == "X":
                    hyperparameter_DA_globals_ch_type_choices["b_ratio_X"] = [0.2, 0.3, 0.4, 0.5]

                elif behavior == "Y":
                    hyperparameter_DA_globals_ch_type_choices["b_ratio_Y"] = [0.2, 0.3, 0.4, 0.5]

                elif behavior == "R":
                    hyperparameter_DA_globals_ch_type_choices["rotation"] = [-45, -30, -15, 0, 15, 30, 45]

        keys, values = zip(*hyperparameter_DA_globals_ch_type_choices.items())
        hyperparameter_DA_globals_ch_type_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        keys, values = zip(*hyperparameter_DA_session_choices.items())
        hyperparameter_DA_session_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        hyperparameter_DA_globals_choices = dict(n_session=[20],
                                                 n_movement_per_session=[3],
                                                 ch_type=hyperparameter_DA_globals_ch_type_all_combination,
                                                 session=hyperparameter_DA_session_all_combination)

        keys, values = zip(*hyperparameter_DA_globals_choices.items())
        hyperparameter_DA_globals_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # df = pd.DataFrame.from_dict(hyperparameter_DA_globals_all_combination)
        # print(tabulate(df, headers='keys', tablefmt='psql'))

        # Choose n random subject
        hyperparameter_DA_globals_all_combination = random.sample(hyperparameter_DA_globals_all_combination, n_subject)

        for i in range(n_subject):
            hyperparameter_DA_globals_all_combination[i]["name"] = "Subject" + str(i + 1 + start_index).zfill(2)

        df = pd.DataFrame.from_dict(hyperparameter_DA_globals_all_combination)
        print(tabulate(df, headers='keys', tablefmt='psql'))

        return hyperparameter_DA_globals_all_combination

    def plot_and_save_behavior(self, hyper_DA_global, ratio_X_change, ratio_Y_change, rotation_change, display=False):

        subject_name = hyper_DA_global["name"]

        if display:
            plt.ion()
        else:
            plt.ioff()

        # Save figure
        path = self.path_data_experiment + "\\" + subject_name + "\\"

        if not (os.path.exists(path)):
            os.makedirs(path)

        # Plot behavior
        fig = plt.figure(figsize=(18, 12))
        plt.plot(ratio_X_change, color="b", label="linear X change")
        plt.plot(ratio_Y_change, color="r", label="linear Y change")
        plt.plot(rotation_change, color="g", label="2D rotation")
        plt.xlabel("Sessions")
        plt.legend()

        fig.savefig(path + self.label + "_behavior.png", dpi=100)

        if display:
            plt.show(block=False)
            plt.pause(0.05)

        plt.close(fig)

    def plot_and_save_images(self, hyper_DA_global, sessions_data_new_dict, sessions_data_new_aug_dict, display=False):

        if display:
            plt.ion()
        else:
            plt.ioff()

        n_session = hyper_DA_global['n_session']
        n_movement_per_session = hyper_DA_global['n_movement_per_session']

        # Save figure
        subject_name = hyper_DA_global["name"]
        path = self.path_data_experiment + "\\" + subject_name + "\\"

        if not (os.path.exists(path)):
            os.makedirs(path)

        # Reshape dictionary
        sessions_data_new_dict_reshaped = {}
        sessions_data_new_aug_dict_reshaped = {}

        count = 0

        for i in range(n_session):
            for j in range(n_movement_per_session):
                sessions_data_new_dict_reshaped[count] = sessions_data_new_dict[i][j]
                sessions_data_new_aug_dict_reshaped[count] = sessions_data_new_aug_dict[i][j]
                count += 1

        max_plot_per_figure = 20
        n_image = len(sessions_data_new_aug_dict_reshaped.keys())
        n_figure = math.ceil(n_image / max_plot_per_figure)
        start = 0
        end = max_plot_per_figure

        for i in range(n_figure):

            temp = {k: sessions_data_new_dict_reshaped[k] for k in list(sessions_data_new_dict_reshaped)[start:end]}

            # Only distortion on X, Y an rotation
            session_scaled_dict, session_image_dict = XML.convert(temp)
            fig = ImageXML.plot_dict_image(dict=session_image_dict, title="Dataset test", display=display)
            fig.savefig(path + self.label + "_distorted_session_image_dict_" + str(i) + ".png", dpi=100)

            if display:
                plt.show(block=False)
                plt.pause(0.05)

            plt.close(fig)

            temp = {k: sessions_data_new_aug_dict_reshaped[k] for k in list(sessions_data_new_aug_dict_reshaped)[start:end]}

            # With random noise and amplitude warping
            session_scaled_dict, session_image_dict = XML.convert(temp)
            fig = ImageXML.plot_dict_image(dict=session_image_dict, title="Dataset test", display=display)
            fig.savefig(path + self.label + "_non_distorted_session_image_dict_" + str(i) + ".png", dpi=100)

            if display:
                plt.show(block=False)
                plt.pause(0.05)

            plt.close(fig)

            start += max_plot_per_figure
            end += max_plot_per_figure

            if end > n_image:
                end = n_image

        session_scaled_dict, session_image_dict = XML.convert(sessions_data_new_aug_dict_reshaped)

        session = n_session // 2
        kk = list(session_image_dict.keys())[session * n_movement_per_session:session * n_movement_per_session + n_movement_per_session]

        # More precise plot
        fig, ax = plt.subplots(figsize=(18, 12))
        for i, k in enumerate(kk):
            ax.plot(session_scaled_dict[k][:, 0], session_scaled_dict[k][:, 1], "b")

        fig.savefig(path + self.label + "_distorted_precise_session_image_dict.png", dpi=100)

        if display:
            plt.show(block=False)
            plt.pause(0.05)

        plt.close(fig)

    def generate(self, XY, hyperparameter_DA_globals):

        for hyper_DA_global in hyperparameter_DA_globals:

            # TODO: 2
            cas = Create_artificial_session(XY, hyper_DA_global=hyper_DA_global)
            cas.generate_change_factor()

            # TODO: 3
            sessions_data_new_dict, sessions_data_new_aug_dict = cas.run()

            self.plot_and_save_behavior(hyper_DA_global=hyper_DA_global,
                                        ratio_X_change=cas.ratio_X_change,
                                        ratio_Y_change=cas.ratio_Y_change,
                                        rotation_change=np.divide(cas.rotation_change, 180.) * np.pi, display=False)

            # self.plot_and_save_images(hyper_DA_global=hyper_DA_global,
            #                           sessions_data_new_dict=sessions_data_new_dict,
            #                           sessions_data_new_aug_dict=sessions_data_new_aug_dict, display=False)

            for session_name in sessions_data_new_dict.keys():  # Session loop
                self.save_data(session_data_new_dict=sessions_data_new_aug_dict[session_name], hyper_DA_global=hyper_DA_global, session_name=session_name)

            behavior_all = self.save_behavior(cas, hyper_DA_global=hyper_DA_global)

            ##############
            # Plot paper #
            db_info = DB_info()
            path_paper = db_info.path_paper
            subject_name = hyper_DA_global["name"]

            if "_".join(self.behavior) == "X":

                figure_name = self.label + "_" + subject_name + "_" + "_".join(self.behavior) + "_image"
                PlotPaper.plot_augmented_image(augmented_image=sessions_data_new_aug_dict, path=path_paper, figure_name=figure_name)
                figure_name = self.label + "_" + subject_name + "_" + "_".join(self.behavior) + "_behavior"
                PlotPaper.plot_behavior(behavior_all=behavior_all, path=path_paper, figure_name=figure_name)

            ##############

    def save_behavior(self, cas, hyper_DA_global):

        subject_name = hyper_DA_global["name"]
        path = self.path_data_experiment + subject_name + "\\"

        ratio_X_change = cas.ratio_X_change
        ratio_Y_change = cas.ratio_Y_change
        rotation_change = np.divide(cas.rotation_change, 180.) * np.pi

        name = self.label + "_behavior"

        behavior_all = {}
        behavior_all["ratio_X_change"] = ratio_X_change
        behavior_all["ratio_Y_change"] = ratio_Y_change
        behavior_all["rotation_change"] = rotation_change
        behavior_all["hyper_DA_global"] = hyper_DA_global

        np.save(path + name + ".npy", behavior_all, allow_pickle=True)

        print("Behavior saved to: " + path)

        return behavior_all

    def save_data(self, session_data_new_dict, hyper_DA_global, session_name):

        # Directly save as numpy file in session folder
        # Save textures as well

        subject_name = hyper_DA_global["name"]
        path = self.path_data_experiment + subject_name + "\\"

        session_scaled_new_dict, session_image_new_dict = XML.convert(session_data_new_dict)

        test_path_session_npy = path + "\\" + str(session_name + 1).zfill(4) + "\\NpyDominant\\"
        test_path_session_XML = path + "\\" + str(session_name + 1).zfill(4) + "\\XMLGesturesDominant\\"
        test_path_session_textures = path + "\\" + str(session_name + 1).zfill(4) + "\\TexturesDominant\\"

        if not (os.path.exists(test_path_session_npy)):
            os.makedirs(test_path_session_npy)
        if not (os.path.exists(test_path_session_XML)):
            os.makedirs(test_path_session_XML)
        if not (os.path.exists(test_path_session_textures)):
            os.makedirs(test_path_session_textures)

        # Check if files already exist. Load and append new data to it if exist
        df_score_path = test_path_session_XML + "Scores.csv"
        df_times_path = test_path_session_XML + "Times.csv"
        df_classes_path = test_path_session_XML + "Classes.csv"

        if os.path.exists(df_score_path):
            df_score = pd.read_csv(df_score_path, delimiter="\t", header=None)
            df_score.columns = ["Names", "Scores"]
        else:
            df_score = pd.DataFrame(columns=["Names", "Scores"])

        if os.path.exists(df_times_path):
            df_times = pd.read_csv(df_times_path, delimiter="\t", header=None)
            df_times.columns = ["Names", "Times"]
        else:
            df_times = pd.DataFrame(columns=["Names", "Times"])

        if os.path.exists(df_classes_path):
            df_classes = pd.read_csv(df_classes_path, delimiter="\t", header=None)
            df_classes.columns = ["Predicted", "True"]
        else:
            df_classes = pd.DataFrame(columns=["Predicted", "True"])

        for i, k in enumerate(list(session_data_new_dict.keys())):

            name = self.label + "_" + str(i + 1).zfill(2)

            ImageXML.save_image(array=session_image_new_dict[k], path=test_path_session_textures, name=name)

            np.save(test_path_session_npy + name + ".npy", session_data_new_dict[k])  # Save 2D vector

            # Create fake .csv file (For convenience in the Create_DB()
            df_score_temp = pd.DataFrame({"Names": [name], "Scores": [1.0]})
            df_score = df_score.append(df_score_temp)

            df_times_temp = pd.DataFrame({"Names": [name], "Times": [1.0]})
            df_times = df_times.append(df_times_temp)

            df_classes_temp = pd.DataFrame({"Predicted": [name], "True": [self.label]})
            df_classes = df_classes.append(df_classes_temp)

        df_score.to_csv(path_or_buf=test_path_session_XML + "Scores.csv", sep='\t', index=False, header=False)
        df_times.to_csv(path_or_buf=test_path_session_XML + "Times.csv", sep='\t', index=False, header=False)
        df_classes.to_csv(path_or_buf=test_path_session_XML + "Classes.csv", sep='\t', index=False, header=False)

behaviors_list = [["T"],        # Target
                  ["X"],        # X change
                  ["Y"],        # Y change
                  ["R"],        # Rotation change
                  ["X", "Y"],   # Combination
                  ["X", "R"],
                  ["Y", "R"],
                  ["X", "Y", "R"]]

for behavior in behaviors_list:

    if behavior == ["T"]:
        n_subject = 1
    else:
        n_subject = 5

    experiment_name = "Artificial"
    manager = Manager(behavior=behavior, experiment_name=experiment_name, n_subject=n_subject)
    manager.run()


