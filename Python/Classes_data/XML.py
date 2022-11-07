import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from PIL import Image
from xml.dom import minidom
from Classes_data.Coordinates_to_image import Coordinates_to_image

class ImageXML:

    @staticmethod
    def show_image(array, name=""):

        fig = plt.figure(figsize=(18, 12))

        plt.imshow(np.transpose(array), cmap='gray')
        plt.show()
        plt.title(name)

        return fig

    @staticmethod
    def plot_dict_image(dict, title="", display=True):

        n_subplot = len(dict.keys())
        n_columns = 5

        n_rows = n_subplot // n_columns
        n_rows += n_subplot % n_columns

        position = range(1, n_subplot + 1)

        if display:
            plt.ion()
        else:
            plt.ioff()

        fig = plt.figure(figsize=(18, 12))

        for i in range(n_subplot):

            ax = fig.add_subplot(n_rows, n_columns, position[i])

            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticklabels([])

            ax.set_title(list(dict.keys())[i])
            array = dict[list(dict.keys())[i]]
            data = np.flipud(np.transpose(array))

            plt.imshow(data, cmap='gray')

        fig.suptitle(title)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.05, hspace=0.3)

        if display:
            plt.show(block=False)
            plt.pause(0.05)

        return fig

    @staticmethod
    def save_image(array, path, name):

        data = np.flipud(np.transpose(array))

        img = Image.fromarray(data)
        img = img.convert('RGB')
        img.save(path + name + ".png")

        img.close()

class XML:

    @staticmethod
    def load_XML(path):

        """ Load XML data

        Parameters
        ----------
        path: str
            path to the folder where the session is

        Returns
        -------
        session_data_dict: dict
            dictionary of point data from /xml file (Recorded from Unity)
        """

        session_data_dict = {}

        filenames = os.listdir(path)
        filenames = [f for f in filenames if ".meta" not in f]

        filenames = [f[:-4] for f in filenames if ".xml"in f]

        for f in filenames:

            items = minidom.parse(path + f + ".xml").getElementsByTagName('Point')

            coordinates = np.empty(shape=(len(items), 2))

            for i, elem in enumerate(items):
                coordinates[i, 0] = elem.attributes['X'].value
                coordinates[i, 1] = elem.attributes['Y'].value

            session_data_dict[f] = coordinates

        return session_data_dict

    @staticmethod
    def load_score(path):

        """ Load .csv data created in Unity during a session

        Parameters
        ----------
        path: str
            path to the folder where the session is

        Returns
        -------
        df_session: df
            dataframe with information such as name of the file, true and predicted label
            (predicted in unity by hand recognition algorithm)
        """

        df_score = pd.read_csv(path + "Scores.csv", delimiter="\t", header=None)
        df_score.columns = ["Names", "Scores"]

        df_times = pd.read_csv(path + "Times.csv", delimiter="\t", header=None)
        df_times.columns = ["Names", "Times"]

        df_classes = pd.read_csv(path + "Classes.csv", delimiter="\t", header=None)
        df_classes.columns = ["True", "Predicted"]

        # Gather information
        df_session = pd.concat([df_score, df_times, df_classes], axis=1)
        df_session = df_session.loc[:, ~df_session.columns.duplicated()]

        df_session = df_session[['Names', 'True', 'Predicted', 'Scores', 'Times']]

        true_classes = []

        for true_class in df_session["True"].values:
            true_classes.append(true_class.split("_")[0])

        df_session["True"] = df_session["Names"]
        df_session["True"] = true_classes

        # print(tabulate(df_session, headers='keys', tablefmt='psql'))

        return df_session

    @staticmethod
    def convert(data_dict):

        """ Load XML data and create image from it

        Parameters
        ----------
        data_dict: dict
            dictionary of point data from /xml file (Recorded from Unity)

        Returns
        -------
        session_scaled_dict: dict
            dictionary of scaled point data (Integer number like the image)
        session_image_dict: dict
            dictionary of image
        """

        data_scaled_dict = {}
        data_image_dict = {}

        for k in data_dict.keys():

            coordinates_to_image = Coordinates_to_image()
            data_scaled_dict[k], data_image_dict[k] = coordinates_to_image.convert(coordinates=data_dict[k])

        return data_scaled_dict, data_image_dict

class SessionXML:

    @staticmethod
    def load_reference(path_to_xml):

        """ Load XML data and create image from it. These data are specifically from the reference data used in Unity to perform the hand recognition

        Parameters
        ----------
        path_to_xml: str
            path to the folder where the .xml references are

        Returns
        -------
        reference_data_dict: dict
            dictionary of point data from /xml file (Recorded from Unity)
        reference_scaled_dict: dict
            dictionary of scaled point data (Integer number like the image)
        reference_image_dict: dict
            dictionary of image
        """

        # Load dataset (reference one)
        reference_data_dict = XML.load_XML(path_to_xml)
        reference_scaled_dict, reference_image_dict = XML.convert(reference_data_dict)

        return reference_data_dict, reference_scaled_dict, reference_image_dict

    @staticmethod
    def load_session(path_to_xml, subject_name, session_name):

        """ Load XML data and create image from it

        Parameters
        ----------
        path_to_xml: str
            path to the folder where the .xml file from the session is
        subject_name: str
            name of the subject
        session_name: str
            name of the session

        Returns
        -------
        session_data_dict: dict
            dictionary of point data from /xml file (Recorded from Unity)
        session_scaled_dict: dict
            dictionary of scaled point data (Integer number like the image)
        session_image_dict: dict
            dictionary of image
        df_session: df
            dataframe with information such as name of the file, true and predicted label (predicted in unity by hand recognition algorithm)
            None if not found
        """

        session_data_dict = XML.load_XML(path=path_to_xml)
        session_scaled_dict, session_image_dict = XML.convert(session_data_dict)

        try:

            df_session = XML.load_score(path=path_to_xml)
            # Add information about session and subject name in the dataframe
            df_session['Session_name'] = pd.Series([session_name for s in range(df_session.shape[0])],
                                                   index=df_session.index)
            df_session['Subject_name'] = pd.Series([subject_name for s in range(df_session.shape[0])],
                                                   index=df_session.index)

        except:

            print("could not find score at: " + path_to_xml)

            df_session = None

        return session_data_dict, session_scaled_dict, session_image_dict, df_session

    @staticmethod
    def load_session_npy(path_to_xml, path_to_npy, subject_name, session_name):

        """ Load XML data and create image from it

        Parameters
        ----------
        path_to_xml: str
            path to the folder where the .xml file from the session is
        path_to_npy: str
            path to the folder where the .npy file from the session is
        subject_name: str
            name of the subject
        session_name: str
            name of the session

        Returns
        -------
        session_data_dict: dict
            dictionary of point data from .xml file (Recorded from Unity)
        session_scaled_dict: dict
            dictionary of scaled point data (Integer number like the image)
        session_image_dict: dict
            dictionary of image
        df_session: df
            dataframe with information such as name of the file, true and predicted label (predicted in unity by hand recognition algorithm)
            None if not found
        """

        session_data_dict = {}

        filenames = os.listdir(path_to_npy)
        filenames = [f for f in filenames if ".meta" not in f]

        filenames = [f[:-4] for f in filenames if ".npy" in f]

        for f in filenames:
            session_data_dict[f] = np.load(path_to_npy + f + ".npy", allow_pickle=True)

        session_scaled_dict, session_image_dict = XML.convert(session_data_dict)

        try:

            df_session = XML.load_score(path=path_to_xml)
            # Add information about session and subject name in the dataframe
            df_session['Session_name'] = pd.Series([session_name for s in range(df_session.shape[0])],
                                                   index=df_session.index)
            df_session['Subject_name'] = pd.Series([subject_name for s in range(df_session.shape[0])],
                                                   index=df_session.index)

        except:

            df_session = None

        return session_data_dict, session_scaled_dict, session_image_dict, df_session
