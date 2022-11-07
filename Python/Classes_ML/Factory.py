from tabulate import tabulate
import numpy as np
from sklearn import preprocessing

class Dictionary_factory():

    @staticmethod
    def split_classes(z, y, unique_class):

        """
        Split Z_train data into classes dictionary
        """

        z_class = {}
        num_data_class = {}

        for i, unique in enumerate(unique_class):

            indices = [i for i, x in enumerate(y) if x == unique]
            z_class[unique] = z[indices]
            num_data_class[unique] = z_class[unique].shape[0]

        return z_class, num_data_class

class Dataframe_factory():

    @staticmethod
    def add_subject_session_index_to_df(df, columns=[], new_columns=[]):

        for col, new_col in zip(columns, new_columns):

            unique = np.unique(df[col].values)

            index_dictionary_invert = {v: k for k, v in dict(enumerate(unique)).items()}

            df[new_col] = df[col].replace(index_dictionary_invert)
            df[new_col] = df[new_col].astype(int)

        return df

    @staticmethod
    def add_image_index_to_df(df, new_columns=""):

        subjects = np.unique(df["Subject_name"].values)
        labels = np.unique(df["True"].values)

        index = []
        for s in subjects:
            for unique in labels:
                n = df[(df['Subject_name'] == s) & (df['True'] == unique)].values.shape[0]
                index.extend([i for i in range(n)])

        df[new_columns] = index

        return df

    @staticmethod
    def extract_subject_session_from_df(df, subject, session):

        df = df[(df['Subject_name'] == subject) & (df['Session_name'] == session)]

        return df

    @staticmethod
    def add_index_columns(df, new_columns=""):

        df[new_columns] = [i for i in range(df.shape[0])]

        return df

    @staticmethod
    def get_columns_as_2D_array(df, columns=[]):

        array = np.empty(shape=(df.shape[0], len(columns)))

        for i, col in enumerate(columns):
            array[:, i] = df[col].values

        return array

    @staticmethod
    def display_df(df):

        print(tabulate(df, headers='keys', tablefmt='psql'))