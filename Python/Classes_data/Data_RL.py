from Classes_data.DB import DB
import numpy as np

from Classes_data.Info import DB_info

class Data_Z_RL:

    def __init__(self, db_info: DB_info, model_name: str):

        self.db_info = db_info
        self.DB_N = self.db_info.DB_N
        self.model_name = model_name

    def get_DB_N(self):

        self.db_info = DB_info()
        self.db_info.get_DB_info(DB_N=self.DB_N)

        # Get database and latent space
        self.db = DB(DB_N=self.DB_N)
        self.XY, self.X_image = self.db.get_DB_data()

        self.Z_train_valid, self.Z_test, self.Z_centroids_dict = self.db.load_latent_space_DB(model_name=self.model_name)

        self.unique_label = np.unique([list(self.Z_test["d_std"][k].keys()) for k in self.Z_test["d_std"].keys()]).tolist()
        self.n_action = len(self.unique_label)
        self.n_episode = 10

        n_iteration_subjects = {}

        for k in self.Z_test["d_std"].keys():
            iterations = [len(self.Z_test["d_std"][k][k2]) for k2 in self.Z_test["d_std"][k].keys()]
            n_iteration_subjects[k] = np.max(iterations)

        self.n_iteration = np.max([n_iteration_subjects[k] for k in n_iteration_subjects.keys()])

        # Organize data for bandits as dictionary
        self.d_mu = {}
        self.d_std = {}

        for k in self.Z_test["d_std"].keys():
            self.d_mu[k] = np.empty(shape=(self.n_action , self.n_iteration))
            self.d_std[k] = np.empty(shape=(self.n_action , self.n_iteration))
            for i, k2 in enumerate(self.Z_test["d_std"][k].keys()):
                self.d_mu[k][i, :] = self.Z_test["d_std"][k][k2].tolist()
                self.d_std[k][i, :] = [0.1 for i in range(self.n_iteration)]

    def display(self):

        print("The number of bandits ", self.n_action)
        print("The number of iterations ", self.n_iteration)
        print("The number of episodes ", self.n_episode)
