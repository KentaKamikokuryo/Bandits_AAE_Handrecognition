import numpy as np
import pandas as pd
from tabulate import tabulate

class Ranked():

    def __init__(self, model_name, search_type="", path="", metric_order="ascending", metric_name="regressor"):

        """ Set path information to the experiment

        Parameters
        ----------
        metric_order: str
            "ascending" or "descending"
            "ascending" for metrics such as accuracy or reward (the higher, the better)
            "descending" for metrics such as RMSE or regret (the lower, the better)

        metric_name: str
            name of the metrics used
        """

        self.search_type = search_type
        self.model_name = model_name

        self.hyperparameters_list = []
        self.metric_mean_list = []
        self.metric_std_list = []
        self.count_params_list = []
        self.id_list = []

        self.path = path

        self.metric_order = metric_order
        self.metric_name = metric_name

        self.name = "Model_" + str(self.model_name) + "_hyperparameter_search_" + self.search_type

    def add(self, hyperparameter, metric_mean=0, metric_std=0, count_params=0, id: str = "none"):

        self.hyperparameters_list.append(hyperparameter)
        self.metric_mean_list.append(metric_mean)
        self.metric_std_list.append(metric_std)
        self.count_params_list.append(count_params)
        self.id_list.append(id)

    def ranked(self):

        if self.metric_name == "ascending":
            idx = np.argsort(self.metric_mean_list)[::-1]
        elif self.metric_name == "descending":
            idx = np.argsort(self.metric_mean_list)
        else:
            idx = np.argsort(self.metric_mean_list)

        self.hyperparameters_list_sorted = np.array(self.hyperparameters_list)[idx].tolist()
        self.metric_mean_list_sorted = np.array(self.metric_mean_list)[idx].tolist()
        self.metric_std_list_sorted = np.array(self.metric_std_list)[idx].tolist()
        self.count_params_list_sorted = np.array(self.count_params_list)[idx].tolist()
        self.id_list_sorted= np.array(self.id_list)[idx].tolist()

        print('Hyperparameters_list has been ranked')
        self.hyperparameter_best = self.hyperparameters_list_sorted[0]

    def display_ranked_metric(self):

        df = pd.DataFrame.from_dict(self.hyperparameters_list_sorted)
        df["Metric_mean"] = self.metric_mean_list_sorted
        df["Metric_sd"] = self.metric_std_list_sorted
        df["count_params"] = self.count_params_list_sorted
        df["id"] = self.id_list_sorted

        print(tabulate(df, headers='keys', tablefmt='psql'))

    def save_ranked_metric(self):

        df = pd.DataFrame.from_dict(self.hyperparameters_list_sorted)
        df["Metric_mean"] = self.metric_mean_list_sorted
        df["Metric_sd"] = self.metric_std_list_sorted
        df["count_params"] = self.count_params_list_sorted
        df["id"] = self.id_list_sorted

        name = "ranked_" + self.name + ".csv"
        df.to_csv(self.path + name)
        print('hyperparameters_list_sorted has been saved to: ' + self.path + name)

    def save_best_hyperparameter(self):

        name = self.name + "_best.npy"
        np.save(self.path + name, self.hyperparameter_best)
        print('hyperparameter_best has been saved to: ' + self.path + name)

    def load_best_hyperparameters(self):

        name = self.name + "_best.npy"
        self.hyperparameter_best = np.load(self.path + name, allow_pickle=True).item()
        print('hyperparameter_best has been load from: ' + self.path + name)

    def load_ranked_metric(self):

        name = "ranked_" + self.name + ".csv"
        self.hyperparameters_list_sorted = pd.read_csv(self.path + name)
        print('hyperparameters_list_sorted has been load from:  ' + self.path + name)

    def display_loaded_ranked_metric(self):

        df = pd.DataFrame.from_dict(self.hyperparameters_list_sorted)
        print(tabulate(df, headers='keys', tablefmt='psql'))
