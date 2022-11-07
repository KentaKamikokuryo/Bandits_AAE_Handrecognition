import numpy as np
import random
import warnings
import pandas as pd
from tabulate import tabulate

class Genetic_algorithm:

    def __init__(self, n_generation, n_agent_generation,  nn_param_choices, display_info=True):

        self.n_generation = n_generation
        self.n_agent_generation = n_agent_generation
        self.nn_param_choices = nn_param_choices

        self.display_info = display_info

        self.metric_mean_gen = []
        self.metric_sd_gen = []

        self.metric_mean_all = []
        self.metric_sd_all = []

        self.networks_all = []

        self.networks_gen_sorted_dict = {}
        self.metric_mean_gen_sorted_dict = {}
        self.metric_sd_gen_sorted_dict = {}

    def set_selection_methods(self, selection_parameters: dict):

        if selection_parameters["name"] == "simple":

            if "ascending" in selection_parameters["metric_order"] or "descending" in selection_parameters["metric_order"]:
                self.metric_order = selection_parameters["metric_order"]
            else:
                warnings.warn("Problem with metrics set - " + "Acceptable choice: " + str(["ascending", "descending"]))

            self.selection_method_name = selection_parameters["name"]
            self.mutate_chance = selection_parameters["mutate_chance"]
            self.random_select = selection_parameters["random_select"]
            self.keep_best = selection_parameters["keep_best"]

        elif selection_parameters["name"] == "tournament":

            if "ascending" in selection_parameters["metric_order"] or "descending" in selection_parameters["metric_order"]:
                self.metric_order = selection_parameters["metric_order"]
            else:
                warnings.warn("Problem with metrics set - " + "Acceptable choice: " + str(["ascending", "descending"]))

            self.selection_method_name = selection_parameters["name"]
            self.mutate_chance = selection_parameters["mutate_chance"]
            self.n_agent_fight = selection_parameters["n_agent_fight"]
            self.keep = selection_parameters["keep"]
            self.keep_best = selection_parameters["keep_best"]

        else:

            examples = [{'name': 'simple', 'metric_order': 'descending', 'mutate_chance': 0.05, 'random_select': 0.05,  'keep_best': 0.05},
                        {'name': 'tournament', 'metric_order': 'descending', 'mutate_chance': 0.05, 'n_agent_fight': 2, 'keep': 0.2, 'keep_best': 0.05}]
            warnings.warn("Problem with selection method set - " + "Example choice: " + str(examples))

    def breed(self, female, male):

        children = []

        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the children
            for param in self.nn_param_choices:
                child[param] = random.choice([female[param], male[param]])

            # Check each key and randomly mutated with probability mutate_chance
            for gene in child.keys():
                if self.mutate_chance > random.random():
                    param = random.choice(list(self.nn_param_choices[gene]))
                    child[gene] = param

            children.append(child)

        return children

    def evolve_simple(self, agents, metrics_mean_list, metrics_sd_list):

        # Sort metrics
        if "descending" in self.metric_order:
            idx = np.argsort(metrics_mean_list)
        elif "ascending" in self.metric_order:
            idx = np.argsort(metrics_mean_list)[::-1]

        metrics_mean_list_sorted = np.array(metrics_mean_list)[idx].tolist()
        metrics_sd_list_sorted = np.array(metrics_sd_list)[idx].tolist()
        networks_sorted = np.array(agents)[idx].tolist()

        if self.display_info:
            print("Genetic Algorithm Advanced - Generation " + str(self.current_generation) + " results ")
            if "descending" in self.metric_order:
                print("Generation average metrics: %.4f" % (np.mean(metrics_mean_list)))
            elif "ascending" in self.metric_order:
                print("Generation average metrics: %.4f" % (np.mean(metrics_mean_list) * 100))

            df = pd.DataFrame.from_dict(agents)
            df["Metrics mean"] = metrics_mean_list_sorted
            df["Metrics sd"] = metrics_sd_list_sorted
            print(tabulate(df, headers='keys', tablefmt='psql'))

        # Get the number we want to keep for the next gen.
        retain_length = int(len(networks_sorted) * self.keep_best)
        parents = networks_sorted[:retain_length]
        parents_mean_metrics = metrics_mean_list_sorted[:retain_length]
        parents_sd_metrics = metrics_sd_list_sorted[:retain_length]

        # Randomly keep some
        idx = random.sample(range(0, len(networks_sorted)), int(self.random_select * len(networks_sorted)))
        parents_random = np.array(networks_sorted)[idx].tolist()
        parents_mean_metrics_random = np.array(metrics_mean_list_sorted)[idx].tolist()
        parents_sd_metrics_random = np.array(metrics_sd_list_sorted)[idx].tolist()

        # Network used for next generation
        parents.extend(parents_random)
        parents_mean_metrics.extend(parents_mean_metrics_random)
        parents_sd_metrics.extend(parents_sd_metrics_random)

        if self.display_info:
            print("Genetic Algorithm Advanced - Selected parents for generation " + str(self.current_generation))
            df = pd.DataFrame.from_dict(parents)
            df["Metrics mean"] = parents_mean_metrics
            df["Metrics sd"] = parents_sd_metrics
            print(tabulate(df, headers='keys', tablefmt='psql'))

        parents = self.evolve(parents)

        return parents

    def evolve_tournament(self, agents, metrics_mean_list, metrics_sd_list):

        # Sort metrics
        if "descending" in self.metric_order:
            idx = np.argsort(metrics_mean_list)
        elif "ascending" in self.metric_order:
            idx = np.argsort(metrics_mean_list)[::-1]

        metrics_mean_list_sorted = np.array(metrics_mean_list)[idx].tolist()
        metrics_sd_list_sorted = np.array(metrics_sd_list)[idx].tolist()
        networks_sorted = np.array(agents)[idx].tolist()

        # Parents we use for the next generation
        retain_length = int(len(networks_sorted) * self.keep)
        parents = []
        parents_mean_metrics = []
        parents_sd_metrics = []

        for i in range(retain_length):

            idx = random.sample(range(0, len(networks_sorted)), self.n_agent_fight)
            metrics_fight = np.array(metrics_mean_list_sorted)[idx].tolist()

            if "descending" in self.metric_order:
                select_id = idx[np.argmin(metrics_fight)]
            elif "ascending" in self.metric_order:
                select_id = idx[np.argmax(metrics_fight)]

            parents.append(networks_sorted[select_id])
            parents_mean_metrics.append(metrics_mean_list_sorted[select_id])
            parents_sd_metrics.append(metrics_sd_list_sorted[select_id])

        # Always keep the n best one
        retain_length = int(len(networks_sorted) * self.keep_best)
        parents_best = networks_sorted[:retain_length]
        parents_mean_metrics_best = metrics_mean_list_sorted[:retain_length]
        parents_sd_metrics_best = metrics_sd_list_sorted[:retain_length]

        # Network used for next generation
        parents.extend(parents_best)
        parents_mean_metrics.extend(parents_mean_metrics_best)
        parents_sd_metrics.extend(parents_sd_metrics_best)

        parents = self.evolve(parents)

        return parents

    def evolve(self, parents):

        parents_length = len(parents)
        desired_length = self.n_agent_generation - parents_length
        children = []

        # Add children, which are bred from two remaining agents.
        while len(children) < desired_length:

            # Get two random parents
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Check if they are not the same
            if male != female:
                male = parents[male]
                female = parents[female]
                babies = self.breed(male, female)
                for baby in babies:
                    if len(children) < desired_length:  # Don't grow larger than desired length.
                        children.append(baby)

        parents.extend(children)

        return parents

    def generate_first_population(self):

        self.agents = []

        for _ in range(0, self.n_agent_generation):
            network = {}
            for key in self.nn_param_choices:
                network[key] = random.choice(self.nn_param_choices[key])
            self.agents.append(network)

        self.current_generation = 0

        if self.display_info:
            print("Genetic Algorithm Advanced - First generation")
            df = pd.DataFrame.from_dict(self.agents)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        return self.agents

    def generate_ith_generation(self, metrics_mean_list, metrics_sd_list):

        self.metric_mean_gen.append(np.mean(metrics_mean_list))
        self.metric_sd_gen.append(np.std(metrics_mean_list))

        self.metric_mean_all.extend(metrics_mean_list)
        self.metric_sd_all.extend(metrics_sd_list)

        self.networks_all.extend(self.agents)

        # Do the evolution.
        if self.current_generation < self.n_generation:
            if self.selection_method_name == "simple":
                self.agents = self.evolve_simple(agents=self.agents, metrics_mean_list=metrics_mean_list, metrics_sd_list=metrics_sd_list)
            elif self.selection_method_name == 'tournament':
                self.agents = self.evolve_tournament(agents=self.agents, metrics_mean_list=metrics_mean_list, metrics_sd_list=metrics_sd_list)

        if self.display_info:
            print("Genetic Algorithm Advanced - New generation " + str(self.current_generation))
            df = pd.DataFrame.from_dict(self.agents)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        self.current_generation += 1

        return self.agents

    def keep_generation_information(self, metrics_mean_list, metrics_sd_list):

        # Keep generation information in dictionary
        # Sort metrics
        if "descending" in self.metric_order:
            idx = np.argsort(metrics_mean_list)
        elif "ascending" in self.metric_order:
            idx = np.argsort(metrics_mean_list)[::-1]

        self.networks_gen_sorted_dict[self.current_generation] = np.array(self.agents)[idx].tolist()
        self.metric_mean_gen_sorted_dict[self.current_generation] = np.array(metrics_mean_list)[idx].tolist()
        self.metric_sd_gen_sorted_dict[self.current_generation] = np.array(metrics_sd_list)[idx].tolist()

    def display_generation_info(self, generation_n=0):

        agents = self.networks_gen_sorted_dict[generation_n]
        metrics_mean_list_sorted = self.metric_mean_gen_sorted_dict[generation_n]
        metrics_sd_list_sorted = self.metric_sd_gen_sorted_dict[generation_n]

        print("Genetic Algorithm Advanced - Generation " + str(generation_n + 1) + "\\" + str(self.n_generation) + " results ")
        if "descending" in self.metric_order:
            print("Generation average metrics: %.4f" % (np.mean(metrics_mean_list_sorted)))
        elif "ascending" in self.metric_order:
            print("Generation average metrics: %.4f" % (np.mean(metrics_mean_list_sorted) * 100))

        df = pd.DataFrame.from_dict(agents)
        df["Metrics mean"] = metrics_mean_list_sorted
        df["Metrics sd"] = metrics_sd_list_sorted
        print(tabulate(df, headers='keys', tablefmt='psql'))
