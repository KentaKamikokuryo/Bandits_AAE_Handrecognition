from Classes_data.Info import Path_info
import os

class PathInfoRL():

    def __init__(self):

        path_info = Path_info()

        self.path_parent_project = path_info.path_parent_project
        self.path_child_project = self.path_parent_project + "\\Bandit"
        self.path_result = self.path_child_project + "\\Bandit_all_result\\"
        self.path_search = self.path_child_project + "\\Bandit_all_search\\"

        if not os.path.exists(self.path_child_project):
            os.makedirs(self.path_child_project)
        if not os.path.exists(self.path_result):
            os.makedirs(self.path_result)
        if not os.path.exists(self.path_search):
            os.makedirs(self.path_search)

    def get_path_result(self, behavior_name: str, arm: int, solver_name: str):

        path_result_behavior = self.path_result + str(behavior_name) + "\\" + "Arm_" + str(arm) + "\\" + solver_name + "\\"

        if not os.path.exists(path_result_behavior):
            os.makedirs(path_result_behavior)

        return path_result_behavior

    def get_path_search(self, behavior_name: str, arm: int):

        path_search_behavior = self.path_search + str(behavior_name) + "\\" + "Arm_" + str(arm) + "\\"

        if not os.path.exists(path_search_behavior):
            os.makedirs(path_search_behavior)

        return path_search_behavior

