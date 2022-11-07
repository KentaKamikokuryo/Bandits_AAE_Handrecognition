import os
import rapidjson
from abc import ABC, abstractmethod
from Classes_game import Type

class Load():
    
    def __init__(self, info):
        
        self.info = info
        
        files_exist = self.check()

        if files_exist:

            jsn = self.load_json_files()

            self.load_all(jsn_event=jsn[0], jsn_frame=jsn[1], jsn_xr=jsn[2])

        else:

            self.all_data = None

    def get_data(self):

        return self.all_data

    def check(self):
        
        if os.path.exists(self.info['path_event']):
            pass
        else:
            print("Event data does not exist...")
            return False
        
        if os.path.exists(self.info['path_frame']):
            pass
        else:
            print("Frame data does not exist...")
            return False
        
        if os.path.exists(self.info['path_XR']):
            pass
        else:
            print("XR data does not exist...")
            return False
        
        return True
    
    def load_json_files(self):

        with open(self.info['path_event']) as f:
            jsn_event = rapidjson.load(f)

        with open(self.info['path_frame']) as f:
            jsn_frame = rapidjson.load(f)

        with open(self.info['path_XR']) as f:
            jsn_xr = rapidjson.load(f)

        return jsn_event, jsn_frame, jsn_xr

    def load_all(self, jsn_event, jsn_frame, jsn_xr):

        self.all_data = All(json_event=jsn_event, json_frame=jsn_frame, json_xr=jsn_xr)

class All():

    def __init__(self, json_event, json_frame, json_xr):

        self.get_info_player(json=json_event)
        self.event = Event(dict_json=json_event)
        self.frame = Frame(dict_json=json_frame)
        self.xr = XR(dict_json=json_xr)

    def get_info_player(self, json: dict):
        self.game_name = json["gameName"]
        self.session_id = json["sessionID"]
        self.score = json["score"]
        self.score_time = json["scoreTime"]
        self.data_time_txt = json["dateTimeTxt"]

class Event(ABC):

    @abstractmethod
    def __init__(self, dict_json: dict):
        pass

class Frame(ABC):

    @abstractmethod
    def __init__(self, dict_json: dict):
        pass

class XR():

    def __init__(self, dict_json):

        self.dict = dict_json

        self.load_all_XR()

    def load_all_XR(self):

        self.load_controller_positions()
        self.load_controller_quaternions()
        self.load_button_press()
        self.load_press_values()
        self.load_press_axes()

    def load_controller_positions(self):

        self.right_controller_positions = Controller_Positions(self.dict['rightControllerPosition'])
        self.left_controller_positions = Controller_Positions(self.dict['leftControllerPosition'])
        self.head_positions = Controller_Positions(self.dict['headsetPosition'])

    def load_controller_quaternions(self):

        self.quaternions_right_controller = Controller_Quaternions(self.dict['rightControllerQuaternion'])
        self.quaternions_left_controller = Controller_Quaternions(self.dict['leftControllerQuaternion'])
        self.quaternions_head = Controller_Quaternions(self.dict['headsetQuaternion'])

    def load_button_press(self):

        self.press_A = Button_Press(self.dict['APress'])
        self.press_B = Button_Press(self.dict['BPress'])
        self.press_X = Button_Press(self.dict['XPress'])
        self.press_Y = Button_Press(self.dict['YPress'])

        self.press_trigger_right = Button_Press(self.dict['triggerRightPress'])
        self.press_tergger_left = Button_Press(self.dict['triggerLeftPress'])

        self.press_grip_right = Button_Press(self.dict['gripRightPress'])
        self.press_grip_left = Button_Press(self.dict['gripLeftPress'])

        self.press_axis_right = Button_Press(self.dict['axisRightPress'])
        self.press_axis_left = Button_Press(self.dict['axisLeftPress'])

        self.press_menu = Button_Press(self.dict['menuPress'])

    def load_press_values(self):

        self.values_trigger_right = Press_Values(self.dict['triggerRightValues'])
        self.values_trigger_left = Press_Values(self.dict['triggerLeftValues'])

        self.values_grip_right = Press_Values(['gripRightValues'])
        self.values_grip_left = Press_Values(['gripLeftValues'])

    def load_press_axes(self):

        self.axes_right_values = Press_Axes(self.dict['axisRightValues'])
        self.axes_left_values = Press_Axes(self.dict['axisLeftValues'])


class Controller_Positions():

    def __init__(self, positions):
        self.positions = [Type.Vector3(position['x'], position['y'], position['z'])
                          for position in positions]

class Controller_Quaternions():

    def __init__(self, quaternions):
        self.quaternions = [Type.Quaternion(quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w'])
                            for quaternion in quaternions]

class Button_Press():

    def __init__(self, press):
        self.press = press

class Press_Values():

    def __init__(self, values):
        self.values = values

class Press_Axes():

    def __init__(self, axes):
        self.axes = [Type.Vector2(axis['x'], axis['y']) for axis in axes]
