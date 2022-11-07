import os

from Classes_game.Load import Load, All, Event, Frame, XR
from Classes_game import Type
import Classes_game.Load as load
from enum import Enum

class Load_Fishing(Load):

    def load_all(self, jsn_event, jsn_frame, jsn_xr):

        self.all_data = AllFishing(json_event=jsn_event, json_frame=jsn_frame, json_xr=jsn_xr)

class AllFishing(All):

    def __init__(self, json_event, json_frame, json_xr):

        self.get_info_player(json=json_event)
        self.event = EventFishing(dict_json=json_event)
        self.frame = FrameFishing(dict_json=json_frame)
        self.xr = XRFishing(dict_json=json_xr)

    def get_info_player(self, json: dict):
        self.game_name = json["gameName"]
        self.session_id = json["sessionID"]
        self.score = json["score"]
        self.score_time = json["scoreTime"]
        self.data_time_txt = json["dateTimeTxt"]

class EventFishing(Event):

    def __init__(self, dict_json: dict):

        self.food = Name(dict_json["foodName"])
        self.food_initial_grid = Position(dict_json["foodInitialGridPosition"])
        self.food_initial = Position(dict_json["foodInitialPosition"])
        self.appear = Time(dict_json["timeAppear"])
        self.disappear = Time(dict_json["timeDisappear"])

class FrameFishing(Frame):

    def __init__(self, dict_json: dict):

        self.food = Name(dict_json["foodName"])
        self.food_initial = Position(dict_json["foodInitialGridPosition"])
        self.food = Position(dict_json["foodInitialPosition"])
        self.grabbed = Success(dict_json["grabbed"])
        self.release = Success(dict_json["release"])
        self.eat = Success(dict_json["eat"])
        self.fall = Success(dict_json["fall"])

class XRFishing(XR):
    pass


class Name():

    def __init__(self, name):

        if name == None:
            print("Name data is not existed. Please confirm it.")
            return
        else:
            self.names = name
            # self.names = [name_type for t in name for name_type in Food_type if name_type.value == t]

class Time():

    def __init__(self, time):

        if time == None:
            print("Time data is not existed. Please confirm it.")
            return
        else:
            self.times = time

class Position():

    def __init__(self, data):

        if data == None:
            print("Position data is not existed. Please confirm it.")
            return
        else:
            self.positions = [Type.Vector3(x=p["x"], y=p["y"], z=p["z"]) for p in data]

class Number():

    def __init__(self, data):

        if data == None:
            print("Number data is not existed. Please confirm it.")
            return
        else:
            self.numbers = data

class Success():

    def __init__(self, list):

        if list == None:
            print("it is not success. Please confirm it")
        else:
            self.success = list

class Food_type(Enum):

    sliching = "Slicing"
    meat = "Meat"
    apple = "Apple"
    carrot = "Carrot"
    steak = "Steak"
    geta = "Geta"

parent_path = os.getcwd()
VR_game_path = parent_path + "\\VR_game"
magic_path = VR_game_path + "\\FishingGame"

event_path = magic_path + "\\LevelDataEvent.json"
frame_path = magic_path + "\\LevelDataFrame.json"
xr_path = magic_path + "\\LevelDataXR.json"

info = {'path_event': event_path, 'path_frame': frame_path, 'path_XR': xr_path}

fishing_data = Load_Fishing(info)