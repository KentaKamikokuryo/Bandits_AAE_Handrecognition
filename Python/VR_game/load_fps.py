import os

from Classes_game.Load import Load, All, Event, Frame, XR
from Classes_game import Type
import Classes_game.Load as load

class Load_FPS(Load):

    def load_all(self, jsn_event, jsn_frame, jsn_xr):

        self.all_data = AllFPS(json_event=jsn_event, json_frame=jsn_frame, json_xr=jsn_xr)

class AllFPS(All):

    def __init__(self, json_event, json_frame, json_xr):

        self.get_info_player(json=json_event)
        self.event = EventFPS(dict_json=json_event)
        self.frame = FrameFPS(dict_json=json_frame)
        self.xr = XRFPS(dict_json=json_xr)

    def get_info_player(self, json: dict):
        self.game_name = json["gameName"]
        self.session_id = json["sessionID"]
        self.score = json["score"]
        self.score_time = json["scoreTime"]
        self.data_time_txt = json["dateTimeTxt"]

class EventFPS(Event):

    def __init__(self, dict_json: dict):

        self.enemy = Name(dict_json["enemy"])
        self.appear = Time(dict_json["timeAppear"])
        self.disappear = Time(dict_json["timeDisappear"])
        self.initial = Position(dict_json["initialPositions"])
        self.dies = Position(dict_json["diedPositions"])
        self.gun = Name(dict_json["gun"])
        self.num_enemy = Number(dict_json["numberOfEnemy"])

class FrameFPS(Frame):

    def __init__(self, dict_json: dict):

        self.enemy = Name(dict_json["enemy"])
        self.pos_enemy = Position(dict_json["enemyPosition"])
        self.gun = Name(dict_json["gun"])
        self.reload_right = Success(dict_json["reloadRight"])
        self.reload_left = Success(dict_json["reloadLeft"])
        self.shoot_right = Success(dict_json["shootRight"])
        self.shoot_left = Success(dict_json["shootLeft"])
        self.change_gun = Success(dict_json["changeGun"])

class XRFPS(XR):
    pass


class Name():

    def __init__(self, name):

        if name == None:
            print("Name data is not existed. Please confirm it.")
            return
        else:
            self.names = name

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

parent_path = os.getcwd()
VR_game_path = parent_path + "\\VR_game"
magic_path = VR_game_path + "\\FPSGame"

event_path = magic_path + "\\LevelDataEvent.json"
frame_path = magic_path + "\\LevelDataFrame.json"
xr_path = magic_path + "\\LevelDataXR.json"

info = {'path_event': event_path, 'path_frame': frame_path, 'path_XR': xr_path}

fps_data = Load_FPS(info)