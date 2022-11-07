import os
from enum import Enum
from Classes_game.Load import Load, All, Event, Frame, XR

class Load_magic(Load):

    def load_all(self, jsn_event, jsn_frame, jsn_xr):

        self.all_data = All_magic(json_event=jsn_event, json_frame=jsn_frame, json_xr=jsn_xr)

class All_magic(All):

    def __init__(self, json_event, json_frame, json_xr):

        self.get_info_player(json=json_event)
        self.event = Event_magic(dict_json=json_event)
        self.frame = Frame_magic(dict_json=json_frame)
        self.xr = XR_magic(dict_json=json_xr)

class Event_magic(Event):

    def __init__(self, dict_json: dict):

        self.end_game_condition = Coindition(dict_json["endGameCondition"])
        self.enemy_type = Magic_types(dict_json["enemyType"])
        self.time_appear = Times(dict_json["timeAppear"])
        self.time_disappear = Times(dict_json["timeDisappear"])
        self.player_type = Magic_types(dict_json["attackPlayer"])
        self.defense_type = Magic_types(dict_json["defensePlayer"])
        self.number_of_attack = Numbers(dict_json["numberOfAttack"])
        self.time_of_attack = Times(dict_json["timeOfAttack"])
        self.time_of_defence = Times(dict_json["timeOfDefence"])

class Frame_magic(Frame):

    def __init__(self, dict_json: dict):

        self.enemy_type = Magic_types(dict_json['enemyType'])
        self.player_type = Magic_types(dict_json['attackPlayer'])
        self.defense_player = Magic_types(dict_json['defensePlayer'])
        self.player_attacking = Attackings(dict_json['playerAttacking'])
        self.enemy_attacking = Attackings(dict_json['enemyAttacking'])

class XR_magic(XR):
    pass


class Magic_type(Enum):

    fire = "Fire"
    water = "Water"
    wind = "Wind"
    none = "None"

class Magic_types():

    def __init__(self, types):
        self.types = [magic_type
                      for type in types
                      for magic_type in Magic_type
                      if magic_type.value == type]

class Condition_enum(Enum):

    win = "Win"
    died = "Died"
    quit = "Quit"

class Coindition():

    def __init__(self, cond):

        for condition_enum in Condition_enum:
            if condition_enum.value == cond:
                self.condision = condition_enum

class Times():

    def __init__(self, times):
        self.times = times

class Numbers():

    def __init__(self, numbers):
        self.numbers = numbers

class Attackings():

    def __init__(self, attackings):
        self.attackings = attackings


parent_path = os.getcwd()
VR_game_path = parent_path + "\\VR_game"
magic_path = VR_game_path + "\\MagicGame"

event_path = magic_path + "\\LevelDataEvent.json"
frame_path = magic_path + "\\LevelDataFrame.json"
xr_path = magic_path + "\\LevelDataXR.json"

info = {'path_event': event_path, 'path_frame': frame_path, 'path_XR': xr_path}

load_magic = Load_magic(info)
magic_data = load_magic.get_data()
