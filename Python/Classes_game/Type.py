class Vector3():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Quaternion():

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Vector2():

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Controller_Positions():

    def __init__(self, positions):
        self.positions = [Vector3(position['x'], position['y'], position['z'])
                          for position in positions]


class Controller_Quaternions():

    def __init__(self, quaternions):
        self.quaternions = [Quaternion(quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w'])
                            for quaternion in quaternions]


class Button_Press():

    def __init__(self, press):
        self.press = press


class Press_Values():

    def __init__(self, values):
        self.values = values


class Press_Axes():

    def __init__(self, axes):
        self.axes = [Vector2(axis['x'], axis['y']) for axis in axes]