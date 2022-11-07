import numpy as np
import matplotlib.pyplot as plt

class Coordinates_to_image:

    def __init__(self):

        self.resolution = 100
        self.border = 10
        self.range_around = 5

    def convert(self, coordinates, plot_it=False):

        coordinates_scaled = coordinates.copy()

        coordinates_scaled *= self.resolution

        min_x = np.abs(np.min(coordinates_scaled[:, 0]))
        min_y = np.abs(np.min(coordinates_scaled[:, 1]))

        coordinates_scaled[:, 0] += min_x
        coordinates_scaled[:, 1] += min_y

        mean_x = np.mean(coordinates_scaled[:, 0])
        mean_y = np.mean(coordinates_scaled[:, 1])

        coordinates_scaled[:, 0] += (self.resolution / 2 - mean_x)
        coordinates_scaled[:, 1] += (self.resolution / 2 - mean_y)

        # Rescale
        diff_x = np.max(coordinates_scaled[:, 0]) - np.min(coordinates_scaled[:, 0])
        diff_y = np.max(coordinates_scaled[:, 1]) - np.min(coordinates_scaled[:, 1])

        if plot_it:
            plt.figure()
            plt.scatter(coordinates_scaled[:, 0], coordinates_scaled[:, 1], color="r", s=50, marker="+")
            plt.xlim(-10, self.resolution + 10)
            plt.ylim(-10, self.resolution + 10)

        if diff_x > diff_y:
            diff = diff_x
        else:
            diff = diff_y

        min_scale = (1 + self.border)
        max_scale = (100 - self.border)

        min_xy = np.min(coordinates_scaled[:, :])
        max_xy = np.max(coordinates_scaled[:, :])

        a = (max_scale - min_scale) / (max_xy - min_xy)
        b = min_scale - min_xy * a

        coordinates_scaled[:, 0] = a * coordinates_scaled[:, 0] + b
        coordinates_scaled[:, 1] = a * coordinates_scaled[:, 1] + b

        # a = (max_scale - min_scale) / (diff)
        # b = min_scale - min_xy * a
        # coordinates_scaled[:, 0] = a * coordinates_scaled[:, 0] + b
        #
        # a = (max_scale - min_scale) / (diff)
        # b = min_scale - min_xy * a
        # coordinates_scaled[:, 1] = a * coordinates_scaled[:, 1] + b

        coordinates_scaled = coordinates_scaled.astype(dtype=np.int32)

        if plot_it:
            plt.figure()
            plt.scatter(coordinates_scaled[:, 0], coordinates_scaled[:, 1], color="r", s=200, marker="o")
            plt.xlim(0, self.resolution)
            plt.ylim(0, self.resolution)

        image = np.zeros(shape=(self.resolution, self.resolution))

        for vector2D in coordinates_scaled:
            image[vector2D[0], vector2D[1]] = 255
            for i in range(-self.range_around, self.range_around):
                for j in range(-self.range_around, self.range_around):
                    new_x = vector2D[0] + i
                    new_y = vector2D[1] + j
                    if new_x > self.resolution or new_x < 0:
                        new_x = vector2D[0]
                    if new_y > self.resolution or new_y < 0:
                        new_y = vector2D[1]

                    image[new_x, new_y] = 255

        return coordinates_scaled, image
