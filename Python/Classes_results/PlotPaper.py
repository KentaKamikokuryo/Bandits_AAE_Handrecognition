from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from Classes_RL.Bandit import Bandit
from Classes_RL.Plot_RL import PlotSolver, PlotResults
from Classes_data.XML import ImageXML, SessionXML, XML

class PlotPaper:

    @staticmethod
    def plot_reference_image(reference_image: dict, path="", figure_name=""):

        # Distortion factor for one label (X, Y and Rotation)
        fig = plt.figure(figsize=(12, 3))
        gs = fig.add_gridspec(1, 6)

        # Reference image
        for i in range(len(reference_image.keys())):

            ax = fig.add_subplot(gs[0, i])
            ax.set_xticks([])
            ax.set_xticks([], minor=True)

            ax.set_yticks([])
            ax.set_yticks([], minor=True)

            label = list(reference_image.keys())[i]
            label = label.split(sep="_")[0]

            ax.set_title(label)

            array = reference_image[list(reference_image.keys())[i]]
            data = np.flipud(np.transpose(array))

            plt.imshow(data, cmap='gray')

        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.2)
        PlotPaper.save_plot(fig=fig, path=path, figure_name=figure_name, close_figure=True)

    @staticmethod
    def plot_augmented_image(augmented_image: dict, path="", figure_name=""):

        n_session = len(augmented_image.keys())
        n_image_in_session = len(augmented_image[0].keys())

        fig = plt.figure(figsize=(12, 4))
        # gs = fig.add_gridspec(n_image_in_session, n_session // 2)
        gs = fig.add_gridspec(2, n_session // 2)

        count = 0
        row = 0

        for i in range(0, n_session):

            if i == 10:
                row = 1
                count = 0

            session_scaled_new_dict, session_image_new_dict = XML.convert(augmented_image[i])

            ax = fig.add_subplot(gs[row, count])
            ax.set_xticks([])
            ax.set_xticks([], minor=True)

            ax.set_yticks([])
            ax.set_yticks([], minor=True)

            array = session_image_new_dict[0]
            data = np.flipud(np.transpose(array))

            plt.imshow(data, cmap='gray')

            count += 1

        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

        PlotPaper.save_plot(fig=fig, path=path, figure_name=figure_name, close_figure=True)

    @staticmethod
    def plot_behavior(behavior_all: dict, path="", figure_name=""):

        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 3)

        fontsize = 12

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(behavior_all["ratio_X_change"][::3], linewidth=2)
        ax.set_xlabel("Session", fontsize=fontsize)
        ax.set_ylabel("Ratio X", fontsize=fontsize)
        ax.set_xlim((0, 20.0))

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(behavior_all["ratio_Y_change"][::3], linewidth=2)
        ax.set_xlabel("Session", fontsize=fontsize)
        ax.set_ylabel("Ratio Y", fontsize=fontsize)
        ax.set_xlim((0, 20.0))

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(behavior_all["rotation_change"][::3] / np.pi * 360, linewidth=2)
        ax.set_xlabel("Session", fontsize=fontsize)
        ax.set_ylabel("Rotation (rad)", fontsize=fontsize)
        ax.set_xlim((0, 20.0))

        plt.subplots_adjust(left=0.08, bottom=0.14, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
        PlotPaper.save_plot(fig=fig, path=path, figure_name=figure_name, close_figure=True)

    @staticmethod
    def plot_latent_space_gradient_trajectories():

        pass

    @staticmethod
    def plot_bandits_results():

        pass

    @staticmethod
    def save_plot(fig, path, figure_name: str, close_figure: bool = True):

        fig.savefig(path + figure_name + ".png")
        print("Figure saved to: " + path + figure_name + ".png")

        if close_figure:
            plt.close(fig)
