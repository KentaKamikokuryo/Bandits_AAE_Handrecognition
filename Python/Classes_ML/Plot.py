import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.animation as animation
import copy
from collections import OrderedDict

class Plot_ML():

    @staticmethod
    def plot_latent_space(z_train, y_train, z_test=None, y_test=None, gradient_train=None, gradient_test=None, fig=None, ax=None, figure_title: str = ""):

        """Plot result of dimensionality reduction. This code can use both of the cases that you have only data that is dimensionally reduced and you have train and test data that is dimensionally reduced
           If you input both of train and test, only test point will be edged

        Parameters
        ----------
        z_train : ndarray(float)
            2D array of train data that is dimensionally reduced dim(num_train_data, n_components)
        y_train : list(str)
            list of train labels
        z_test : ndarray(float)
            2D array of test data that is dimensionally reduced dim(num_test_data, n_components)
        y_test : list(str)
            list of test labels
        fig: matplotlib fig
            will create a new figure if None
        ax: matplotlib ax
            will create a new ax if None

        Returns
        -------
        fig : figure object
            the figure that axes object belongs to
        ax : axes object
            the axes that has been plotted
        """

        Plot_ML.set_rcParams()

        if not fig and not ax:
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(1, 1, 1)

        unique_class = np.unique(y_train)  # make unique_class ascending order automatically
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(unique_class)))
        colors_thesis = Plot_ML.make_colors(name_list=['tab:blue', 'tab:red', 'tab:pink', 'tab:purple', 'tab:olive', 'tab:green'])
        colors = dict(zip(unique_class, colors_thesis))
        # colors = dict(zip(unique_class, colors))
        gradient_cmap = plt.cm.coolwarm
        gradient_cmaps_thesis = [plt.cm.Blues, plt.cm.Reds, plt.cm.RdPu, plt.cm.Purples, plt.cm.YlOrBr, plt.cm.Greens]
        gradient_cmaps_thesis = dict(zip(unique_class, gradient_cmaps_thesis))

        for unique in unique_class:

            indices = [i for i, x in enumerate(y_train) if x == unique]

            X = z_train[:, 0][indices]
            Y = z_train[:, 1][indices]

            if (gradient_train is not None):
                c = np.array(gradient_train)[indices]
                ax.scatter(X, Y, c=c, s=20, marker='o', ec='none', cmap=gradient_cmaps_thesis[unique])
                ax.scatter([], [], c=Plot_ML.get_color_from_cmap(gradient_cmaps_thesis[unique], 0.7), label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none')
                # ax.scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none', cmap=gradient_cmaps_thesis[unique])
                # ax.scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique) + '_train'), s=20, marker='o', ec='none', cmap=gradient_cmap)
                # ax.scatter(X, Y, c=c, label=str(unique) + '_train', s=20, marker='o', ec='none', cmap=plt.cm.coolwarm)
            else:
                ax.scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none')
                # ax.scatter(X, Y, color=colors[unique], label=str(unique) + '_train', s=20, marker='o', ec='none')

            if (z_test is not None) and (y_test is not None):

                indices = [i for i, x in enumerate(y_test) if x == unique]

                X = z_test[:, 0][indices]
                Y = z_test[:, 1][indices]

                if (gradient_test is not None):
                    c = np.array(gradient_test)[indices]
                    ax.scatter(X, Y, c=c, s=50, marker='o', ec='k', cmap=gradient_cmaps_thesis[unique])
                    ax.scatter([], [], c=Plot_ML.get_color_from_cmap(gradient_cmaps_thesis[unique], 0.7), label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k')
                    # ax.scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', cmap=gradient_cmaps_thesis[unique])
                    # ax.scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', cmap=gradient_cmap)
                    # ax.scatter(X, Y, c=c, label=str(unique) + '_test', s=50, marker='o', ec='k', cmap=plt.cm.coolwarm)
                else:
                    ax.scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', lw=0.5, cmap=gradient_cmaps_thesis[unique])
                    # ax.scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', lw=0.5, cmap=gradient_cmap)
                    # ax.scatter(X, Y, color=colors[unique], label=str(unique) + '_test', s=50, marker='o', ec='k', lw=0.5, cmap=plt.cm.coolwarm)

        ax.set_title(Plot_ML.remove_underbar(figure_title))
        # ax.set_title(figure_title)

        Plot_ML.set_margin(ax=ax, top=0.05, bottom=0.05, left=0.05, right=0.7)
        # Plot_ML.set_margin(ax=ax, top=0.25, bottom=0.25, left=0.25, right=0.25)
        # ax.set_xmargin(0.75)

        ax.legend(labelspacing=0.5)
        ax.set_xlabel(Plot_ML.remove_underbar('1st_component'))
        ax.set_ylabel(Plot_ML.remove_underbar('2nd_component'))
        # ax.set_xlabel('1st_component')
        # ax.set_ylabel('2nd_component')
        ax.set_aspect('equal')

        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        legend = ax.legend(by_label.values(), by_label.keys(), markerscale=1, title="Class", bbox_to_anchor=(0, 0, 1, 1))
        # legend = ax.legend(by_label.values(), by_label.keys(), markerscale=1, title="Class", bbox_to_anchor=(0, 0, 1, 1), fancybox=True, shadow=True)
        plt.setp(legend.get_title(), fontsize=20)
        # fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.05, hspace=0.1)

        return fig, ax

    @staticmethod
    def plot_latent_space_subplot(z_train, y_train, z_test=None, y_test=None, gradient_train=None, gradient_test=None, figure_title:str = ""):

        """Plot result of dimensionality reduction so that each class be subplot.
           This code can use both of the cases that you have only data that is dimensionally reduced and you have train and test data that is dimensionally reduced
           If you input both of train and test, only test point will be edged

        Parameters
        ----------
        z_train : ndarray(float)
            2D array of train data that is dimensionally reduced dim(num_train_data, n_components)
        y_train : list(str)
            list of train labels
        z_test : ndarray(float)
            2D array of test data that is dimensionally reduced dim(num_test_data, n_components)
        y_test : list(str)
            list of test labels

        # TODO: add doc for gradient_train and gradient_test

        Returns
        -------
        fig : figure object
            the figure that axes object belongs to
        ax : dictionary of axes object
            the axes that has been plotted
        """

        Plot_ML.set_rcParams()

        unique_class = np.unique(y_train)  # make unique_class ascending order automatically
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(unique_class)))
        colors_thesis = Plot_ML.make_colors(name_list=['tab:blue', 'tab:red', 'tab:pink', 'tab:purple', 'tab:olive', 'tab:green'])
        colors = dict(zip(unique_class, colors_thesis))
        # colors = dict(zip(unique_class, colors))
        gradient_cmap = plt.cm.coolwarm
        gradient_cmaps_thesis = [plt.cm.Blues, plt.cm.Reds, plt.cm.RdPu, plt.cm.Purples, plt.cm.YlOrBr, plt.cm.Greens]
        gradient_cmaps_thesis = dict(zip(unique_class, gradient_cmaps_thesis))

        scatters_dict = dict()

        fig = plt.figure(figsize=(20, 10))

        if len(unique_class) < 3:
            axes = {y: fig.add_subplot(1, len(unique_class), i + 1) for i, y in enumerate(unique_class)}
        else:
            axes = {y: fig.add_subplot(2, 3, i + 1) for i, y in enumerate(unique_class)}

        if (z_test is not None) and (y_test is not None):

            for i, unique in enumerate(unique_class):

                indices = [i for i, x in enumerate(y_train) if x == unique]

                X = z_train[:, 0][indices]
                Y = z_train[:, 1][indices]

                if (gradient_train is not None):
                    c = np.array(gradient_train)[indices]
                    axes[unique].scatter(X, Y, c=c, s=20, marker='o', ec='none', cmap=gradient_cmaps_thesis[unique])
                    axes[unique].scatter([], [], c=Plot_ML.get_color_from_cmap(gradient_cmaps_thesis[unique], 0.7), label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none')
                    # axes[unique].scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none', cmap=gradient_cmaps_thesis[unique])
                    # axes[unique].scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none', cmap=gradient_cmap)
                    # axes[unique].scatter(X, Y, c=c, label=str(unique) + '_train', s=20, marker='o', ec='none', cmap=plt.cm.coolwarm)
                else:
                    axes[unique].scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique)+'_train'), s=20, marker='o', ec='none')
                    # axes[unique].scatter(X, Y, color=colors[unique], label=str(unique) + '_train', s=20, marker='o', ec='none')

                if (z_test is not None) and (y_test is not None):

                    indices = [i for i, x in enumerate(y_test) if x == unique]

                    X = z_test[:, 0][indices]
                    Y = z_test[:, 1][indices]

                    if (gradient_test is not None):
                        c = np.array(gradient_test)[indices]
                        axes[unique].scatter(X, Y, c=c, s=50, marker='o', ec='k', cmap=gradient_cmaps_thesis[unique])
                        axes[unique].scatter([], [], c=Plot_ML.get_color_from_cmap(gradient_cmaps_thesis[unique], 0.7), label=Plot_ML.remove_underbar(str(unique) + '_test'), s=50, marker='o', ec='k')
                        # axes[unique].scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', cmap=gradient_cmaps_thesis[unique])
                        # axes[unique].scatter(X, Y, c=c, label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', cmap=gradient_cmap)
                        # axes[unique].scatter(X, Y, c=c, label=str(unique) + '_test', s=50, marker='o', ec='k', cmap=plt.cm.coolwarm)
                    else:
                        axes[unique].scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique) + '_test'), s=50, marker='o', ec='k', lw=0.5, cmap=gradient_cmaps_thesis[unique])
                        # axes[unique].scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique)+'_test'), s=50, marker='o', ec='k', lw=0.5, cmap=gradient_cmap)
                        # axes[unique].scatter(X, Y, color=colors[unique], label=str(unique) + '_test', s=50, marker='o', ec='k', lw=0.5, cmap=plt.cm.coolwarm)

                axes[unique].set_title(str(unique))

                axes[unique].set_title("label: " + str(unique))
                axes[unique].legend()
                axes[unique].set_xlabel(Plot_ML.remove_underbar('1st_component'))
                axes[unique].set_ylabel(Plot_ML.remove_underbar('2nd_component'))
                # axes[unique].set_xlabel('1st_component')
                # axes[unique].set_ylabel('2nd_component')
                axes[unique].set_aspect('equal')

        fig.suptitle(Plot_ML.remove_underbar(figure_title))
        # fig.suptitle(figure_title)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.90, wspace=0.2, hspace=0.1)

        return fig, axes

    @staticmethod
    def plot_trajectories(z_test_dict, fig=None, ax=None):

        colors_trajectory_thesis = Plot_ML.make_colors(name_list=['cyan', 'orangered', 'hotpink', 'mediumorchid', 'yellow', 'lime'])
        colors = dict(zip(z_test_dict.keys(), colors_trajectory_thesis))

        if not fig and not ax:
            return
        else:

            for i, unique in enumerate(z_test_dict.keys()):

                z = z_test_dict[unique]

                X = z[:, 0]
                Y = z[:, 1]

                ax.plot(X, Y, linewidth=3, color=colors[unique], label=Plot_ML.remove_underbar(str(unique) + '_trajectory'))
                # ax.plot(X, Y, linewidth=3, color="cyan", label=Plot_ML.remove_underbar(str(unique)+'_trajectory'))
                # ax.plot(X, Y, linewidth=3, color="cyan")
                ax.legend()
                ax.scatter(X, Y, s=12, color=colors[unique])
                # ax.scatter(X, Y, s=10, color="cyan")

    @staticmethod
    def plot_trajectories_subplot(z_test_dict, fig=None, axes=None):

        colors_trajectory_thesis = Plot_ML.make_colors(name_list=['cyan', 'orangered', 'hotpink', 'mediumorchid', 'yellow', 'lime'])
        colors = dict(zip(z_test_dict.keys(), colors_trajectory_thesis))

        if not fig and not axes:
            return

        else:

            for i, unique in enumerate(z_test_dict.keys()):

                z = z_test_dict[unique]

                X = z[:, 0]
                Y = z[:, 1]

                axes[unique].plot(X, Y, linewidth=3, color=colors[unique], label=Plot_ML.remove_underbar(str(unique) + '_trajectory'))
                # axes[unique].plot(X, Y, linewidth=3, color="cyan", label=Plot_ML.remove_underbar(str(unique)+'_trajectory'))
                # axes[unique].plot(X, Y, linewidth=3, color="cyan")
                axes[unique].legend()
                axes[unique].scatter(X, Y, s=10, color=colors[unique])
                # axes[unique].scatter(X, Y, s=10, color="cyan")

    @staticmethod
    def plot_centroids(c, fig=None, ax=None):

        """Plot centroids

        Parameters
        ----------
        c : dict
            dictionary of centroids of each train cluster. Key is consider as the label. Each key contain a numpy array with 2 values
        fig: matplotlib fig
            will create a new figure if None
        ax: matplotlib ax
            will create a new ax if None

        Returns
        -------
        None
        """

        if not fig and not ax:
            return
        else:
            cmap = plt.get_cmap('tab10')
            unique_class = np.unique(list(c.keys()))
            colors = cmap(np.linspace(0, 1, len(unique_class)))
            colors_centroid_thesis = Plot_ML.make_colors(name_list=['lightblue', 'lightsalmon', 'lightpink', 'plum', 'khaki', 'lightgreen'])
            colors = dict(zip(unique_class, colors_centroid_thesis))
            # colors = dict(zip(unique_class, colors))

            for i, unique in enumerate(unique_class):

                X = c[unique][0]
                Y = c[unique][1]

                ax.scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique)+'_centroid'), s=300, linewidth=2, marker="p", edgecolor=(0., 0., 0., 0.5))
                # ax.scatter(X, Y, color=colors[unique], s=200, linewidth=1, marker="p", edgecolor=(0., 0., 0., 0.5))
                ax.legend()

    @staticmethod
    def plot_centroids_subplot(c, fig=None, axes=None):

        if not fig and not axes:
            return
        else:
            cmap = plt.get_cmap('tab10')
            unique_class = np.unique(list(c.keys()))
            colors = cmap(np.linspace(0, 1, len(unique_class)))
            colors_centroid_thesis = Plot_ML.make_colors(name_list=['lightblue', 'lightsalmon', 'lightpink', 'plum', 'khaki', 'lightgreen'])
            colors = dict(zip(unique_class, colors_centroid_thesis))
            # colors = dict(zip(unique_class, colors))

            for i, unique in enumerate(unique_class):

                X = c[unique][0]
                Y = c[unique][1]

                axes[unique].scatter(X, Y, color=colors[unique], label=Plot_ML.remove_underbar(str(unique) + '_centroid'), s=300, linewidth=2, marker="p", edgecolor=(0., 0., 0., 0.5))
                # axes[unique].scatter(X, Y, color=colors[unique], s=200, linewidth=1, marker="p", edgecolor=(0., 0., 0., 0.5))
                axes[unique].legend()

    @staticmethod
    def plot_confidence_ellipse(ellipse_dict, fig=None, ax=None):  # CI : Confidence Interval

        """Plot confidence ellipse

        Parameters
        ----------
        ellipse_dict : dict
            dictionary of matplotlib object representing the ellipse of each cluster. Key is consider as the label.
        fig: matplotlib fig
            will create a new figure if None
        ax: matplotlib ax
            will create a new ax if None

        Returns
        -------
        None
        """

        ellipse_dict = copy.deepcopy(ellipse_dict)  # copy so we can reuse it

        if not fig and not ax:
            return
        else:

            for y in ellipse_dict.keys():
                ax.add_patch(ellipse_dict[y])

    @staticmethod
    def plot_confidence_ellipse_subplot(ellipse_dict, fig=None, axes=None):

        """Plot confidence ellipse

        Parameters
        ----------
        ellipse_dict : dict
            dictionary of matplotlib object representing the ellipse of each cluster. Key is consider as the label.
        fig: matplotlib fig
            will create a new figure if None
        axes : dict of axes object
            dictonary of axes object that has latent space that is subplot. Must contain the same keys as ellipse_dict

        Returns
        -------
        None
        """

        ellipse_dict = copy.deepcopy(ellipse_dict)  # copy so we can reuse it

        if not fig and not axes:
            return
        else:
            for y in ellipse_dict.keys():
                axes[y].add_patch(ellipse_dict[y])
                axes[y].legend()

    @staticmethod
    def savefig(path_figure, figure_name, fig):

        fig.tight_layout()

        fig.savefig(path_figure + figure_name + ".png", dpi=200)
        print("Figure saved to: " + path_figure + figure_name + ".png")

        plt.close(fig)

    @staticmethod
    def set_margin(ax, top=0.0, bottom=0.0, left=0.0, right=0.0):

        x_min = ax.get_xlim()[0]
        x_max = ax.get_xlim()[1]
        y_min = ax.get_ylim()[0]
        y_max = ax.get_ylim()[1]

        x_lange = x_max - x_min
        y_lange = y_max - y_min

        X_min = x_min - x_lange * left
        X_max = x_max + x_lange * right
        Y_min = y_min - y_lange * bottom
        Y_max = y_max + y_lange * top

        ax.set_xlim(X_min, X_max)
        ax.set_ylim(Y_min, Y_max)

        return ax

    @staticmethod
    def make_colors(name_list: list):

        colors = []

        base_name_list = list(mcolors.BASE_COLORS)
        tab_name_list = list(mcolors.TABLEAU_COLORS)
        css_name_list = list(mcolors.CSS4_COLORS)

        for name in name_list:

            if name in base_name_list:
                colors.append(mcolors.BASE_COLORS[name])

            elif name in tab_name_list:
                colors.append(mcolors.TABLEAU_COLORS[name])

            elif name in css_name_list:
                colors.append(mcolors.CSS4_COLORS[name])

        return colors

    @staticmethod
    def get_color_from_cmap(cmap: matplotlib.colors.Colormap, rate: float=1.0):

        if (rate < 0) or (rate > 1.0):
            rate = 1.0

        i = int(255 * rate)

        RGBA = [cmap(i)]

        return RGBA

    @staticmethod
    def remove_underbar(string: str):

        if "_" in string:
            removed_string = string.replace("_", " ")

        else:
            removed_string = string

        return removed_string

    @staticmethod
    def set_rcParams(font_family="Times New Roman", font_size=18, math_font="stix",
                     figure_width=6.4, figure_height=4.8, dpi=100, top=0.9, bottom=0.1, left=0.025, right=0.975, width_space=0.8, height_space=0.5,
                     frame_width=1.0, title_pad=10.0, label_pad=4.0, label_size='large', grid=False,
                     tick_in=True, minor_tick=False, major_size=5.0, major_width=1.0, major_pad=3.5,
                     line_width=1.5, marker_size=6.0,
                     legend_edge_color='black', legend_face_alpha=0.7, legend_font_size='small', legend_marker_size=1.0, legend_column_space=2.0, legend_shadow=False, fancy_box=False):

        plt.rcParams['font.family'] = font_family  # "sans-serif"
        plt.rcParams['font.size'] = font_size
        plt.rcParams['mathtext.fontset'] = math_font

        plt.rcParams['figure.figsize'] = [figure_width, figure_height]  # [6.4, 4.8]
        plt.rcParams['figure.dpi'] = dpi  # 100
        plt.rcParams['figure.subplot.top'] = top  # 0.88
        plt.rcParams['figure.subplot.bottom'] = bottom  # 0.11
        plt.rcParams['figure.subplot.left'] = left  # 0.125
        plt.rcParams['figure.subplot.right'] = right  # 0.9
        plt.rcParams['figure.subplot.wspace'] = width_space  # 0.2
        plt.rcParams['figure.subplot.hspace'] = height_space  # 0.2

        plt.rcParams['axes.linewidth'] = frame_width  # 1.0
        plt.rcParams['axes.titlepad'] = title_pad  # 6.0
        plt.rcParams['axes.labelpad'] = label_pad  # 4.0
        plt.rcParams['axes.labelsize'] = label_size  # 'medium'
        plt.rcParams['axes.grid'] = grid

        if tick_in:
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
        else:
            plt.rcParams['xtick.direction'] = 'out'
            plt.rcParams['ytick.direction'] = 'out'

        plt.rcParams['xtick.minor.visible'] = minor_tick
        plt.rcParams['ytick.minor.visible'] = minor_tick
        plt.rcParams['xtick.major.size'] = major_size  # 3.5
        plt.rcParams['ytick.major.size'] = major_size  # 3.5
        plt.rcParams['xtick.major.width'] = major_width  # 0.8
        plt.rcParams['ytick.major.width'] = major_width  # 0.8
        plt.rcParams['xtick.major.pad'] = major_pad  # 3.5
        plt.rcParams['ytick.major.pad'] = major_pad  # 3.5

        plt.rcParams['lines.linewidth'] = line_width  # 1.5
        plt.rcParams['lines.markersize'] = marker_size  # 6.0

        plt.rcParams['legend.edgecolor'] = legend_edge_color  # 'black'
        plt.rcParams['legend.framealpha'] = legend_face_alpha  # 0.8
        plt.rcParams['legend.fontsize'] = legend_font_size  # 'medium'
        plt.rcParams['legend.markerscale'] = legend_marker_size  # 1.0
        plt.rcParams['legend.columnspacing'] = legend_column_space  # 2.0
        plt.rcParams['legend.shadow'] = legend_shadow
        plt.rcParams['legend.fancybox'] = fancy_box