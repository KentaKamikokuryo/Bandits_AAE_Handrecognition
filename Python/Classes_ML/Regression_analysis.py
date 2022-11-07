import numpy as np
import os, math
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from sklearn import metrics
import scipy as sp
from math import *
from sklearn import metrics
from math import sqrt

class Regression_analysis():

    def __init__(self, x, y, deg=1, model_name=""):

        self.x = x
        self.y = y

        self.deg = deg

        self.model_name = model_name
        self.is_analysis_done = False
        self.is_analysis_can_be_done = False

        if deg in [1, 2, 3]:
            self.is_analysis_can_be_done = True
        else:
            print("Error in settings the order of the analysis")
            self.is_analysis_can_be_done = False

    def analysis(self):

        if self.is_analysis_can_be_done:

            metric_rmse = math.sqrt(metrics.mean_squared_error(self.x, self.y))

            if self.deg == 1:  # Linear regression

                # Linear regression
                b, m, slope, intercept, r_value, r_squared = self.linear_regression(x=self.x, y=self.y)

                self.analysis_results = {'rmse': metric_rmse,
                                         'b': b,
                                         'm': m,
                                         'slope': slope,
                                         'intercept': intercept,
                                         'r_value': r_value,
                                         'r_squared': r_squared,
                                         'x': self.x,
                                         'y': self.y}

            elif self.deg == 2:  # Second order polyfit

                r_squared, model = self.polyfit(x=self.x, y=self.y, deg=self.deg)

                self.analysis_results = {'rmse': metric_rmse,
                                         'r_squared': np.round(r_squared, decimals=4),
                                         'x': self.x,
                                         'y': self.y}

            elif self.deg == 3:  # Third order polyfit

                r_squared, model = self.polyfit(x=self.x, y=self.y, deg=self.deg)

                self.analysis_results = {'rmse': metric_rmse,
                                         'r_squared': np.round(r_squared, decimals=4),
                                         'x': self.x,
                                         'y': self.y}

            self.is_analysis_done = True

            return self.analysis_results

        else:

            return None

    def plot_analysis(self, fig=None, axes=None, figure_suffix="default", path_figure=None, save_figure=False):

        if self.is_analysis_done:

            name_figure = str(self.model_name) + "_" + figure_suffix

            self.bland_altman_plot(x=self.x, y=self.y, deg=self.deg, fig=fig, axes=axes, title=name_figure, name_figure=name_figure,
                                   save_figure=save_figure, path_save=path_figure)

        else:

            print("No analysis done. Please run analysis() first")

    @staticmethod
    def linear_regression(x, y):

        # x: true values
        # y: predicted values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
        r_squared = r_value ** 2

        slope = np.round(slope, decimals=4)
        intercept = np.round(intercept, decimals=4)
        r_value = np.round(r_value, decimals=4)
        r_squared = np.round(r_squared, decimals=4)

        # Fit with polyfit
        b, m = polyfit(x, y, 1)

        return b, m, slope, intercept, r_value, r_squared

    @staticmethod
    def polyfit(x, y, deg):

        # Polyfit and predict
        model = np.poly1d(np.polyfit(x, y, deg))
        r_squared = metrics.r2_score(y, model(x))

        return r_squared, model

    @staticmethod
    def bland_altman_plot(x, y, deg=1, fig=None, axes=None, title="", name_figure="", save_figure=False, path_save=None):

        # x: true values
        # y: predicted values

        if axes is not None and axes is not None:
            pass
        else:
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))

        mean = np.mean([x, y], axis=0)
        diff = x - y  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        axes[0].scatter(x, y, color='b', label='')

        # Linear regression
        if deg == 1:
            b, m, slope, intercept, r_value, r_squared = Regression_analysis.linear_regression(x, y)
            axes[0].plot(x, b + m * x, '-', color='k', linewidth=3)  # regression lines
            props = dict(boxstyle='round', alpha=0.5,
                         facecolor='none', edgecolor='blue')
            axes[0].text(0.05, 0.95, '$r$=' + str(r_value), fontsize=12, bbox=props, transform=axes[0].transAxes)
            axes[0].text(0.05, 0.9, '$R^2$=' + str(r_squared), fontsize=12, bbox=props, transform=axes[0].transAxes)
            axes[0].text(0.05, 0.85, '$Slope$=' + str(slope), fontsize=12, bbox=props, transform=axes[0].transAxes)
            axes[0].text(0.05, 0.8, '$Intercept$=' + str(intercept), fontsize=12, bbox=props, transform=axes[0].transAxes)

        elif deg == 2 or deg == 3:

            r_squared, model = Regression_analysis.polyfit(x=x, y=y, deg=deg)
            x_pred = [i for i in np.linspace(np.min(x), np.max(x), 50)]
            y_pred = [model(i) for i in x_pred]
            axes[0].plot(x_pred, y_pred, label="Polyfit_" + str(deg))

            props = dict(boxstyle='round', alpha=0.5,
                         facecolor='none', edgecolor='blue')
            axes[0].text(0.05, 0.95, 'Polyfit_' + str(deg) + '$ - R^2$=' + str(r_squared), fontsize=12, bbox=props, transform=axes[0].transAxes)

        # Bland-altman plot
        axes[1].scatter(mean, diff, color='b', label="")
        axes[1].axhline(md, linewidth=5, color='k', linestyle='--')
        axes[1].axhline(md + 1.96 * sd, color='k', linestyle='--')
        axes[1].axhline(md - 1.96 * sd, color='k', linestyle='--')
        axes[1].set_ylim([md + 3 * sd, md - 3 * sd])

        axes[0].grid()
        axes[0].set_xlabel('True values', fontsize=15)
        axes[0].set_ylabel('Predicted values', fontsize=15)

        axes[1].grid()
        axes[1].set_xlabel("Mean", fontsize=15)
        axes[1].set_ylabel("Differences", fontsize=15)

        fig.suptitle(title, fontsize=15)

        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

        if save_figure and path_save is not None:

            if path_save == "":
                print("No path defined")
            else:
                fig_save_path = path_save + name_figure + ".png"
                fig.savefig(fname=fig_save_path, format='png', dpi=400)
                plt.close(fig)
