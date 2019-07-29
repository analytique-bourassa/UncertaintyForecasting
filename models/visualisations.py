import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

DEFAULT_TICKS_SIZE = 19
DEFAULT_LABEL_SIZE = 22
DEFAULT_LINEWIDTH = 3.0
DEFAULT_TITLE_SIZE = 26
DEFAULT_LEGEND_SIZE = 19

from matplotlib import rcParams
rcParams['axes.titlepad'] = 20

class Visualisator():

    @staticmethod
    def show_time_series(data, title):

        plt.title(title, size=DEFAULT_TITLE_SIZE)
        plt.plot(data, linewidth=DEFAULT_LINEWIDTH)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.ylabel("y", size=DEFAULT_LABEL_SIZE, rotation=0)
        plt.xlabel("Time", size=DEFAULT_LABEL_SIZE)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_distribution(data, title,name, normal_fit=True):

        if normal_fit:
            x = np.linspace(min(data), max(data), 100)
            y = norm.pdf(x, loc=data.mean(), scale=data.std())
            plt.plot(x, y, label="normal fit", linewidth=DEFAULT_LINEWIDTH)

        plt.title(title, size=DEFAULT_TITLE_SIZE)
        plt.ylabel("Density", size=DEFAULT_LABEL_SIZE)
        plt.xlabel(name, size=DEFAULT_LABEL_SIZE)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        sns.distplot(data,
                     norm_hist=True,
                     label="density of %s" % name, kde=False)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_epoch_convergence(data, title, name, number_of_burned_step=200):

        n_epochs = data.shape[0]
        plt.title(title, size=DEFAULT_TITLE_SIZE)
        plt.plot(range(number_of_burned_step, n_epochs), data[number_of_burned_step:])

        plt.ylabel(name, size=DEFAULT_LABEL_SIZE, rotation=0)
        plt.xlabel("epoch", size=DEFAULT_LABEL_SIZE)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.show()

    @staticmethod
    def show_predictions(y_pred, y_true, dataset="training"):

        plt.plot(y_pred, label="predictions", linewidth=DEFAULT_LINEWIDTH)
        plt.plot(y_true, label="true values", linewidth=DEFAULT_LINEWIDTH)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.ylabel("y", size=DEFAULT_LABEL_SIZE, rotation=0)
        plt.xlabel("Time", size=DEFAULT_LABEL_SIZE)

        plt.title("Comparing predictions with true values on %s set" % dataset, size=DEFAULT_TITLE_SIZE)
        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()
