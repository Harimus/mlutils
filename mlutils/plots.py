import matplotlib.pyplot as plt


def plot_mean_std(means, stds, figtitle="", filename=""):
    """
    Plot the mean and standard deviation of a set of values.
    Inputs:
    :param means: List or array of mean values.
    :param stds: List or array of standard deviation values.
    :param figtitle: Title of the figure.
    :param filename: If provided, saves the figure to this file. If empty, return figure.
    :return: If filename is empty, returns the figure and axis objects.
    """
    fig, ax = plt.subplots()
    ax.plot(means)
    ax.fill_between(range(len(means)), means - stds, means + stds, alpha=0.5)
    ax.set_title(figtitle)
    if filename != "":
        fig.savefig(filename)
        plt.close(fig)
    else:
        return fig, ax
