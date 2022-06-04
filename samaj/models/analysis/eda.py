from typing import Tuple

# Data Analysis and Visualization
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


# ==============================================================================
# EXPLORATORY DATA ANALYSIS & VISUALIZATION HELPERS
# ==============================================================================


def is_normal_distribution(data: np.array) -> bool:
    """Determines if the given 1D array is normally distributed.
    As per the definition, we will check whether the mean == median == mode.
    Parameter:
    data (np.array): array-like, contains numbers
    Returns: bool
    """
    mean, median = data.mean(), np.median(data)
    modes, _ = stats.mode(data)
    print(f"Mean of data: {mean}.")
    print(f"Median of data: {median}.")
    print(f"Mode(s) of data: {modes}.")
    return mean == median and np.isin(mean, modes)


def find_remove_outlier_iqr(data: np.array) -> Tuple[np.array, np.array]:
    """Remove the outliers from the X data.
    Parameter:
    data (np.array): array-like, contains numbers
    Returns: tuple: 2 arrays that have the outlier points
                    and the rest of the data, respectively
    """
    # A: calculate interquartile range
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    print(f"IQR of data: {iqr}.")
    # B: Calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # C: identify outliers
    outliers = data[np.where(np.logical_or(data < lower, data > upper))]
    # D: remove outliers
    data_with_no_outliers = data[np.where(np.logical_and(data > lower, data < upper))]
    return outliers, data_with_no_outliers


def make_heatmap(df: pd.DataFrame, map_title: str, figure_title=None) -> None:
    """Plots the diagonal correlation matrix of a dataset using Seaborn.
    Credit to the Seaborn Documentation for inspiring this cell:
    https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    Parameters:
    df(pandas.DataFrame): encapsulates the dataset being used to make heatmap.
    map_title(str): the title for the heatmap (shown in the figure itself)
    figure_title(str): the filename for the PNG version of the plot. Optional.
    Returns: None
    """
    sns.set_style("darkgrid")
    # A: Compute the correlation matrix
    corr = df.corr()
    # B: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # C: Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))
    # D: Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # E: Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )
    plt.title(map_title)
    # F: save it (if desired) and show!
    if figure_title:
        plt.savefig(figure_title)
    plt.show()


def plot3D(
    X_domain: np.array,
    Y_domain: np.array,
    Z: np.array,
    xyz=None,
    title=None,
    E=None,
    A=None,
    figname=None,
    save=False,
) -> None:
    """
    Plots a 3D plane using Matplotlib.
    This is mainly intended to be used in part 2 of this assessment.
    It heavily leverages the plot_surface() function from Matplotlib.pyplot:
    https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html#surface-plots
    Parameters:
    X_domain, Y_domain (np.array): these are two square matrices
                                   which (together) represent the
                                   domain over which Z is calculated
    Z(np.array): represents the Z-values needed for the surface
    xyz(np.array): optional array of XYZ points to scatter on the plot
    title(str): for the plot
    E(int), A(int): if provided, these determine the angle at which the plot
                    is viewed - i.e. they rotate the X and Y axis respectively
    figname(str): what to name the PNG file of the saved plot image
    save(bool): whether or not to save the plot as an image file
    Returns: None
    """
    # A: set up a new figure size
    fig = plt.figure(figsize=((12, 8)))
    # B: get the 3D axes
    ax = fig.gca(projection="3d")
    # C: plot the plane!
    ax.plot_surface(X_domain, Y_domain, Z, cmap=cm.Spectral)
    # D: make the plot interactive
    if E and A:
        ax.view_init(E, A)
    # E: plot the observations as well
    if xyz is not None:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        plt.scatter(x, y, zs=z, c="g", s=20)
    # F: stylize!
    if title:
        plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    if save:  # only save if intentional, otherwise overwrites previous image
        plt.savefig(figname)
    plt.show()
