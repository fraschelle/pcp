#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from copy import deepcopy


def check_data(data, labels):
    """Check whether each row of data array has the same length as labels."""
    for i in range(len(data)):
        assert len(data[i]) is len(labels), "data dimension (%d) does not " \
            "match with labels (%d)" % (len(data[i]), len(labels))


def check_styles(data, styles):
    """Check whether each row of data array has a provided style."""
    if styles:
        assert len(data) is len(styles), "data dimension (%d) does not " \
            "match with styles (%d)" % (len(data), len(styles))
    else:
        return [dict(color="steelblue",
                     linestyle="solid",
                     linewidth=0.25,
                     clip_on=False,)] * len(data)
    return styles


def check_formatting(ytype, labels):
    """Check whether ytype and labels are compatible (lists of same dimension), and construct an ytype if ytype is not provided."""
    if ytype:
        assert len(ytype) is len(labels), "dimension (%d) does not " \
            "match with labels (%d)" % (len(ytype), len(labels))
    else:
        ytype = [[]] * len(labels)
    return ytype


def set_ytype(ytype, data, colorbar):
    for i in range(len(ytype)):
        if not ytype[i]:
            if type(data[0][i]) is str:
                ytype[i] = "categorial"
            else:
                ytype[i] = "linear"
    if colorbar: 
        assert ytype[len(ytype) - 1] == "linear", "colorbar axis needs to " \
            "be linear"
    return ytype


def set_ylabels(ylabels, data, ytype):
    for i in range(len(ylabels)): 
        # Generate ylabels for string values
        if not ylabels[i] and ytype[i] == "categorial":
            ylabel = []
            for j in range(len(data)):
                if data[j][i] not in ylabel:
                    ylabel.append(data[j][i])
            ylabel.sort()
            if len(ylabel) == 1:
                ylabel.append("")
            ylabels[i] = ylabel
    return ylabels


def replace_str_values(data, ytype, ylabels):
    for i in range(len(ytype)):
        if ytype[i] == "categorial":
            for j in range(len(data)):
                data[j][i] = ylabels[i].index(data[j][i])
    return np.array(data).transpose()


def set_ylim(ylim, data):
    for i in range(len(ylim)):
        if not ylim[i]:
            ylim[i] = [np.min(data[i, :]), np.max(data[i, :])]
            if ylim[i][0] == ylim[i][1]:
                ylim[i] = [ylim[i][0] * 0.95, ylim[i][1] * 1.05]
            if ylim[i] == [0.0, 0.0]:
                ylim[i] = [0.0, 1.0]
    return ylim


def get_score(data, ylim):
    """Construct a score based on the last row of data."""
    ymin = ylim[len(ylim) - 1][0]
    ymax = ylim[len(ylim) - 1][1]
    score = (np.copy(data[len(ylim) - 1, :]) - ymin) / (ymax - ymin)
    return score


# Rescale data of secondary y-axes to scale of first y-axis
def rescale_data(data, ytype, ylim):
    """Rescale the data according the ytype (in particular, in case one wants a log-scale."""
    min0 = ylim[0][0]
    max0 = ylim[0][1]
    scale = max0 - min0
    for i in range(1, len(ylim)):
        mini = ylim[i][0]
        maxi = ylim[i][1]
        if ytype[i] == "log":
            logmin = np.log10(mini)
            logmax = np.log10(maxi)
            span = logmax - logmin
            data[i, :] = ((np.log10(data[i, :]) - logmin) / span) * scale + min0
        else:
            data[i, :] = ((data[i, :] - mini) / (maxi - mini)) * scale + min0
    return data


def get_path(data, i):
    """Construct the Path object associated to the i-th column of data."""
    n = data.shape[0] # number of y-axes
    verts = list(zip([x for x in np.linspace(0, n - 1, n * 3 - 2)], 
        np.repeat(data[:, i], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    return path


def parallel_coordinates(
    data,
    labels,
    styles=[],
    ytype=None,
    ylim=None,
    ylabels=None,
    figsize=(10, 5),
    rect=[0.125, 0.1, 0.75, 0.8],
    curves=True,
    alpha=1.0,
    colorbar=True,
    colorbar_width=0.02,
    colormap=plt.get_cmap("inferno"),
):
    """
    Parallel Coordinates Plot 

    Parameters
    ----------
    data: nested array
        Inner arrays containing data for each curve.
    labels: list
        Labels for y-axes.
    styles: list, optional
        linestyle for each data curve, default is the same style (blue curve)
    ytype: list, optional
        Default "None" allows linear axes for numerical values and categorial 
        axes for data of type string. If ytype is passed, logarithmic axes are 
        also possible, e.g.  ["categorial", "linear", "log", [], ...]. Vacant 
        fields must be filled with an empty list []. 
    ylim: list, optional
        Custom min and max values for y-axes, e.g. [[0, 1], [], ...].
    ylabels: list, optional (not recommended)
        Only use this option if you want to print more categories than you have
        in your dataset for categorial axes. You also have to set the right 
        ylim for this option to work correct.
    figsize: (float, float), optional
        Width, height in inches.
    rect: array, optional
        [left, bottom, width, height], defines the position of the figure on
        the canvas. This is also the position of the left-most axe.
    curves: bool, optional
        If True, B-spline curve is drawn. Default is True.
    alpha: float, optional
        Alpha value for blending the curves. In use only when curves is True.
    colorbar: bool, optional
        If True, colorbar is drawn using the last value (column) of data.
    colorbar_width: float, optional
        Defines the width of the colorbar. Default is 0.02.
    colormap: matplotlib.colors.Colormap, optional
        Specify colors for colorbar. Default is inferno.

    Returns
    -------
    `~matplotlib.figure.Figure`
    """

    [left, bottom, width, height] = rect
    data = deepcopy(data)

    # Check data
    check_data(data, labels)
    styles = check_styles(data, styles)
    ytype = check_formatting(ytype, labels)
    ylim = check_formatting(ylim, labels)
    ylabels = check_formatting(ylabels, labels)

    # Setup data
    ytype = set_ytype(ytype, data, colorbar)
    ylabels = set_ylabels(ylabels, data, ytype)
    data = replace_str_values(data, ytype, ylabels)
    ylim = set_ylim(ylim, data)
    score = get_score(data, ylim)
    data = rescale_data(data, ytype, ylim)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_axes([left, bottom, width, height])
    axes = [ax0] + [ax0.twinx() for i in range(data.shape[0] - 1)]

    # Plot curves
    for i in range(data.shape[1]):
        if colorbar:
            style = dict(color=colormap(score[i]),
                         linestyle="solid",
                         linewidth=1.25,
                         clip_on=False)
        else:
            style = styles[i]
        if curves:
            path = get_path(data, i)
            patch = PathPatch(
                path,
                facecolor="None",
                lw=style['linewidth'],
                alpha=alpha,
                edgecolor=style['color'],
                clip_on=style['clip_on'])
            ax0.add_patch(patch)
        else:
            ax0.plot(data[:, i], **style)

    # Format x-axis
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position("none")
    ax0.set_xlim([0, data.shape[0] - 1])
    ax0.set_xticks(range(data.shape[0]))
    ax0.set_xticklabels(labels)

    # Format y-axis
    for i, ax in enumerate(axes):
        ax.spines["left"].set_position(("axes", 1 / (len(labels) - 1) * i))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.yaxis.set_ticks_position("left")
        ax.set_ylim(ylim[i])
        if ytype[i] == "log":
            ax.set_yscale("log")
        if ytype[i] == "categorial":
            ax.set_yticks(range(len(ylabels[i])))
        if ylabels[i]:
            ax.set_yticklabels(ylabels[i])

    if colorbar:
        bar = fig.add_axes([left + width, bottom, colorbar_width, height])
        norm = mpl.colors.Normalize(vmin=ylim[i][0], vmax=ylim[i][1])
        mpl.colorbar.ColorbarBase(bar,
                                  cmap=colormap,
                                  norm=norm,
                                  orientation="vertical")
        bar.tick_params(size=0)
        bar.set_yticklabels([])

    return fig


if __name__ == "__main__":
    # Minimal working examples
    data = [["ResNet", 0.0001, 4, 0.2],
            ["ResNet", 0.0003, 8, 1.0],
            ["DenseNet", 0.0005, 4, 0.65],
            ["DenseNet", 0.0007, 8, 0.45],
            ["DenseNet", 0.001, 2, 0.8]]
    labels = ["Network", "Learning rate", "Batchsize", "F-Score"]
    parallel_coordinates(data, labels)
    plt.show()
    parallel_coordinates(data, labels, curves=False)
    plt.show()
    parallel_coordinates(data, labels, curves=False, colorbar=False)
    plt.show()
    colormap = plt.get_cmap('Pastel2')
    styles = [dict(color=colormap(0), linestyle='dashed', linewidth=6.75),
              dict(color='orange', linestyle='solid', linewidth=0.75),
              dict(linestyle='dotted', linewidth=1.75),
              dict(linestyle='dashed', linewidth=2.5),
              dict(color=colormap(2), linestyle='dashdot', linewidth=5, 
                   marker='D', markersize=12)]
    parallel_coordinates(
        data, labels, curves=False, colorbar=False, styles=styles)
    plt.show()
