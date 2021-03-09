# Pratyush Kumar, pratyushkumar@ucsb.edu

import sys
import numpy as np
import mpctools as mpc
import scipy.linalg
import matplotlib.pyplot as plt
import casadi
import collections
import pickle
import plottools
import time

FIGURE_SIZE_A4 = (9, 10)
PRESENTATION_FIGSIZE = (6, 6)
PAPER_FIGSIZE = (6, 6)

def get_plotting_arrays(simdata, plot_range):
    """ Get data and return for plotting. """
    start, end = plot_range
    u = simdata.u[start:end, :]
    x = simdata.x[start:end, :]
    y = simdata.y[start:end, :]
    t = simdata.t[start:end]/60 # Convert to hours.
    # Return t, x, y, u.
    return (t, x, y, u)

def get_plotting_array_list(*, simdata_list, plot_range):
    """ Get all data as lists. """
    ulist, xlist, ylist = [], [], []
    for simdata in simdata_list:
        t, x, y, u = get_plotting_arrays(simdata, plot_range)
        ulist += [u]
        xlist += [x]
        ylist += [y]
    # Return lists.
    return (t, ulist, ylist, xlist)

class TwoReacPlots:
    """ Single class containing functions for the two reaction
        example system. """

    labels = [r'$C_A \ (\textnormal{mol/m}^3)$', 
              r'$C_B \ (\textnormal{mol/m}^3)$',
              r'$C_C \ (\textnormal{mol/m}^3)$',
              r'$C_{Af} \ (\textnormal{mol/m}^3)$']

    @staticmethod
    def plot_xudata(*, t, xlist, ulist, legend_names, 
                       legend_colors, figure_size,
                       ylabel_xcoordinate, title_loc):
        """ Plot the performance loss economic MPC parameters."""
        nrow = len(TwoReacPlots.labels)
        (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                      sharex=True, figsize=figure_size,
                                      gridspec_kw=dict(wspace=0.4))
        legend_handles = []
        for (x, u, color) in zip(xlist, ulist, legend_colors):
            # First plot the states.
            for row in range(nrow-1):
                handle = axes[row].plot(t, x[:, row], color)
                axes[row].set_ylabel(TwoReacPlots.labels[row])
                axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
            # Plot the input in the last row.
            row += 1
            axes[row].step(t, u[:, 0], color, where='post')
            axes[row].set_ylabel(TwoReacPlots.labels[row])
            axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
            axes[row].set_xlabel('Time (hr)')
            axes[row].set_xlim([np.min(t), np.max(t)])
            legend_handles += handle
        figure.legend(handles = legend_handles,
                      labels = legend_names,
                      loc = title_loc, ncol=len(legend_names))
        # Return figure.
        return [figure]

    @staticmethod
    def plot_sscosts(*, us, sscosts, legend_colors, 
                        legend_names, figure_size, 
                        ylabel_xcoordinate, left_label_frac):
        """ Plot the profit curves. """
        (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                      sharex=True, 
                                      figsize=figure_size, 
                                      gridspec_kw=dict(left=left_label_frac))
        xlabel = r'$C_{Af} \ (\textnormal{mol/m}^3)$'
        ylabel = r'Cost ($\$ $)'
        for (cost, color) in zip(sscosts, legend_colors):
            # Plot the corresponding data.
            axes.plot(us, cost, color)
        axes.legend(legend_names)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel, rotation=False)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
        axes.set_xlim([np.min(us), np.max(us)])
        return [figure]

# def plot_avg_profits(*, t, avg_stage_costs,
#                     legend_colors, legend_names, 
#                     figure_size=PAPER_FIGSIZE, 
#                     ylabel_xcoordinate=-0.15):
#     """ Plot the profit. """
#     (figure, axes) = plt.subplots(nrows=1, ncols=1,
#                                   sharex=True,
#                                   figsize=figure_size,
#                                   gridspec_kw=dict(left=0.15))
#     xlabel = 'Time (hr)'
#     ylabel = '$\Lambda_k$'
#     for (cost, color) in zip(avg_stage_costs, legend_colors):
#         # Plot the corresponding data.
#         profit = -cost
#         axes.plot(t, profit, color)
#     axes.legend(legend_names)
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel, rotation=True)
#     axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
#     axes.set_xlim([np.min(t), np.max(t)])
#     # Return.
#     return [figure]

# def plot_val_metrics(*, num_samples, val_metrics, colors, legends, 
#                      figure_size=PAPER_FIGSIZE,
#                      ylabel_xcoordinate=-0.11, 
#                      left_label_frac=0.15):
#     """ Plot validation metric on open loop data. """
#     (figure, axes) = plt.subplots(nrows=1, ncols=1, 
#                                   sharex=True, 
#                                   figsize=figure_size, 
#                                   gridspec_kw=dict(left=left_label_frac))
#     xlabel = 'Hours of training samples'
#     ylabel = 'MSE'
#     num_samples = num_samples/60
#     for (val_metric, color) in zip(val_metrics, colors):
#         # Plot the corresponding data.
#         axes.semilogy(num_samples, val_metric, color)
#     axes.legend(legends)
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
#     axes.set_xlim([np.min(num_samples), np.max(num_samples)])
#     figure.suptitle('Mean squared error (MSE) - Validation data', 
#                     x=0.52, y=0.92)
#    # Return the figure object.
#     return [figure]