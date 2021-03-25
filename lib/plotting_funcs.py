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
import itertools

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
                       ylabel_xcoordinate, title_loc, 
                       font_size):
        """ Plot the performance loss economic MPC parameters."""
        plt.rcParams.update({'font.size': font_size})
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
                        ylabel_xcoordinate, left_label_frac, 
                        font_size):
        """ Plot the profit curves. """
        plt.rcParams.update({'font.size': font_size})
        (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                      sharex=True, 
                                      figsize=figure_size, 
                                      gridspec_kw=dict(left=left_label_frac))
        xlabel = r'$C_{Afs} \ (\textnormal{mol/m}^3)$'
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

class CstrFlashPlots:

    ylabels = [r'$H_r \ (\textnormal{m})$',
               r'$C_{Br} \ (\textnormal{mol/m}^3)$', 
               r'$T_r \ (K)$',
               r'$H_b \ (\textnormal{m})$',
               r'$C_{Bb} \ (\textnormal{mol/m}^3)$',
               r'$T_b \ (K)$']

    xlabels = [r'$H_r \ (\textnormal{m})$',
               r'$C_{Ar} \ (\textnormal{mol/m}^3)$', 
               r'$C_{Br} \ (\textnormal{mol/m}^3)$',
               r'$C_{Cr} \ (\textnormal{mol/m}^3)$',
               r'$T_r \ (K)$',
               r'$H_b \ (\textnormal{m})$',
               r'$C_{Ab} \ (\textnormal{mol/m}^3)$',
               r'$C_{Bb} \ (\textnormal{mol/m}^3)$',
               r'$C_{Cb} \ (\textnormal{mol/m}^3)$',
               r'$T_b \ (K)$']

    ulabels = [r'$F \ (\textnormal{m}^3/\textnormal{min})$',
               r'$D \ (\textnormal{m}^3/\textnormal{min})$']

    def plot_inputs(t, ulist, figure_size, ylabel_xcoordinate, 
                    plot_ulabel, legend_names, legend_colors, 
                    title_loc):
        """ Plot the training input data. """
        nrow = len(CstrFlashPlots.ulabels)
        (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                      sharex=True, figsize=figure_size,
                                      gridspec_kw=dict(wspace=0.5))
        legend_handles = []
        for (u, color) in zip(ulist, legend_colors):
            for row in range(nrow):
                handle = axes[row].step(t, u[:, row], color, where='post')
                axes[row].set_ylabel(CstrFlashPlots.ulabels[row])
                axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
            axes[row].set_xlabel('Time (hr)')
            axes[row].set_xlim([np.min(t), np.max(t)])
            legend_handles += handle
        if plot_ulabel:
            figure.legend(handles = legend_handles,
                        labels = legend_names,
                        loc = title_loc, ncol=len(legend_names))
        return [figure]

    def plot_outputs(t, ylist, figure_size, ylabel_xcoordinate,
                     legend_names, legend_colors, title_loc):
        """ Plot the training input data."""
        nrow = len(CstrFlashPlots.ylabels)
        (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                    sharex=True, figsize=figure_size,
                                    gridspec_kw=dict(wspace=0.5))
        legend_handles = []
        for (y, color) in zip(ylist, legend_colors):
            for row in range(nrow):
                handle = axes[row].plot(t, y[:, row], color)
                axes[row].set_ylabel(CstrFlashPlots.ylabels[row])
                axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
            axes[row].set_xlabel('Time (hr)')
            axes[row].set_xlim([np.min(t), np.max(t)])
            legend_handles += handle
        figure.legend(handles = legend_handles,
                      labels = legend_names,
                      loc = title_loc, ncol=len(legend_names))
        return [figure]

    def plot_states(t, xlist, figure_size, ylabel_xcoordinate,
                    legend_names, legend_colors, title_loc):
        """ Plot the training outputs. """
        nrow, ncol = 5, 2
        (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
                                      sharex=True, figsize=figure_size,
                                      gridspec_kw=dict(wspace=0.5))
        legend_handles = []
        for (x, color) in zip(xlist, legend_colors):
            state_index = 0
            for row, col in itertools.product(range(nrow), range(ncol)):
                handle = axes[row, col].plot(t, x[:, state_index], color)
                axes[row, col].set_ylabel(CstrFlashPlots.xlabels[state_index])
                axes[row, col].get_yaxis().set_label_coords(ylabel_xcoordinate, 
                                                            0.5)
                if row == nrow - 1:
                    axes[row, col].set_xlabel('Time (hr)')
                    axes[row, col].set_xlim([np.min(t), np.max(t)])
                state_index += 1
            legend_handles += handle
        figure.legend(handles = legend_handles,
                    labels = legend_names,
                    loc = title_loc, ncol=len(legend_names))
        return [figure]

    def plot_data(*, t, ulist, ylist, xlist,
                     figure_size, u_ylabel_xcoordinate, 
                     x_ylabel_xcoordinate, legend_names, 
                     legend_colors, title_loc, y_ylabel_xcoordinate=None, 
                     plot_y=True, plot_ulabel=True):
        figures = []
        figures += CstrFlashPlots.plot_inputs(t, ulist, figure_size, 
                               u_ylabel_xcoordinate,
                               plot_ulabel, legend_names, legend_colors, 
                               title_loc)
        if plot_y:
            figures += CstrFlashPlots.plot_outputs(t, ylist, figure_size, 
                                   y_ylabel_xcoordinate, legend_names, 
                                   legend_colors, title_loc)
        figures += CstrFlashPlots.plot_states(t, xlist, figure_size, 
                               x_ylabel_xcoordinate, legend_names, 
                               legend_colors, title_loc)

        # Return the figure object.
        return figures

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