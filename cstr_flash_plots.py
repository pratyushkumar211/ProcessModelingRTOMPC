# [depends] cstr_flash_parameters.pickle cstr_flash_train.pickle
# [depends] cstr_flash_ssopt.pickle
# [depends] %LIB%/hybridid.py
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE)

ylabels = [r'$H_r \ (\textnormal{m})$',
           r'$C_{Ar} \ (\textnormal{mol/m}^3)$', 
           r'$T_r \ (K)$',
           r'$H_b \ (\textnormal{m})$', 
           r'$C_{Ab} \ (\textnormal{mol/m}^3)$',
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
           r'$Q_r \ (\textnormal{KJ/min})$',
           r'$D \ (\textnormal{m}^3/\textnormal{min})$',
           r'$Q_b \ (\textnormal{KJ/min})$']

def plot_inputs(t, udatum, figure_size, ylabel_xcoordinate, 
                   data_type, legend_names, legend_colors):
    """ Plot the training input data. """
    nrow = len(ulabels)
    if data_type == 'open_loop':
        udatum = [udatum[0]]
    (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                  sharex=True, figsize=figure_size,
                                  gridspec_kw=dict(wspace=0.4))
    legend_handles = []
    for (u, color) in zip(udatum, legend_colors):
        for row in range(nrow):
            handle = axes[row].step(t, u[:, row], color)
            axes[row].set_ylabel(ulabels[row])
            axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        axes[row].set_xlabel('Time (hr)')
        axes[row].set_xlim([np.min(t), np.max(t)])
        if data_type == 'closed_loop' and row == 3:
            axes[row].set_ylim([100., 500.])
        legend_handles += handle
    if data_type == 'closed_loop':
        figure.legend(handles = legend_handles,
                      labels = legend_names,
                      loc = (0.32, 0.9), ncol=len(legend_names))
    return [figure]

def plot_outputs(t, ydatum, figure_size, ylabel_xcoordinate,
                 legend_names, legend_colors):
    """ Plot the training input data."""
    nrow = len(ylabels)
    (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                  sharex=True, figsize=figure_size,
                                  gridspec_kw=dict(wspace=0.4))
    legend_handles = []
    for (y, color) in zip(ydatum, legend_colors):
        for row in range(nrow):
            handle = axes[row].step(t, y[:, row], color)
            axes[row].set_ylabel(ylabels[row])
            axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        axes[row].set_xlabel('Time (hr)')
        axes[row].set_xlim([np.min(t), np.max(t)])
        legend_handles += handle
    figure.legend(handles = legend_handles,
                  labels = legend_names,
                  loc = (0.32, 0.9), ncol=len(legend_names))
    return [figure]

def plot_states(t, xdatum, figure_size, ylabel_xcoordinate,
                legend_names, legend_colors):
    """ Plot the training outputs. """
    nrow, ncol = 5, 2
    (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
                                  sharex=True, figsize=figure_size,
                                  gridspec_kw=dict(wspace=0.6))
    legend_handles = []
    for (x, color) in zip(xdatum, legend_colors):
        state_index = 0
        for row, col in itertools.product(range(nrow), range(ncol)):
            handle = axes[row, col].plot(t, x[:, state_index], color)
            axes[row, col].set_ylabel(xlabels[state_index])
            axes[row, col].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
            if row == nrow - 1:
                axes[row, col].set_xlabel('Time (hr)')
                axes[row, col].set_xlim([np.min(t), np.max(t)])
            state_index += 1
        legend_handles += handle
    figure.legend(handles = legend_handles,
                  labels = legend_names,
                  loc = (0.32, 0.9), ncol=len(legend_names))
    return [figure]

def plot_data(*, t, udatum, ydatum, xdatum,
                 data_type, legend_names, legend_colors,
                 figure_size=PAPER_FIGSIZE,
                 linewidth=0.8, markersize=1.):
    figures = []
    figures += plot_inputs(t, udatum, figure_size, -0.1,
                           data_type, legend_names, legend_colors)
    figures += plot_outputs(t, ydatum, figure_size, -0.1, 
                            legend_names, legend_colors)
    figures += plot_states(t, xdatum, figure_size, -0.25, 
                           legend_names, legend_colors)

    # Return the figure object.
    return figures

#def plot_openloop_sols(*, t, useq, xseq,
#                          legend_names, legend_colors,
#                          figure_size=PAPER_FIGSIZE):
#    figures = []
#    figures += plot_inputs(t, [useq], figure_size, -0.1,
#                           data_type, legend_names, ['legend_colors'])
#    figures += plot_states(t, [xseq], figure_size, -0.25, 
#                           legend_names, legend_colors)

    # Return the figure object.
#    return figures

def get_plotting_arrays(data, plot_range):
    """ Get data and return for plotting. """
    start, end = plot_range
    u = data.u[start:end, :]
    x = data.x[start:end, :]
    y = data.y[start:end, :]
    t = data.t[start:end]/60 # Convert to hours.
    return (t, x, y, u)

def get_datum(*, simdata_list, plot_range):
    """ Get all data as lists. """
    udatum, xdatum, ydatum = [], [], []
    for simdata in simdata_list:
        t, x, y, u = get_plotting_arrays(simdata, plot_range)
        udatum += [u]
        xdatum += [x]
        ydatum += [y]
    return (t, udatum, ydatum, xdatum)

def plot_cost_pars(t, cost_pars,
                   figure_size=PAPER_FIGSIZE, 
                   ylabel_xcoordinate=-0.15):
    """ Plot the economic MPC cost parameters. """
    num_pars = cost_pars.shape[1]
    (figure, axes_list) = plt.subplots(nrows=num_pars, ncols=1,
                                  sharex=True,
                                  figsize=figure_size,
                                  gridspec_kw=dict(left=0.18))
    xlabel = 'Time (hr)'
    ylabels = ['Energy Price ($\$$/kW)',
                'Raw Material Cost ($\$$/mol-A)',
                'Product Price ($\$$/mol-B)']
    for (axes, pari, ylabel) in zip(axes_list, range(num_pars), ylabels):
        # Plot the corresponding data.
        axes.plot(t, cost_pars[:len(t), pari])
        axes.set_ylabel(ylabel)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel(xlabel)
    axes.set_xlim([np.min(t), np.max(t)])
    return [figure]

def plot_avg_profits(*, t, avg_stage_costs,
                    legend_colors, legend_names, 
                    figure_size=PAPER_FIGSIZE, 
                    ylabel_xcoordinate=-0.15):
    """ Plot the profit. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                                  sharex=True,
                                  figsize=figure_size,
                                  gridspec_kw=dict(left=0.15))
    xlabel = 'Time (hr)'
    ylabel = '$\Lambda_k$'
    for (cost, color) in zip(avg_stage_costs, legend_colors):
        # Plot the corresponding data.
        profit = -cost
        axes.plot(t, profit, color)
    axes.legend(legend_names)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=True)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlim([np.min(t), np.max(t)])
    # Return.
    return [figure]

def main():
    """ Load the pickle file and plot. """

    # Get the data for plotting.
    cstr_flash_parameters = PickleTool.load(filename=
                                            "cstr_flash_parameters.pickle",
                                            type='read')
    cstr_flash_train = PickleTool.load(filename="cstr_flash_train.pickle",
                                       type='read')
    cstr_flash_empc = PickleTool.load(filename="cstr_flash_empc.pickle",
                                       type='read')

    # Collect data to plot open-loop predictions.
    val_predictions = cstr_flash_train['val_predictions']
    simdata_list = [cstr_flash_parameters['training_data'][-1], 
                    cstr_flash_parameters['greybox_val_data']]
    (t, udatum, ydatum, xdatum) = get_datum(simdata_list=simdata_list, 
                                       plot_range = (120, 14*60))
    ydatum.append(val_predictions[0].y[:720, :])
    xdatum.append(val_predictions[0].x[:720, :])
    legend_names = ['Plant', 'Grey-Box', 'Hybrid']
    legend_colors = ['b', 'g', 'm']
    figures = []
    figures += plot_data(t=t, udatum=udatum, ydatum=ydatum,
                              xdatum=xdatum, data_type='open_loop',
                              legend_names=legend_names,
                              legend_colors=legend_colors)

    # Plot the closed-loop simulation.
    legend_names = ['Plant', 'Hybrid']
    legend_colors = ['b', 'm']
    cl_data_list = cstr_flash_empc['cl_data_list']
    (t, udatum, ydatum, xdatum) = get_datum(simdata_list=cl_data_list,
                                       plot_range = (0, 24*60))
    figures += plot_data(t=t, udatum=udatum, ydatum=ydatum,
                              xdatum=xdatum, data_type='closed_loop',
                              legend_names=legend_names,
                              legend_colors=legend_colors)
    
    # Plot the plant profit in time.
    figures += plot_avg_profits(t=t,
                            avg_stage_costs=cstr_flash_empc['avg_stage_costs'], 
                            legend_colors=legend_colors,
                            legend_names=legend_names)

    # Plot the empc costs.
    figures += plot_cost_pars(t=t, 
                              cost_pars=cstr_flash_empc['cost_pars'][:24*60, :])

    # Save PDF.
    with PdfPages('cstr_flash_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()