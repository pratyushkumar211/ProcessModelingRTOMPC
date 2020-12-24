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
from hybridid import (PickleTool, PAPER_FIGSIZE, plot_profit_curve)

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

def plot_inputs(t, u, figure_size, ylabel_xcoordinate):
    """ Plot the training input data."""
    nrow = len(ulabels)
    (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                  sharex=True, figsize=figure_size,
                                  gridspec_kw=dict(wspace=0.4))
    for row in range(nrow):
        handle = axes[row].step(t, u[:, row], 'b')
        axes[row].set_ylabel(ulabels[row])
        axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes[row].set_xlabel('Time (hr)')
    axes[row].set_xlim([np.min(t), np.max(t)])
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

def plot_openloop_data(*, t, u, ydatum, xdatum,
                          legend_names, legend_colors,
                          figure_size=PAPER_FIGSIZE,
                          linewidth=0.8, markersize=1.):
    figures = []
    figures += plot_inputs(t, u, figure_size, -0.1)
    figures += plot_outputs(t, ydatum, figure_size, -0.1, 
                            legend_names, legend_colors)
    figures += plot_states(t, xdatum, figure_size, -0.25, 
                           legend_names, legend_colors)

    # Return the figure object.
    return figures

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
    xdatum, ydatum = [], []
    for simdata in simdata_list:
        t, x, y, u = get_plotting_arrays(simdata, plot_range)
        xdatum += [x]
        ydatum += [y]
    return (t, u, ydatum, xdatum)

def main():
    """ Load the pickle file and plot. """
    cstr_flash_parameters = PickleTool.load(filename=
                                            "cstr_flash_parameters.pickle",
                                            type='read')
    cstr_flash_train = PickleTool.load(filename="cstr_flash_train.pickle",
                                       type='read')
    val_predictions = cstr_flash_train['val_predictions']
    simdata_list = [cstr_flash_parameters['training_data'][-1], 
                    cstr_flash_parameters['greybox_val_data']]
    (t, u, ydatum, xdatum) = get_datum(simdata_list=simdata_list, 
                                       plot_range = (120, 12*60))
    ydatum.append(val_predictions[0].y[:600, :])
    legend_names = ['Plant', 'Grey-Box', 'Black-box']
    legend_colors = ['b', 'g', 'm']
    figures = []
    figures += plot_openloop_data(t=t, u=u, ydatum=ydatum,
                                  xdatum=xdatum,
                                  legend_names=legend_names,
                                  legend_colors=legend_colors)
    #figures += plot_val_model_predictions(plantsim_data=
    #                              tworeac_parameters['training_data'][-1],
    #        modelsim_datum=[tworeac_parameters['greybox_validation_data'], 
    #                        tworeac_train['val_predictions']],
    #                plot_range=(0, 6*60), 
    #            tsteps_steady=tworeac_parameters['parameters']['tsteps_steady'])
    
    #figures += plot_profit_curve(us=ssopt['us'],
    #                             costs=ssopt['costs'])
    with PdfPages('cstr_flash_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()