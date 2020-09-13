""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PRESENTATION_FIGSIZE)

def plot_training_data(*, training_data, plot_range,
                          figure_size=PRESENTATION_FIGSIZE,
                          ylabel_xcoordinate=-0.1, 
                          linewidth=0.8):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=5, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    (start, end) = plot_range
    ylabels = [r'$C_a \ (\textnormal{mol/m}^3)$', 
               r'$C_b \ (\textnormal{mol/m}^3)$', 
               r'$C_c \ (\textnormal{mol/m}^3)$',
               r'$C_d \ (\textnormal{mol/m}^3)$',
               r'$C_{a0} \ (\textnormal{mol/m}^3)$']
    time = training_data.time/60
    data_list = [training_data.Ca, training_data.Cb, 
                 training_data.Cc, training_data.Cd, training_data.Ca0]
    for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
        if ylabel == '$C_c(\textnormal{mol/m}^3)':
            axes.plot(time, data[start:end], 'o')
        else:
            axes.plot(time, data[start:end])
        axes.set_ylabel(ylabel)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    # Return the figure object.
    return [figure]

def plot_val_model_predictions(*, cascadesim_datum, plot_range,
                            figure_size=PRESENTATION_FIGSIZE,
                            ylabel_xcoordinate=-0.08, 
                            linewidth=0.8):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=3, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    ylabels = ['$T_{z}(^\circ$C)', '$\dot{Q}_c$ (kW)', '$i$']
    legend_colors = ['b', 'g']
    legend_handles = []
    for (cascadesim_data, 
         legend_color) in zip(cascadesim_datum, legend_colors):
        time = cascadesim_data.time/3600
        data_list = [(cascadesim_data.Tzone, cascadesim_data.Tzone_sp), 
                      cascadesim_data.upi, cascadesim_data.i]
        for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
            if ylabel == '$T_{z}(^\circ$C)':
                (Tzone, Tsp) = data
                legend_temp_handle = axes.plot(time, Tzone, legend_color)
                axes.plot(time, Tsp, '--' + legend_color)
                axes.plot(time, cascadesim_data.Tzone_lb, 'k')
                axes.plot(time, cascadesim_data.Tzone_ub, 'k')
            elif ylabel == '$\dot{Q}_c$ (kW)':
                axes.plot(time, data, legend_color)
                axes.plot(time, cascadesim_data.Qcool_lb, 'k')
                axes.plot(time, cascadesim_data.Qcool_ub, 'k')
            else:
                axes.plot(time, data, legend_color)
            axes.set_ylabel(ylabel, rotation=False)
            axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        legend_handles += legend_temp_handle 
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    figure.legend(handles = legend_handles,
                  labels = (r'$\alpha$ = 30',
                            r'$\alpha$ = 50'), 
                  loc = (0.32, 0.9), ncol=3)
    # Return the figure object.
    return [figure]

def main():
    """ Load the pickle file and plot. """
    threereac_parameters = PickleTool.load(filename=
                                       "threereac_parameters.pickle", 
                                       type='read')
    figures = []
    figures += plot_training_data(training_data=
                                  threereac_parameters['train_val_datum'][0], 
                                  plot_range=(0, 24*60))
    with PdfPages('threereac_plots.pdf', 
                  'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()
