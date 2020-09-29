# [depends] tworeac_parameters.pickle
# [depends] %LIB%/hybridid.py
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE)

def plot_training_data(*, training_data, plot_range,
                          figure_size=PAPER_FIGSIZE,
                          ylabel_xcoordinate=-0.1, 
                          linewidth=0.8, 
                          markersize=1.):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=4, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    (start, end) = plot_range
    ylabels = [r'$C_a \ (\textnormal{mol/m}^3)$', 
               r'$C_b \ (\textnormal{mol/m}^3)$', 
               r'$C_c \ (\textnormal{mol/m}^3)$',
               r'$C_{a0} \ (\textnormal{mol/m}^3)$']
    time = training_data.t/60
    data_list = [training_data.y[:, 0], training_data.y[:, 1], 
                 training_data.x[:, 2], training_data.u]
    for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
        if ylabel in [r'$C_a \ (\textnormal{mol/m}^3)$' , 
                      r'$C_b \ (\textnormal{mol/m}^3)$']:
            axes.plot(time[start:end], data[start:end], 'bo', 
                      markersize=markersize)
        else:
            axes.plot(time[start:end], data[start:end])
        axes.set_ylabel(ylabel)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    # Return the figure object.
    return [figure]

def plot_val_model_predictions(*, plantsim_data, 
                                  modelsim_datum, plot_range,
                                  tsteps_steady,
                                  figure_size=PAPER_FIGSIZE,
                                  ylabel_xcoordinate=-0.1, 
                                  linewidth=0.8, 
                                  markersize=1.):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=3, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    (start, end) = plot_range
    ylabels = [r'$C_A \ (\textnormal{mol/m}^3)$', 
               r'$C_B \ (\textnormal{mol/m}^3)$',
               r'$C_{A0} \ (\textnormal{mol/m}^3)$']
    model_legend_colors = ['g', 'm']
    legend_handles = []
    plant_data_list = [plantsim_data.y[tsteps_steady:, 0], 
                       plantsim_data.y[tsteps_steady:, 1], 
                       plantsim_data.u[tsteps_steady:]]
    for (modelsim_data, 
         model_legend_color) in zip(modelsim_datum, model_legend_colors):
        time = plantsim_data.t[tsteps_steady:]/60
        model_data_list = [modelsim_data.y[:, 0], modelsim_data.y[:, 1], 
                           plantsim_data.u[tsteps_steady:]]
        for (axes, plantdata, 
             modeldata, ylabel) in zip(axes_array, plant_data_list, 
                                       model_data_list, ylabels):
            if ylabel == r'$C_c \ (\textnormal{mol/m}^3)$':
                axes.plot(time[start:end], plantdata[start:end])
                axes.plot(time[start:end], 
                            modeldata[start:end], model_legend_color)
            else:
                plant_legend_handle = axes.plot(time[start:end], 
                                                plantdata[start:end], 'b')
                model_legend_handle = axes.plot(time[start:end], 
                          modeldata[start:end], model_legend_color)
            axes.set_ylabel(ylabel)
            axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        legend_handles += model_legend_handle
    legend_handles.insert(0, plant_legend_handle[0])
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    figure.legend(handles = legend_handles,
                  labels = ('Plant', 'Grey-box', 'Hybrid-Model'), 
                  loc = (0.3, 0.9), ncol=2)
    # Return the figure object.
    return [figure]

#def plot_idmodel_vs_training_data():
#    """ Function to plot the data requirements 
#        of the hybrid model. """
#    None
#    return 

def main():
    """ Load the pickle file and plot. """
    tworeac_parameters = PickleTool.load(filename=
                                         "tworeac_parameters_nonlin.pickle", 
                                         type='read')
    tworeac_train = PickleTool.load(filename=
                                    "tworeac_train_nonlin.pickle", 
                                    type='read')
    figures = []
    figures += plot_training_data(training_data=
                                  tworeac_parameters['training_data'][0], 
                                  plot_range=(0, 4*60))
    figures += plot_val_model_predictions(plantsim_data=
                                  tworeac_parameters['training_data'][-1],
            modelsim_datum=[tworeac_parameters['greybox_validation_data'], 
                            tworeac_train['val_predictions']],
                    plot_range=(0, 3*60), 
                tsteps_steady=tworeac_parameters['parameters']['tsteps_steady'])
    with PdfPages('tworeac_plots_nonlin.pdf', 
                  'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()