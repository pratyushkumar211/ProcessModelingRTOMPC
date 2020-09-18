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
                          linewidth=0.8, 
                          markersize=1.):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=5, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    (start, end) = plot_range
    ylabels = [r'$C_a \ (\textnormal{mol/m}^3)$', 
               r'$C_b \ (\textnormal{mol/m}^3)$', 
               r'$C_c \ (\textnormal{mol/m}^3)$',
               r'$C_d \ (\textnormal{mol/m}^3)$',
               r'$F \ (\textnormal{m}^3/\textnormal{min})$']
    time = training_data.time/60
    data_list = [training_data.Ca, training_data.Cb, 
                 training_data.Cc, training_data.Cd, training_data.F]
    for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
        if ylabel == r'$C_c \ (\textnormal{mol/m}^3)$':
            axes.plot(time[start:end], data[start:end], 'bo', 
                      markersize=markersize)
        else:
            axes.plot(time[start:end], data[start:end], 'b')
        axes.set_ylabel(ylabel)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    # Return the figure object.
    return [figure]

def plot_val_model_predictions(*, plantsim_data, 
                                  modelsim_datum, plot_range,
                                  figure_size=PRESENTATION_FIGSIZE,
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
               r'$F \ (\textnormal{m}^3/\textnormal{min})$']
    legend_colors = ['g']
    legend_handles = []
    plant_data_list = [plantsim_data.Ca, plantsim_data.Cb, 
                       plantsim_data.Cc, plantsim_data.F]
    for (modelsim_data, 
         legend_color) in zip(modelsim_datum, legend_colors):
        time = modelsim_data.time/60
        model_data_list = [modelsim_data.Ca, modelsim_data.Cb, 
                           modelsim_data.Cc, modelsim_data.F]
        for (axes, plantdata, 
             modeldata, ylabel) in zip(axes_array, plant_data_list, 
                                       model_data_list, ylabels):
            if ylabel == r'$C_c \ (\textnormal{mol/m}^3)$':
                axes.plot(time[start:end], plantdata[start:end], 'bo', 
                           markersize=markersize)
                axes.plot(time[start:end], 
                            modeldata[start:end], legend_color)
            else:
                plant_legend_handle = axes.plot(time[start:end], 
                                                plantdata[start:end], 'b')
                model_legend_handle = axes.plot(time[start:end], 
                          modeldata[start:end], legend_color)
            axes.set_ylabel(ylabel)
            axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        legend_handles += model_legend_handle
    legend_handles.insert(0, plant_legend_handle[0])
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    figure.legend(handles = legend_handles,
                  labels = ('Plant', 'Grey-box'), 
                  loc = (0.32, 0.9), ncol=2)
    # Return the figure object.
    return [figure]

#def plot_idmodel_vs_training_data():
#    """ Function to plot the data requirements 
#        of the hybrid model. """
#    None
#    return 

def main():
    """ Load the pickle file and plot. """
    threereac_parameters = PickleTool.load(filename=
                                       "threereac_parameters.pickle", 
                                       type='read')
    #threereac_train = PickleTool.load(filename=
    #                                   "threereac_train.pickle", 
    #                                  type='read')
    figures = []
    figures += plot_training_data(training_data=
                                  threereac_parameters['training_data'][0], 
                                  plot_range=(0, 4*60))
    figures += plot_val_model_predictions(plantsim_data=
                                  threereac_parameters['training_data'][-1],
            modelsim_datum=[threereac_parameters['greybox_validation_data']],
                    plot_range=(0, 4*60))
    with PdfPages('threereac_plots.pdf', 
                  'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()