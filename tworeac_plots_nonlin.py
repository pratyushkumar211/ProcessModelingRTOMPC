# [depends] tworeac_parameters_nonlin.pickle tworeac_train_nonlin.pickle
# [depends] tworeac_ssopt_nonlin.pickle
# [depends] %LIB%/hybridid.py
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE, plot_profit_curve)

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
    ylabels = [r'$C_A \ (\textnormal{mol/m}^3)$', 
               r'$C_B \ (\textnormal{mol/m}^3)$', 
               r'$C_C \ (\textnormal{mol/m}^3)$',
               r'$C_{Af} \ (\textnormal{mol/m}^3)$']
    time = training_data.t/60
    data_list = [training_data.y[:, 0], training_data.y[:, 1], 
                 training_data.x[:, 2], training_data.u]
    for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
        if ylabel in [r'$C_A \ (\textnormal{mol/m}^3)$' , 
                      r'$C_B \ (\textnormal{mol/m}^3)$']:
            axes.plot(time[start:end], data[start:end], 'bo', 
                      markersize=markersize)
        else:
            axes.plot(time[start:end], data[start:end])
        axes.set_ylabel(ylabel)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time[start:end]), np.max(time[start:end])])
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
               r'$C_{Af} \ (\textnormal{mol/m}^3)$']
    model_legend_colors = ['green', 'dimgray', 'tomato']
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
    axes.set_xlim([np.min(time[start:end]), np.max(time[start:end])])
    labels = ('Plant', 'Grey-box', 'Black-box', 'Hybrid')
    figure.legend(handles = legend_handles, labels = labels, 
                  loc = (0.07, 0.9), ncol=4)
    # Return the figure object.
    return [figure]

def plot_sub_gaps(*, num_samples, sub_gaps, colors, legends, 
                  figure_size=PAPER_FIGSIZE,
                  ylabel_xcoordinate=-0.11, 
                  left_label_frac=0.15):
    """ Plot the suboptimality gaps. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                  sharex=True, 
                                  figsize=figure_size, 
                                  gridspec_kw=dict(left=left_label_frac))
    ylabel = r'$\% \ $ Suboptimality Gap'
    xlabel = 'Hours of training samples'
    num_samples = num_samples/60
    for (sub_gap, color) in zip(sub_gaps, colors):
        # Plot the corresponding data.
        axes.plot(num_samples, sub_gap, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    axes.set_xlim([np.min(num_samples), np.max(num_samples)])
    # Return the figure object.
    return [figure]

def plot_val_metrics(*, num_samples, val_metrics, colors, legends, 
                     figure_size=PAPER_FIGSIZE,
                     ylabel_xcoordinate=-0.11, 
                     left_label_frac=0.15):
    """ Plot validation metric on open loop data. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size, 
                                    gridspec_kw=dict(left=left_label_frac))
    xlabel = 'Hours of training samples'
    ylabel = 'MSE'
    num_samples = num_samples/60
    for (val_metric, color) in zip(val_metrics, colors):
        # Plot the corresponding data.
        axes.semilogy(num_samples, val_metric, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    axes.set_xlim([np.min(num_samples), np.max(num_samples)])
    figure.suptitle('Mean squared error (MSE) - Validation data', 
                    x=0.52, y=0.92)
    # Return the figure object.
    return [figure]

def plot_cost_mse_curve(*, us, cost_mses,
                         colors, legends, ylim,
                         figure_size=PAPER_FIGSIZE,
                         ylabel_xcoordinate=-0.2,
                         left_label_frac=0.21):
    """ Plot the profit curves. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                                  sharex=True,
                                  figsize=figure_size,
                                  gridspec_kw=dict(left=left_label_frac))
    xlabel = r'$C_{Af} \ (\textnormal{mol/m}^3)$'
    ylabel = r'$e(C_{Af})$'
    for (cost_mse, color) in zip(cost_mses, colors):
        # Plot the corresponding data.
        axes.plot(us, cost_mse, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=True)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlim([np.min(us), np.max(us)])
    #axes.set_ylim(ylim)
    return [figure]

def main():
    """ Load the pickle file and plot. """
    tworeac_parameters = PickleTool.load(filename=
                                         "tworeac_parameters_nonlin.pickle", 
                                         type='read')
    (parameters, training_data, 
     greybox_validation_data) = (tworeac_parameters['parameters'], 
                                 tworeac_parameters['training_data'], 
                                 tworeac_parameters['greybox_validation_data'])
    tworeac_train = PickleTool.load(filename=
                                    "tworeac_train_nonlin.pickle", 
                                    type='read')
    num_samples = tworeac_train['num_samples']
    val_metrics = tworeac_train['val_metrics']
    val_predictions = tworeac_train['val_predictions']
    ssopt = PickleTool.load(filename="tworeac_ssopt_nonlin.pickle", 
                            type='read')
    sub_gaps = ssopt['sub_gaps']
    figures = []

    # Plot training data
    figures += plot_training_data(training_data=training_data[0], 
                                  plot_range=(0, 6*60))

    # Plot predictions on validation data.
    #val_predictions.pop(0)
    modelsim_datum = [greybox_validation_data] + val_predictions
    figures += plot_val_model_predictions(plantsim_data=training_data[-1],
                                    modelsim_datum=modelsim_datum,
                                    plot_range=(0, 6*60), 
                                    tsteps_steady=parameters['tsteps_steady'])

    # Plot cost curve.
    for costs in ssopt['costss']:
        figures += plot_profit_curve(us=ssopt['us'], 
                                    costs=costs,
                                    colors=['blue', 'green', 
                                            'dimgray', 'tomato'],
                                    legends=['Plant', 'Grey-box', 'Black-box', 
                                             'Hybrid'],
                                    ylabel_xcoordinate=-0.21,
                                    left_label_frac=0.21)

    cost_mse_curve_legends = [r'$N_s = 3 \ \textnormal{hours}$',
                              r'$N_s = 4 \ \textnormal{hours}$',
                              r'$N_s = 5 \ \textnormal{hours}$',
                              r'$N_s = 8 \ \textnormal{hours}$']
    for model_cost_mse in ssopt['cost_mses']:
        figures += plot_cost_mse_curve(us=ssopt['us'], 
                                       cost_mses=model_cost_mse,
                                       colors=['blue', 'green',
                                               'dimgray', 'tomato'],
                                       ylim = [0., 15.],
                                       legends=cost_mse_curve_legends)

    # PLot validation metrics.
    figures += plot_val_metrics(num_samples=num_samples, 
                                val_metrics=val_metrics, 
                                colors=['dimgray', 'tomato'], 
                                legends=['Black-box', 'Hybrid'])
    # Plot suboptimality gaps.
    figures += plot_sub_gaps(num_samples=num_samples, 
                             sub_gaps=sub_gaps, 
                             colors=['dimgray', 'tomato'], 
                             legends=['Black-box', 'Hybrid'])

    with PdfPages('tworeac_plots_nonlin.pdf', 
                  'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()