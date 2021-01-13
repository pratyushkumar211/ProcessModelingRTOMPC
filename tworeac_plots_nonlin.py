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
from hybridid import PickleTool, PAPER_FIGSIZE, plot_profit_curve
from hybridid import get_plotting_array_list

labels = [r'$C_A \ (\textnormal{mol/m}^3)$', 
          r'$C_B \ (\textnormal{mol/m}^3)$',
          r'$C_C \ (\textnormal{mol/m}^3)$',
          r'$C_{Af} \ (\textnormal{mol/m}^3)$']

def plot_xudata(*, t, xlist, ulist,
                   legend_names, legend_colors,
                   figure_size=PAPER_FIGSIZE,
                   ylabel_xcoordinate=-0.1):
    """ Plot the performance loss economic MPC parameters."""
    nrow = len(labels)
    (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                  sharex=True, figsize=figure_size,
                                  gridspec_kw=dict(wspace=0.4))
    legend_handles = []
    for (x, u, color) in zip(xlist, ulist, legend_colors):
        # First plot the states.
        for row in range(nrow-1):
            handle = axes[row].step(t, x[:, row], color)
            axes[row].set_ylabel(labels[row])
            axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        # Plot the input in the last row.
        row += 1
        axes[row].step(t, u[:, 0], color)
        axes[row].set_ylabel(labels[row])
        axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        axes[row].set_xlabel('Time (hr)')
        axes[row].set_xlim([np.min(t), np.max(t)])
        legend_handles += handle
    figure.legend(handles = legend_handles,
                  labels = legend_names,
                  loc = (0.25, 0.9), ncol=len(legend_names))
    # Return figure.
    return [figure]

#def plot_sub_gaps(*, num_samples, sub_gaps, colors, legends, 
#                  figure_size=PAPER_FIGSIZE,
#                  ylabel_xcoordinate=-0.11, 
#                  left_label_frac=0.15):
#    """ Plot the suboptimality gaps. """
#    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
#                                  sharex=True, 
#                                  figsize=figure_size, 
#                                  gridspec_kw=dict(left=left_label_frac))
#    ylabel = r'$\% \ $ Suboptimality Gap'
#    xlabel = 'Hours of training samples'
#    num_samples = num_samples/60
#    for (sub_gap, color) in zip(sub_gaps, colors):
#        # Plot the corresponding data.
#        axes.plot(num_samples, sub_gap, color)
#    axes.legend(legends)
#    axes.set_xlabel(xlabel)
#    axes.set_ylabel(ylabel)
#    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
#    axes.set_xlim([np.min(num_samples), np.max(num_samples)])
#    # Return the figure object.
#    return [figure]

#def plot_val_metrics(*, num_samples, val_metrics, colors, legends, 
#                     figure_size=PAPER_FIGSIZE,
#                     ylabel_xcoordinate=-0.11, 
#                     left_label_frac=0.15):
#    """ Plot validation metric on open loop data. """
#    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
#                                        sharex=True, 
#                                        figsize=figure_size, 
#                                    gridspec_kw=dict(left=left_label_frac))
#    xlabel = 'Hours of training samples'
#    ylabel = 'MSE'
#    num_samples = num_samples/60
#    for (val_metric, color) in zip(val_metrics, colors):
#        # Plot the corresponding data.
#        axes.semilogy(num_samples, val_metric, color)
#    axes.legend(legends)
#    axes.set_xlabel(xlabel)
#    axes.set_ylabel(ylabel)
#    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
#    axes.set_xlim([np.min(num_samples), np.max(num_samples)])
#    figure.suptitle('Mean squared error (MSE) - Validation data', 
#                    x=0.52, y=0.92)
    # Return the figure object.
#    return [figure]

#def plot_cost_mse_curve(*, us, cost_mses,
#                         colors, legends, ylim,
#                         figure_size=PAPER_FIGSIZE,
#                         ylabel_xcoordinate=-0.2,
#                         left_label_frac=0.21):
#    """ Plot the profit curves. """
#    (figure, axes) = plt.subplots(nrows=1, ncols=1,
#                                  sharex=True,
#                                  figsize=figure_size,
#                                  gridspec_kw=dict(left=left_label_frac))
#    xlabel = r'$C_{Af} \ (\textnormal{mol/m}^3)$'
#    ylabel = r'$e(C_{Af})$'
#    for (cost_mse, color) in zip(cost_mses, colors):
        # Plot the corresponding data.
#        axes.plot(us, cost_mse, color)
#    axes.legend(legends)
#    axes.set_xlabel(xlabel)
#    axes.set_ylabel(ylabel, rotation=True)
#    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
#    axes.set_xlim([np.min(us), np.max(us)])
    #axes.set_ylim(ylim)
#    return [figure]

def main():
    """ Load the pickle file and plot. """
    tworeac_parameters = PickleTool.load(filename=
                                         "tworeac_parameters_nonlin.pickle",
                                         type='read')
    (parameters, training_data, 
     greybox_validation_data) = (tworeac_parameters['parameters'], 
                                 tworeac_parameters['training_data'], 
                                 tworeac_parameters['greybox_validation_data'])
    #tworeac_train = PickleTool.load(filename=
    #                                "tworeac_train_nonlin.pickle", 
    #                                type='read')
    #num_samples = tworeac_train['num_samples']
    #val_metrics = tworeac_train['val_metrics']
    #val_predictions = tworeac_train['val_predictions']
    #ssopt = PickleTool.load(filename="tworeac_ssopt_nonlin.pickle", 
    #                        type='read')
    #sub_gaps = ssopt['sub_gaps']

    # Create a figures list.
    figures = []

    # Plot validation data.
    legend_names = ['Plant', 'Grey-box']
    legend_colors = ['b', 'g']
    valdata_list = [training_data[-1], greybox_validation_data]
    t, ulist, ylist, xlist = get_plotting_array_list(simdata_list=valdata_list,
                                                     plot_range=(0, 6*60))
    figures += plot_xudata(t=t, xlist=xlist, ulist=ulist,
                           legend_names=legend_names,
                           legend_colors=legend_colors)

    # Plot predictions on validation data.
    #val_predictions.pop(0)
    #modelsim_datum = [greybox_validation_data] + val_predictions
    #figures += plot_val_model_predictions(plantsim_data=training_data[-1],
    #                                modelsim_datum=modelsim_datum,
    #                                plot_range=(0, 6*60), 
    #                                tsteps_steady=parameters['tsteps_steady'])

    # Plot cost curve.
    #for costs in ssopt['costss']:
    #    figures += plot_profit_curve(us=ssopt['us'], 
    #                                costs=costs,
    #                                colors=['blue', 'green', 
    #                                        'dimgray', 'tomato'],
    #                                legends=['Plant', 'Grey-box', 'Black-box', 
    #                                         'Hybrid'],
    #                                ylabel_xcoordinate=-0.21,
    #                                left_label_frac=0.21)

    #cost_mse_curve_legends = [r'$N_s = 3 \ \textnormal{hours}$',
    #                          r'$N_s = 4 \ \textnormal{hours}$',
    #                          r'$N_s = 5 \ \textnormal{hours}$',
    #                          r'$N_s = 8 \ \textnormal{hours}$']
    #for model_cost_mse in ssopt['cost_mses']:
    #    figures += plot_cost_mse_curve(us=ssopt['us'], 
    #                                   cost_mses=model_cost_mse,
    #                                   colors=['blue', 'green',
    #                                           'dimgray', 'tomato'],
    #                                   ylim = [0., 15.],
    #                                   legends=cost_mse_curve_legends)

    # PLot validation metrics.
    #figures += plot_val_metrics(num_samples=num_samples, 
    #                            val_metrics=val_metrics, 
    #                            colors=['dimgray', 'tomato'], 
    #                            legends=['Black-box', 'Hybrid'])
    # Plot suboptimality gaps.
    #figures += plot_sub_gaps(num_samples=num_samples, 
    #                         sub_gaps=sub_gaps, 
    #                         colors=['dimgray', 'tomato'], 
    #                         legends=['Black-box', 'Hybrid'])

    with PdfPages('tworeac_plots_nonlin.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()