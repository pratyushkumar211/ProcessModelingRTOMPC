# [depends] tworeac_parameters_nonlin.pickle tworeac_train_nonlin.pickle
# [depends] tworeac_empc_nonlin.pickle
# [depends] %LIB%/hybridid.py
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import PickleTool
from plotting_funcs import PAPER_FIGSIZE, TwoReacPlots
from hybridid import get_plotting_array_list

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
    ylabels = [r'$c_a$', r'$c_b$']
    for (axes, pari, ylabel) in zip(axes_list, range(num_pars), ylabels):
        # Plot the corresponding data.
        axes.plot(t, cost_pars[:len(t), pari])
        axes.set_ylabel(ylabel, rotation=False)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel(xlabel)
    axes.set_xlim([np.min(t), np.max(t)])
    return [figure]

def main():
    """ Load the pickle files and plot. """

    # Load parameters.
    tworeac_parameters = PickleTool.load(filename="tworeac_parameters.pickle",
                                         type='read')
    parameters = tworeac_parameters['parameters']
    training_data = tworeac_parameters['training_data']
    greybox_val_data = tworeac_parameters['greybox_val_data']
    
    # Load Black-box data after training.
    tworeac_bbtrain = PickleTool.load(filename="tworeac_bbtrain.pickle",
                                      type='read')
    bb_predictions = tworeac_bbtrain['val_predictions']

    # Load Koopman data after NN training.
    tworeac_kooptrain = PickleTool.load(filename="tworeac_kooptrain.pickle",
                                      type='read')
    koop_predictions = tworeac_kooptrain['val_predictions']


    # Load data after Koopman training.
    #tworeac_kooptrain = PickleTool.load(filename=
    #                                "tworeac_kooptrain_nonlin.pickle",
    #                                type='read')
    #koopval_predictions = tworeac_kooptrain['val_predictions']

    # Load data after Koopman training.
    #tworeac_encdeckooptrain = PickleTool.load(filename=
    #                                "tworeac_encdeckooptrain_nonlin.pickle",
    #                                type='read')
    #encdeckoopval_predictions = tworeac_encdeckooptrain['val_predictions']


    # Create a figures list.
    figures = []

    # Plot validation data.
    legend_names = ['Plant', 'Grey-box', 'Black-box', 'Koopman']
    legend_colors = ['b', 'g', 'dimgrey', 'm']
    valdata_list = [training_data[-1], greybox_val_data]
    valdata_list += bb_predictions
    valdata_list += koop_predictions
    #valdata_list += encdeckoopval_predictions
    t, ulist, ylist, xlist = get_plotting_array_list(simdata_list=
                                                     valdata_list[:2],
                                                     plot_range=(10, 24*60+10))
    (t, ulist_train, 
     ylist_train, xlist_train) = get_plotting_array_list(simdata_list=
                                                     valdata_list[2:],
                                                     plot_range=(0, 24*60))
    ulist += ulist_train
    ylist += ylist_train
    xlist += xlist_train
    figures += TwoReacPlots.plot_xudata(t=t, xlist=xlist, ulist=ulist,
                                        legend_names=legend_names,
                                        legend_colors=legend_colors, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.1, 
                                        title_loc=(0.25, 0.9))

    # Plot validation metrics to show data requirements.
    #num_samples = tworeac_train['num_samples']
    #val_metrics = tworeac_train['val_metrics']
    #figures += plot_val_metrics(num_samples=num_samples,
    #                            val_metrics=val_metrics, 
    #                            colors=['dimgray', 'm'], 
    #                            legends=['Black-box', 'Hybrid'])

    # Load data for the economic MPC simulation.
    # tworeac_empc = PickleTool.load(filename=
    #                                 "tworeac_empc_nonlin.pickle", 
    #                                 type='read')
    # cl_data_list = tworeac_empc['cl_data_list']
    # cost_pars = tworeac_empc['cost_pars']
    # avg_stage_costs=tworeac_empc['avg_stage_costs']
    # openloop_sols = tworeac_empc['openloop_sols']

    # # Plot first open-loop simulation.
    # legend_names = ['Plant', 'Grey-box', 'Koopman']
    # legend_colors = ['b', 'g', 'm']
    # udatum = [openloop_sols[0][0], openloop_sols[1][0], openloop_sols[2][0]]
    # xdatum = [openloop_sols[0][1], openloop_sols[1][1], openloop_sols[2][1]]
    # figures += plot_openloop_sols(t=t, udatum=udatum, xdatum=xdatum,
    #                               legend_names=legend_names,
    #                               legend_colors=legend_colors)

    # # Plot closed-loop simulation data.
    # t, ulist, ylist, xlist = get_plotting_array_list(simdata_list=
    #                                                  cl_data_list,
    #                                                  plot_range=(0, 8*60))
    # figures += plot_xudata(t=t, xlist=xlist, ulist=ulist,
    #                        legend_names=legend_names,
    #                        legend_colors=legend_colors)

    # # Plot empc pars.
    # figures += plot_cost_pars(t=t, cost_pars=cost_pars)

    # # Plot profit curve.
    # figures += plot_avg_profits(t=t,
    #                             avg_stage_costs=avg_stage_costs, 
    #                             legend_colors=legend_colors,
    #                             legend_names=legend_names)
    
    # Plot the RTO simulation data.
    #cl_data_list = tworeac_rto['cl_data_list']
    #t, ulist, ylist, xlist = get_plotting_array_list(simdata_list=
    #                                                 cl_data_list,
    #                                                 plot_range=(0, 8*60))
    #figures += plot_xudata(t=t, xlist=xlist, ulist=ulist,
    #                       legend_names=legend_names,
    #                       legend_colors=legend_colors)
    #figures += plot_avg_profits(t=t,
    #                        avg_stage_costs=tworeac_rto['avg_stage_costs'], 
    #                        legend_colors=legend_colors,
    #                        legend_names=legend_names)                     

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

    # Plot suboptimality gaps.
    #figures += plot_sub_gaps(num_samples=num_samples, 
    #                         sub_gaps=sub_gaps, 
    #                         colors=['dimgray', 'tomato'], 
    #                         legends=['Black-box', 'Hybrid'])

    with PdfPages('tworeac_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()