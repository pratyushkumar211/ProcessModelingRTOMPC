# [depends] %LIB%/hybridId.py %LIB%/plottingFuncs.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_bbnntrain.pickle
# [depends] tworeac_hybtrain.pickle
# [depends] tworeac_ssopt.pickle
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridId import PickleTool
from plottingFuncs import PAPER_FIGSIZE, TwoReacPlots, plotAvgCosts
from plottingFuncs import get_plotting_array_list

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
    plant_pars = tworeac_parameters['plant_pars']
    
    # Load Black-Box data after training.
    tworeac_bbnntrain = PickleTool.load(filename=
                                      "tworeac_bbnntrain_dyndata.pickle",
                                      type='read')
    bbnn_predictions = tworeac_bbnntrain['val_predictions']

    # Load Hybrid data after training.
    tworeac_hybtrain = PickleTool.load(filename=
                                     "tworeac_hybtrain_dyndata.pickle",
                                     type='read')
    hyb_predictions = tworeac_hybtrain['val_predictions']

    # Load the steady state cost computations.
    tworeac_ssopt = PickleTool.load(filename="tworeac_ssopt.pickle",
                                     type='read')

    # List to store figures.
    figures = []

    # Plot training data.
    training_data = tworeac_parameters['training_data_dyn'][:5]
    for data in training_data:

        t, ulist, ylist, xlist = get_plotting_array_list(simdata_list = [data],
                                                     plot_range=(10, 6*60+10))
        figures += TwoReacPlots.plot_xudata(t=t, xlist=ylist, ulist=ulist,
                                            legend_names=None,
                                            legend_colors=['b'], 
                                            figure_size=PAPER_FIGSIZE, 
                                            ylabel_xcoordinate=-0.1, 
                                            title_loc=None)

    # Plot validation data.
    legend_names = ['Plant', 'Black-Box-NN', 'Hybrid']
    legend_colors = ['b', 'dimgrey', 'm']
    valdata_plant = tworeac_parameters['training_data_dyn'][-1]
    valdata_list = [valdata_plant]
    valdata_list += bbnn_predictions
    valdata_list += hyb_predictions
    t, ulist, ylist, xlist = get_plotting_array_list(simdata_list=
                                                     valdata_list[:1],
                                                     plot_range=(10, 6*60+10))
    (_, ulist_val, 
     ylist_val, xlist_val) = get_plotting_array_list(simdata_list=
                                                     valdata_list[1:],
                                                     plot_range=(0, 6*60))
    ulist += ulist_val
    ylist += ylist_val
    xlist += xlist_val
    figures += TwoReacPlots.plot_xudata(t=t, xlist=ylist, ulist=ulist,
                                        legend_names=legend_names,
                                        legend_colors=legend_colors, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.1, 
                                        title_loc=(0.23, 0.9))

    # Plot validation metrics to show data requirements.
    #num_samples = tworeac_train['num_samples']
    #val_metrics = tworeac_train['val_metrics']
    #figures += plot_val_metrics(num_samples=num_samples,
    #                            val_metrics=val_metrics, 
    #                            colors=['dimgray', 'm'], 
    #                            legends=['Black-box', 'Hybrid'])

    # Steady state cost curves.
    us = tworeac_ssopt['us']
    sscosts = tworeac_ssopt['sscosts']
    legend_names = ['Plant', 'Black-Box-NN', 'Hybrid']
    legend_colors = ['b', 'dimgrey', 'm']
    figures += TwoReacPlots.plot_sscosts(us=us, sscosts=sscosts, 
                                        legend_colors=legend_colors, 
                                        legend_names=legend_names, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.12, 
                                        left_label_frac=0.15)

    # Load data for the economic MPC simulation.
    # tworeac_empc = PickleTool.load(filename="tworeac_empc.pickle", 
    #                                 type='read')
    # tworeac_rtompc_plant = PickleTool.load(filename=
    #                                "tworeac_rtompc_plant.pickle", 
    #                                 type='read')
    # tworeac_rtompc_hybrid = PickleTool.load(filename=
    #                                "tworeac_rtompc_hybrid.pickle", 
    #                                 type='read')
    # tworeac_rtompc_picnn = PickleTool.load(filename=
    #                                "tworeac_rtompc_picnn.pickle", 
    #                                 type='read')
    # clDataList = [tworeac_rtompc_plant['clData'], 
    #               tworeac_rtompc_hybrid['clData']]

    # # Load data for the economic MPC simulation.
    # tworeac_empc_twotier = PickleTool.load(filename=
    #                                 "tworeac_empc_twotier.pickle", 
    #                                 type='read')
    # clDataListTwoTier = tworeac_empc_twotier['clDataList']
    # stageCostListTwoTier = tworeac_empc_twotier['stageCostList']

    # Plot closed-loop simulation data.
    # legend_names = ['Plant', 'Hybrid', 'PICNN']
    # legend_colors = ['b', 'g', 'm', 'orange']
    # t, ulist, ylist, xlist = get_plotting_array_list(simdata_list = 
    #                                             clDataList,
    #                                     plot_range = (0, 2*24*60))
    # figures += TwoReacPlots.plot_xudata(t=t, xlist=ylist, ulist=ulist,
    #                                     legend_names=legend_names,
    #                                     legend_colors=legend_colors, 
    #                                     figure_size=PAPER_FIGSIZE, 
    #                                     ylabel_xcoordinate=-0.1, 
    #                                     title_loc=(0.32, 0.9))

    # # Plot empc pars.
    # econPars, _ = getEconDistPars()
    # figures += plot_cost_pars(t=t, cost_pars=econPars)

    # # Plot profit curve.
    # stageCostList = [tworeac_rtompc_plant['avgStageCosts'], 
    #                  tworeac_rtompc_hybrid['avgStageCosts']]
    # t = np.arange(0, len(stageCostList[0]), 1)/60
    # figures += plotAvgCosts(t=t, stageCostList=
    #                           stageCostList, 
    #                           legend_colors=legend_colors,
    #                           legend_names=legend_names)
    
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