# [depends] %LIB%/hybridId.py %LIB%/plottingFuncs.py
# [depends] reac_parameters.pickle
# [depends] reac_bbnntrain.pickle
# [depends] reac_hybtrain.pickle
# [depends] reac_ssopt.pickle
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridId import PickleTool
from plottingFuncs import PAPER_FIGSIZE, ReacPlots, plotAvgCosts
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
                                       sharex=True, figsize=figure_size,
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
    # Figure list.
    return [figure]

def main():
    """ Load the pickle files and plot. """

    # Load parameters.
    reac_parameters = PickleTool.load(filename="reac_parameters.pickle",
                                         type='read')
    plant_pars = reac_parameters['plant_pars']
    
    # Load Black-Box data after training.
    reac_bbnntrain = PickleTool.load(filename=
                                      "reac_bbnntrain_dyndata.pickle",
                                      type='read')
    bbnn_predictions = reac_bbnntrain['val_predictions']

    # Load Hybrid data after training.
    reac_hybfullgbtrain = PickleTool.load(filename=
                                     "reac_hybfullgbtrain_dyndata.pickle",
                                     type='read')
    hybfullgb_predictions = reac_hybfullgbtrain['val_predictions']

    # Load Hybrid data after training.
    reac_hybpartialgbtrain = PickleTool.load(filename=
                                     "reac_hybpartialgbtrain_dyndata.pickle",
                                     type='read')
    hybpartialgb_predictions = reac_hybpartialgbtrain['val_predictions']


    # Load the steady state cost computations.
    reac_ssopt = PickleTool.load(filename="reac_ssopt.pickle",
                                     type='read')

    # Load the rate analysis computations.
    reac_fullgbRateAnalysis = PickleTool.load(filename=
                                     "reac_fullgbRateAnalysis.pickle",
                                     type='read')
    reac_partialgbRateAnalysis = PickleTool.load(filename=
                                     "reac_partialgbRateAnalysis.pickle",
                                     type='read')

    # List to store figures.
    figures = []

    # Plot training data.
    # training_data = reac_parameters['training_data_dyn'][:5]
    # for data in training_data:

    #     (t, ulist, xlist, 
    #      ylist, plist) = get_plotting_array_list(simdata_list = [data],
    #                                              plot_range=(10, 6*60+10))

    #     # xu data.
    #     figures += ReacPlots.plot_yxudata(t=t, ylist=ylist, 
    #                                         xlist=xlist, ulist=ulist,
    #                                         legend_names=None,
    #                                         legend_colors=['b'], 
    #                                         figure_size=PAPER_FIGSIZE, 
    #                                         ylabel_xcoordinate=-0.1, 
    #                                         title_loc=None)

        # yup data.
        # figures += ReacPlots.plot_yupdata(t=t, ylist=ylist, ulist=ulist,
        #                                      plist=plist,
        #                                     legend_names=None,
        #                                     legend_colors=['b'], 
        #                                     figure_size=PAPER_FIGSIZE, 
        #                                     ylabel_xcoordinate=-0.1, 
        #                                     title_loc=None)

    # Plot validation data.
    legend_names = ['Plant', 'Hybrid - 1', 'Hybrid - 2']
    legend_colors = ['b', 'm', 'g']
    valdata_plant = reac_parameters['training_data_dyn'][-1]
    valdata_list = [valdata_plant]
    #valdata_list += bbnn_predictions
    valdata_list += hybfullgb_predictions
    valdata_list += hybpartialgb_predictions
    t, ulist, xlist, ylist, plist = get_plotting_array_list(simdata_list=
                                                     valdata_list[:1],
                                                     plot_range=(10, 6*60+10))
    (_, ulist_val, 
     xlist_val, ylist_val, plist_val) = get_plotting_array_list(simdata_list=
                                                     valdata_list[1:],
                                                     plot_range=(0, 6*60))
    ulist += ulist_val
    ylist += ylist_val
    xlist += xlist_val
    figures += ReacPlots.plot_yxudata(t=t, ylist=ylist, 
                                      xlist=xlist, ulist=ulist,
                                        legend_names=legend_names,
                                        legend_colors=legend_colors, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.1, 
                                        title_loc=(0.25, 0.9))

    # Plot validation metrics to show data requirements.
    #num_samples = reac_train['num_samples']
    #val_metrics = reac_train['val_metrics']
    #figures += plot_val_metrics(num_samples=num_samples,
    #                            val_metrics=val_metrics, 
    #                            colors=['dimgray', 'm'], 
    #                            legends=['Black-box', 'Hybrid'])

    # Steady state Concentrations.
    us = reac_ssopt['us']
    Ny = reac_parameters['plant_pars']['Ny']
    xs_list = reac_ssopt['xs']
    Nss_data = xs_list[0].shape[0]
    #xs_list[1] = np.concatenate((xs_list[1][:, :Ny], 
    #                             np.tile(np.nan, (Nss_data, 1))), axis=-1)
    xs_list.pop(1)
    xs_list[2] = np.concatenate((xs_list[2][:, :Ny], 
                                 np.tile(np.nan, (Nss_data, 1))), axis=-1)
    legend_names = ['Plant', 'Hybrid-1', 'Hybrid-2']
    legend_colors = ['b', 'm', 'g']
    figures += ReacPlots.plot_xsvus(us=us, xs_list=xs_list, 
                                        legend_colors=legend_colors, 
                                        legend_names=legend_names, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.12, 
                                        title_loc=(0.25, 0.9))

    # Steady state cost curves.
    sscosts = reac_ssopt['sscosts']
    sscosts.pop(1)
    figures += ReacPlots.plot_sscosts(us=us, sscosts=sscosts, 
                                        legend_colors=legend_colors, 
                                        legend_names=legend_names, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.12, 
                                        left_label_frac=0.15)

    # Make the histograms.
    fullgbErrors = reac_fullgbRateAnalysis[0]
    partialgbErrors = reac_partialgbRateAnalysis[0]
    xlabels = ['$\dfrac{|r_1 - r_{1-NN}|}{r_1}$',
               '$\dfrac{|r_2 - r_{2-NN}|}{r_2}$',
               '$\dfrac{|r_3 - r_{3-NN}|}{r_3}$']
    xlims_list = [[0., 0.05], [0., 1.0], [0., 3.]]
    legend_names = ['Hybrid-1', 'Hybrid-2']
    for reaction, xlabel, xlims in zip(['r1', 'r2', 'r3'], 
                                       xlabels, xlims_list):

        # Loop over the errors.
        rErrors = [fullgbErrors[reaction], partialgbErrors[reaction]]
        figures += ReacPlots.plot_ErrorHistogram(rErrors=rErrors, 
                                                xlabel=xlabel, ylabel='Frequency',
                                                figure_size=PAPER_FIGSIZE, 
                                                left_frac=0.12, nBins=1500, 
                                                legend_names=legend_names,
                                                xlims=xlims)

    # One more plot for the -3r2 + r3 function.
    rErrors = [partialgbErrors['r2r3LumpedErrors']]
    xlabel = '$\dfrac{|-3r_2 + r_3 - (-3r_{2-NN}+r_{3-NN})|}{|-3r_2 + r_3|}$'
    xlims = [0., 0.1]
    legend_names = ['Hybrid - 2']
    figures += ReacPlots.plot_ErrorHistogram(rErrors=rErrors, 
                                            xlabel=xlabel, ylabel='Frequency',
                                            figure_size=PAPER_FIGSIZE, 
                                            left_frac=0.12, nBins=1500, 
                                            legend_names=legend_names,
                                            xlims=xlims)

    # # Make the 3D scatter plot.
    # errorsOnTrain = reac_rateAnalysis[2]
    # xlims = [0.02, 0.8]
    # ylims = [0.02, 0.6]
    # zlims = [0.02, 0.2]

    # # Loop over the errors.
    # figures += ReacPlots.plotDataSamples3D(ydata=
    #                                         errorsOnTrain['ysamples'], 
    #                                         figure_size=PAPER_FIGSIZE, 
    #                                         left_frac=0.12,
    #                                         xlims=xlims, ylims=ylims, 
    #                                         zlims=zlims, markersize=1.)

    # Load data for the economic MPC simulation.
    # reac_empc = PickleTool.load(filename="reac_empc.pickle", 
    #                                 type='read')
    # reac_rtompc_plant = PickleTool.load(filename=
    #                                "reac_rtompc_plant.pickle", 
    #                                 type='read')
    # reac_rtompc_hybrid = PickleTool.load(filename=
    #                                "reac_rtompc_hybrid.pickle", 
    #                                 type='read')
    # reac_rtompc_picnn = PickleTool.load(filename=
    #                                "reac_rtompc_picnn.pickle", 
    #                                 type='read')
    # clDataList = [reac_rtompc_plant['clData'], 
    #               reac_rtompc_hybrid['clData']]

    # # Load data for the economic MPC simulation.
    # reac_empc_twotier = PickleTool.load(filename=
    #                                 "reac_empc_twotier.pickle", 
    #                                 type='read')
    # clDataListTwoTier = reac_empc_twotier['clDataList']
    # stageCostListTwoTier = reac_empc_twotier['stageCostList']

    # Plot closed-loop simulation data.
    # legend_names = ['Plant', 'Hybrid', 'PICNN']
    # legend_colors = ['b', 'g', 'm', 'orange']
    # t, ulist, ylist, xlist = get_plotting_array_list(simdata_list = 
    #                                             clDataList,
    #                                     plot_range = (0, 2*24*60))
    # figures += ReacPlots.plot_xudata(t=t, xlist=ylist, ulist=ulist,
    #                                     legend_names=legend_names,
    #                                     legend_colors=legend_colors, 
    #                                     figure_size=PAPER_FIGSIZE, 
    #                                     ylabel_xcoordinate=-0.1, 
    #                                     title_loc=(0.32, 0.9))

    # # Plot empc pars.
    # econPars, _ = getEconDistPars()
    # figures += plot_cost_pars(t=t, cost_pars=econPars)

    # # Plot profit curve.
    # stageCostList = [reac_rtompc_plant['avgStageCosts'], 
    #                  reac_rtompc_hybrid['avgStageCosts']]
    # t = np.arange(0, len(stageCostList[0]), 1)/60
    # figures += plotAvgCosts(t=t, stageCostList=
    #                           stageCostList, 
    #                           legend_colors=legend_colors,
    #                           legend_names=legend_names)
    
    # Plot the RTO simulation data.
    #cl_data_list = reac_rto['cl_data_list']
    #t, ulist, ylist, xlist = get_plotting_array_list(simdata_list=
    #                                                 cl_data_list,
    #                                                 plot_range=(0, 8*60))
    #figures += plot_xudata(t=t, xlist=xlist, ulist=ulist,
    #                       legend_names=legend_names,
    #                       legend_colors=legend_colors)
    #figures += plot_avg_profits(t=t,
    #                        avg_stage_costs=reac_rto['avg_stage_costs'], 
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

    with PdfPages('reac_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()