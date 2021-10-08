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
from commonPlottingFuncs import PAPER_FIGSIZE, get_plotting_array_list


def plot_xudata(*, t, ylist, xlist, ulist, legend_names,
                    legend_colors, figure_size,
                    ylabel_xcoordinate, title_loc):
    """ Plot x and u data. 
        The states that are measured are plotted with measurement noise.
    """

    # Labels for the y axis.
    ylabels = [r'$C_A \ (\textnormal{mol/m}^3)$',
               r'$C_B \ (\textnormal{mol/m}^3)$',
               r'$C_C \ (\textnormal{mol/m}^3)$',
               r'$C_{Af} \ (\textnormal{mol/m}^3)$']

    # Number of rows.
    nrow = len(ylabels)

    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                  sharex=True, figsize=figure_size)

    # List to store handles for title labels.
    legend_handles = []

    # Loop through all the trajectories.
    for (y, x, u, color) in zip(ylist, xlist, ulist, legend_colors):
        
        # First plot the states.
        for row in range(nrow):
            
            # Plot depending on the row.
            if 0 <= row <= 1:
                axes[row].plot(t, y[:, row], color)
            if row == 2:
                axes[row].plot(t, x[:, 2], color)
            if row == 3:
                handle = axes[row].plot(t, u[:, 0], color)

            # Axes labels.
            axes[row].set_ylabel(ylabels[row])
            axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        
        # Store handles.
        legend_handles += handle

    # Overall asthetics of the x axis.
    axes[row].set_xlabel('Time (min)')
    axes[row].set_xlim([np.min(t), np.max(t)])

    # Create a figure title with legends. 
    if legend_names is not None:
        figure.legend(handles = legend_handles, labels = legend_names,
                      loc = title_loc,
                      ncol=len(legend_names)//2)

    # Return figure.
    return [figure]

def plot_xsvus(*, us, xs_list, legend_names, 
                  legend_colors, figure_size,
                  ylabel_xcoordinate, title_loc):
    """ Plot steady state xs and us. """
    
    # Y-axis labels.
    ylabels = [r'$C_{As} \ (\textnormal{mol/m}^3)$', 
               r'$C_{Bs} \ (\textnormal{mol/m}^3)$',
               r'$C_{Cs} \ (\textnormal{mol/m}^3)$']
    xlabel = r'$C_{Afs} \ (\textnormal{mol/m}^3)$'

    # Number of rows.
    nrow = len(ylabels)

    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
                                  sharex=True, figsize=figure_size)
    
    # List to store handles.
    legend_handles = []

    # Loop through all the steady-state profiles.
    for xs, color in zip(xs_list, legend_colors):
        
        # Plot the steady state.
        for row in range(nrow):

            # Plot the steady-state profiles.
            handle = axes[row].plot(us, xs[:, row], color)
            
            # Y-axis label.
            axes[row].set_ylabel(ylabels[row])
            axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        
        # Store handle.
        legend_handles += handle

    # X-axis label and limits.
    axes[row].set_xlabel(xlabel)
    axes[row].set_xlim([np.min(us), np.max(us)])

    # Name legends if provided. 
    if legend_names is not None:
        figure.legend(handles = legend_handles,
                        labels = legend_names,
                        loc = title_loc, ncol=len(legend_names)//2)

    # Return.
    return [figure]

def plot_sscosts(*, us, sscosts, legend_colors, 
                    legend_names, figure_size, 
                    ylabel_xcoordinate, left_label_frac):
    """ Plot the cost curves. """
    
    # Create figure and axes.
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                    sharex=True, 
                                    figsize=figure_size, 
                                    gridspec_kw=dict(left=left_label_frac))

    # X and Y labels.
    xlabel = r'$C_{Afs} \ (\textnormal{mol/m}^3)$'
    ylabel = r'Cost ($\$ $)'

    # Costs and legend colors.
    for (cost, color) in zip(sscosts, legend_colors):

        # Plot costs.
        axes.plot(us, cost, color)

    # Legends and labels.        
    axes.legend(legend_names)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    
    # Limits on x axis.
    axes.set_xlim([np.min(us), np.max(us)])
    
    # Return figure.
    return [figure]

def plot_rPercentErrors(*, xGrids, yGrids, zvals, rErrors, 
                            figure_size, xlabel, ylabel, rateTitle, 
                            ylabel_xcoordinate, left_frac, 
                            wspace, right_frac):
    """ Make the plots. """

    # Make plots.
    figures = []
    for (zval, xGrid, yGrid, rError) in zip(zvals, xGrids, yGrids, rErrors):
        
        # Create figures.
        figure, axes = plt.subplots(nrows=1, ncols=1, 
                                    sharex=True, figsize=figure_size,
                                    gridspec_kw=dict(left=left_frac, 
                                                        right=right_frac,
                                                        wspace=wspace))

        # Contour plot.
        mesh = axes.pcolormesh(xGrid, yGrid, rError, cmap='viridis')
        figure.colorbar(mesh, ax=axes)

        # X and Y labels.
        axes.set_ylabel(ylabel)
        axes.set_xlabel(xlabel)

        # Limits.
        axes.set_xlim([np.min(xGrid), np.max(xGrid)])
        axes.set_ylim([np.min(yGrid), np.max(yGrid)])

        # Title.
        title = rateTitle + str(zval) + ' (mol/m$^3$)' 
        axes.set_title(title)

        # Add into the figures list.
        figures += [figure]

    # Return the figure.
    return figures

def plot_ErrorHistogram(*, rErrors, figure_size, xlabel, ylabel, 
                            left_frac, nBins, legend_names,
                            xlims):
    """ Make the plots. """
    
    # Create figures.
    figure, axes = plt.subplots(nrows=1, ncols=1, 
                                figsize=figure_size,
                                gridspec_kw=dict(left=left_frac))

    # Contour plot.
    for rError in rErrors:
        axes.hist(rError, bins=nBins)

    # Legend. 
    if legend_names is not None:
        axes.legend(legend_names)

    # X and Y labels.
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    # X and Y limits.
    axes.set_xlim(xlims)

    # Return the figure.
    return [figure]

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
                                      "reac_bbnntrain.pickle",
                                      type='read')
    bbnn_predictions = reac_bbnntrain['val_predictions']

    # Load Hybrid data after training.
    reac_hybfullgbtrain = PickleTool.load(filename=
                                     "reac_hybfullgbtrain.pickle",
                                     type='read')
    hybfullgb_predictions = reac_hybfullgbtrain['val_predictions']

    # Load Hybrid data after training.
    reac_hybpartialgbtrain = PickleTool.load(filename=
                                     "reac_hybpartialgbtrain.pickle",
                                     type='read')
    hybpartialgb_predictions = reac_hybpartialgbtrain['val_predictions']


    # Load the steady state cost computations.
    reac_ssopt = PickleTool.load(filename="reac_ssopt_curve.pickle",
                                     type='read')

    # # Load the rate analysis computations.
    # reac_fullgbRateAnalysis = PickleTool.load(filename=
    #                                  "reac_fullgbRateAnalysis.pickle",
    #                                  type='read')
    # reac_partialgbRateAnalysis = PickleTool.load(filename=
    #                                  "reac_partialgbRateAnalysis.pickle",
    #                                  type='read')

    # List to store figures.
    figures = []

    # Plot training data.
    training_data = reac_parameters['training_data'][:5]
    for data in training_data:

        (t, ulist, xlist, 
         ylist, plist) = get_plotting_array_list(simdata_list = [data],
                                                 plot_range=(2, 6*60+10))

        # xu data.
        figures += plot_xudata(t=t, ylist=ylist, 
                                xlist=xlist, ulist=ulist,
                                legend_names=None,
                                legend_colors=['b'], 
                                figure_size=PAPER_FIGSIZE, 
                                ylabel_xcoordinate=-0.1, 
                                title_loc=None)

    # Plot validation data.
    legend_names = ['Plant', 'Black-Box-NN', 
                    'Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['b', 'dimgrey', 'm', 'g']
    valdata_plant = reac_parameters['training_data'][-1]
    valdata_list = [valdata_plant]
    valdata_list += [bbnn_predictions]
    valdata_list += [hybfullgb_predictions]
    valdata_list += [hybpartialgb_predictions]
    t, ulist, xlist, ylist, plist = get_plotting_array_list(simdata_list=
                                                     valdata_list[:1],
                                                     plot_range=(2, 6*60+10))
    (_, ulist_val, 
     xlist_val, ylist_val, plist_val) = get_plotting_array_list(simdata_list=
                                                     valdata_list[1:],
                                                     plot_range=(0, 6*60))
    ulist += ulist_val
    ylist += ylist_val
    xlist += xlist_val
    figures += plot_xudata(t=t, ylist=ylist, 
                            xlist=xlist, ulist=ulist,
                            legend_names=legend_names,
                            legend_colors=legend_colors, 
                            figure_size=PAPER_FIGSIZE, 
                            ylabel_xcoordinate=-0.1, 
                            title_loc=(0.25, 0.9))

    # Steady state Concentrations.
    us = reac_ssopt['us']
    Ny = reac_parameters['plant_pars']['Ny']
    xs_list = reac_ssopt['xs']
    Nss_data = xs_list[0].shape[0]
    xs_list[1] = np.concatenate((xs_list[1][:, :Ny], 
                                 np.tile(np.nan, (Nss_data, 1))), axis=-1)
    xs_list[3] = np.concatenate((xs_list[3][:, :Ny], 
                                 np.tile(np.nan, (Nss_data, 1))), axis=-1)
    legend_names = ['Plant', 'Black-Box-NN', 
                    'Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['b', 'dimgrey', 'm', 'g']
    figures += plot_xsvus(us=us, xs_list=xs_list, 
                          legend_colors=legend_colors, 
                          legend_names=legend_names, 
                          figure_size=PAPER_FIGSIZE, 
                          ylabel_xcoordinate=-0.12, 
                          title_loc=(0.25, 0.9))

    # # Steady state cost curves.
    # sscosts = reac_ssopt['sscosts']
    # sscosts.pop(1)
    # figures += ReacPlots.plot_sscosts(us=us, sscosts=sscosts, 
    #                                     legend_colors=legend_colors, 
    #                                     legend_names=legend_names, 
    #                                     figure_size=PAPER_FIGSIZE, 
    #                                     ylabel_xcoordinate=-0.12, 
    #                                     left_label_frac=0.15)

    # Make the histograms.
    # fullgbErrors = reac_fullgbRateAnalysis[0]
    # partialgbErrors = reac_partialgbRateAnalysis[0]
    # xlabels = ['$\dfrac{|r_1 - r_{1-NN}|}{r_1}$',
    #            '$\dfrac{|r_2 - r_{2-NN}|}{r_2}$']
    # xlims_list = [[0., 0.1], [0., 0.1]]
    # legend_names = ['Hybrid-1', 'Hybrid-2']
    # for reaction, xlabel, xlims in zip(['r1', 'r2'], 
    #                                    xlabels, xlims_list):

    #     # Loop over the errors.
    #     rErrors = [fullgbErrors[reaction], partialgbErrors[reaction]]
    #     figures += ReacPlots.plot_ErrorHistogram(rErrors=rErrors, 
    #                                             xlabel=xlabel, ylabel='Frequency',
    #                                             figure_size=PAPER_FIGSIZE, 
    #                                             left_frac=0.12, nBins=2000, 
    #                                             legend_names=legend_names,
    #                                             xlims=xlims)

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