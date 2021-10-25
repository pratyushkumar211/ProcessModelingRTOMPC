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
from plottingFuncs import PAPER_FIGSIZE, get_plotting_array_list
from plottingFuncs import plot_histogram

def plot_xudata(*, t, ylist, xlist, ulist, legend_names,
                    legend_colors, figure_size,
                    ylabel_xcoordinate, title_loc):
    """ Plot x and u data. 
        The states that are measured are plotted with measurement noise.
    """

    # Labels for the y axis.
    ylabels = [r'$c_A \ (\textnormal{mol/m}^3)$',
               r'$c_B \ (\textnormal{mol/m}^3)$',
               r'$c_C \ (\textnormal{mol/m}^3)$',
               r'$c_{Af} \ (\textnormal{mol/m}^3)$']

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
    ylabels = [r'$c_{As} \ (\textnormal{mol/m}^3)$', 
               r'$c_{Bs} \ (\textnormal{mol/m}^3)$',
               r'$c_{Cs} \ (\textnormal{mol/m}^3)$']
    xlabel = r'$c_{Afs} \ (\textnormal{mol/m}^3)$'

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
                    ylabel_xcoordinate):
    """ Plot the cost curves. """
    
    # Create figure and axes.
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                  sharex=True, figsize=figure_size)

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

def plot_r1Errors(*, r1CaRange, r1Errors, legend_colors,
                     legend_names, figure_size,
                     xlabel, ylabel, ylabel_xcoordinate, 
                     left_frac):
    """ Plot errors in the first reaction. """

    # Create figures.
    figure, axes = plt.subplots(nrows=1, ncols=1,
                                sharex=True, figsize=figure_size, 
                                gridspec_kw=dict(left=left_frac))
    
    # Make plots.
    for (r1Error, color) in zip(r1Errors, legend_colors):

        # Contour plot.
        handle = axes.semilogy(r1CaRange, r1Error, color=color)

    # X and Y labels.
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    # X-axis limits.
    axes.set_xlim([np.min(r1CaRange), np.max(r1CaRange)])

    # Plot legend names.
    if legend_names is not None:
        axes.legend(legend_names)

    # Return.
    return [figure]

def plot_fullGbR2Errors(*, r2XGrid, r2YGrid, r2Errors, 
                           figure_size, xlabel, ylabel, title, 
                           ylabel_xcoordinate, left_frac, 
                           wspace, right_frac, title_y):
    """ Plot errors in reaction rate 2. """
        
    # Create figure/axes.
    figure, axes = plt.subplots(nrows=1, ncols=1, 
                                sharex=True, figsize=figure_size,
                                gridspec_kw=dict(left=left_frac, 
                                                 right=right_frac,
                                                 wspace=wspace))

    # Contour plot.
    mesh = axes.pcolormesh(r2XGrid, r2YGrid, r2Errors, cmap='viridis')
    figure.colorbar(mesh, ax=axes)  

    # Labels.
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    axes.set_title(title, loc='center', y=title_y)

    # Limits.
    axes.set_xlim([np.min(r2XGrid), np.max(r2XGrid)])
    axes.set_ylim([np.min(r2YGrid), np.max(r2YGrid)])

    # Return the figure.
    return [figure]

# def plot_cost_pars(t, cost_pars,
#                    figure_size=PAPER_FIGSIZE, 
#                    ylabel_xcoordinate=-0.15):
#     """ Plot the economic MPC cost parameters. """

#     num_pars = cost_pars.shape[1]
#     (figure, axes_list) = plt.subplots(nrows=num_pars, ncols=1,
#                                        sharex=True, figsize=figure_size,
#                                        gridspec_kw=dict(left=0.18))
#     xlabel = 'Time (hr)'
#     ylabels = [r'$c_a$', r'$c_b$']
#     for (axes, pari, ylabel) in zip(axes_list, range(num_pars), ylabels):
#         # Plot the corresponding data.
#         axes.plot(t, cost_pars[:len(t), pari])
#         axes.set_ylabel(ylabel, rotation=False)
#         axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
#     axes.set_xlabel(xlabel)
#     axes.set_xlim([np.min(t), np.max(t)])
#     # Figure list.
#     return [figure]

def main():
    """ Load the pickle files and plot. """

    # Plant parameters.
    reac_parameters = PickleTool.load(filename="reac_parameters.pickle",
                                         type='read')
    
    # Load Black-Box data after training.
    reac_bbnntrain = PickleTool.load(filename=
                                      "reac_bbnntrain.pickle",
                                      type='read')

    # Load full hybrid model data after training.
    reac_hybfullgbtrain = PickleTool.load(filename=
                                     "reac_hybfullgbtrain.pickle",
                                     type='read')

    # Load partial hybrid model data after training.
    reac_hybpartialgbtrain = PickleTool.load(filename=
                                     "reac_hybpartialgbtrain.pickle",
                                     type='read')

    # Load the steady state computations.
    reac_ssopt = PickleTool.load(filename="reac_ssopt_curve.pickle",
                                     type='read')

    # Load the rate analysis computations.
    reac_rateanalysis = PickleTool.load(filename=
                                     "reac_rateanalysis.pickle",
                                     type='read')

    # Load the optimization analysis computations.
    reac_ssopt_optimizationanalysis = PickleTool.load(filename=
                                     "reac_ssopt_optimizationanalysis.pickle",
                                     type='read')

    # List to store figures.
    figures = []

    # Plot training data.
    training_data = reac_parameters['training_data_withnoise'][:5]
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

    # Get Black-Box and Hybrid model predictions.
    bbnn_predictions = reac_bbnntrain[1]['val_predictions']
    hybfullgb_predictions = reac_hybfullgbtrain[1]['val_predictions']
    hybpartialgb_predictions = reac_hybpartialgbtrain[1]['val_predictions']

    # Plot validation data.
    legend_names = ['Plant', 'Black-Box-NN',
                    'Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['b', 'dimgrey', 'm', 'g']
    valdata_plant = reac_parameters['training_data_nonoise'][-1]
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
    us = reac_ssopt[0]['us']
    Ny = reac_parameters['plant_pars']['Ny']
    xs_list = reac_ssopt[0]['xs']
    legend_names = ['Plant', 'Black-Box-NN', 
                    'Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['b', 'dimgrey', 'm', 'g']
    figures += plot_xsvus(us=us, xs_list=xs_list, 
                          legend_colors=legend_colors, 
                          legend_names=legend_names, 
                          figure_size=PAPER_FIGSIZE, 
                          ylabel_xcoordinate=-0.12, 
                          title_loc=(0.25, 0.9))

    # Steady state cost curves.
    sscosts = reac_ssopt[0]['sscosts']
    figures += plot_sscosts(us=us, sscosts=sscosts, 
                            legend_colors=legend_colors, 
                            legend_names=legend_names, 
                            figure_size=PAPER_FIGSIZE, 
                            ylabel_xcoordinate=-0.12)

    # Steady state cost curves. 
    sscosts = reac_ssopt[1]['sscosts']
    legend_names = ['Plant', 'Hybrid - FullGb']
    legend_colors = ['b', 'm']
    figures += plot_sscosts(us=us, sscosts=sscosts, 
                            legend_colors=legend_colors, 
                            legend_names=legend_names, 
                            figure_size=PAPER_FIGSIZE, 
                            ylabel_xcoordinate=-0.12)

    # Make error histogram.
    # Reaction - 1
    fGbErrors = reac_rateanalysis[1]['fGbErrors']
    pGbErrors = reac_rateanalysis[1]['pGbErrors']
    xlabel = r'$\dfrac{|\textnormal{Rate}-\textnormal{Rate}_{\textnormal{NN}}|}'
    xlabel += r'{\textnormal{Rate}}$, (\textnormal{Reaction-1})'
    ylabel = 'Frequency'
    xlims = [0., 0.05]
    ylims = [0, 80]
    legend_names = ['Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['b', 'm']
    rErrors = [fGbErrors['r1Errors'], pGbErrors['r1Errors']]
    binRange = 0, 0.05
    figures += plot_histogram(data_list=rErrors, legend_colors=legend_colors, 
                              legend_names=legend_names, 
                              figure_size=PAPER_FIGSIZE, xlabel=xlabel, 
                              ylabel=ylabel, nBins=1000, xlims=xlims, 
                              ylims=ylims)

    # Make error histogram.
    # Reaction - 2
    xlabel = r'$\dfrac{|\textnormal{Rate}-\textnormal{Rate}_{\textnormal{NN}}|}'
    xlabel += r'{\textnormal{Rate}}$, (\textnormal{Reaction-2})'
    xlims = [0., 0.3]
    ylims = [0, 50]
    rErrors = [fGbErrors['r2Errors'], pGbErrors['r2Errors']]
    figures += plot_histogram(data_list=rErrors, legend_colors=legend_colors, 
                              legend_names=legend_names, 
                              figure_size=PAPER_FIGSIZE, xlabel=xlabel, 
                              ylabel=ylabel, nBins=1000, xlims=xlims, 
                              ylims=ylims)

    # Plot errors in the state-space.
    # Reaction - 1.
    fGbErrorsInStateSpace = reac_rateanalysis[0]
    r1Errors = [fGbErrorsInStateSpace['r1Errors']]
    r1CaRange = fGbErrorsInStateSpace['r1CaRange']
    xlabel = r'$c_A \ (\textnormal{mol/m}^3)$'
    ylabel = r'$\dfrac{|\textnormal{Rate}-\textnormal{Rate}_{\textnormal{NN}}|}'
    ylabel += r'{\textnormal{Rate}}$, (\textnormal{Reaction-1})'
    legend_colors = ['k']
    figures += plot_r1Errors(r1CaRange=r1CaRange,
                             r1Errors=r1Errors,
                             legend_colors=legend_colors,
                             legend_names=None,
                             figure_size=PAPER_FIGSIZE,
                             xlabel=xlabel, ylabel=ylabel,
                             ylabel_xcoordinate=-0.1, 
                             left_frac=0.15)

    # Plot errors in the state-space. 
    # Reaction -2.
    fGbErrorsInStateSpace = reac_rateanalysis[0]
    r2XGrid = fGbErrorsInStateSpace['r2XGrid']
    r2YGrid = fGbErrorsInStateSpace['r2YGrid']
    r2Errors = fGbErrorsInStateSpace['r2Errors']
    xlabel = r'$c_B \ (\textnormal{mol/m}^3)$'
    ylabel = r'$c_C \ (\textnormal{mol/m}^3)$'
    title = r'$\dfrac{|\textnormal{Rate}-\textnormal{Rate}_{\textnormal{NN}}|}'
    title += r'{\textnormal{Rate}}$, (\textnormal{Reaction-2})'
    figures += plot_fullGbR2Errors(r2XGrid=r2XGrid, 
                                   r2YGrid=r2YGrid, r2Errors=r2Errors, 
                                   figure_size=PAPER_FIGSIZE, 
                                   xlabel=xlabel, ylabel=ylabel, title=title, 
                                   ylabel_xcoordinate=-0.1, left_frac=0.12, 
                                   wspace=0.1, right_frac=0.95, title_y=1.02)

    # Plot the optimization analysis results.
    # Suboptimality in inputs, cost type 1.
    optAnalysis = reac_ssopt_optimizationanalysis[0]
    usGaps = optAnalysis['usGaps'][1:]
    xlabel = r'$\dfrac{|u_s - u_s^{*}|}{u_s^{*}}$'
    ylabel = 'Frequency'
    xlims = [0., 0.8]
    ylims = [0, 100]
    legend_names = ['Black-Box-NN', 'Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['dimgrey', 'm', 'g']
    figures += plot_histogram(data_list=usGaps, legend_colors=legend_colors, 
                              legend_names=legend_names, 
                              figure_size=PAPER_FIGSIZE, xlabel=xlabel, 
                              ylabel=ylabel, nBins=200, xlims=xlims,     
                              ylims=ylims)

    # Plot the optimization analysis results.
    # Suboptimality in cost, cost type 1.
    subGaps = optAnalysis['subGaps'][1:]
    xlabel = r'$\dfrac{|V_s - V_s^{*}|}{V_s^{*}}$'
    xlims = [0., 0.2]
    ylims = [0, 100]
    legend_names = ['Black-Box-NN', 'Hybrid - FullGb', 'Hybrid - PartialGb']
    legend_colors = ['dimgrey', 'm', 'g']
    figures += plot_histogram(data_list=subGaps, legend_colors=legend_colors, 
                              legend_names=legend_names, 
                              figure_size=PAPER_FIGSIZE, xlabel=xlabel, 
                              ylabel=ylabel, nBins=200, xlims=xlims, 
                              ylims=ylims)

    # Suboptimality in inputs, cost type 2.
    optAnalysis = reac_ssopt_optimizationanalysis[1]
    usGaps = optAnalysis['usGaps'][1:]
    xlabel = r'$\dfrac{|u_s - u_s^{*}|}{u_s^{*}}$'
    xlims = [0., 0.2]
    ylims = [0, 100]
    legend_names = ['Hybrid - FullGb']
    legend_colors = ['m']
    figures += plot_histogram(data_list=usGaps, legend_colors=legend_colors, 
                              legend_names=legend_names, 
                              figure_size=PAPER_FIGSIZE, xlabel=xlabel, 
                              ylabel=ylabel, nBins=200, xlims=xlims,     
                              ylims=ylims)

    # Suboptimality in cost, cost type 2.
    subGaps = optAnalysis['subGaps'][1:]
    xlabel = r'$\dfrac{|V_s - V_s^{*}|}{V_s^{*}}$'
    xlims = [0., 0.015]
    ylims = [0, 100]
    legend_names = ['Hybrid - FullGb']
    legend_colors = ['m']
    figures += plot_histogram(data_list=subGaps, legend_colors=legend_colors, 
                              legend_names=legend_names, 
                              figure_size=PAPER_FIGSIZE, xlabel=xlabel, 
                              ylabel=ylabel, nBins=200, xlims=xlims,
                              ylims=ylims)

    with PdfPages('reac_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()