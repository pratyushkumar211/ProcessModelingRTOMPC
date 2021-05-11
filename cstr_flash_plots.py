# [depends] %LIB%/hybridid.py %LIB%/plotting_funcs.py
# [depends] cstr_flash_parameters.pickle cstr_flash_bbtrain.pickle
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import PickleTool 
from plotting_funcs import (PAPER_FIGSIZE, get_plotting_array_list, 
                            CstrFlashPlots, plotAvgProfits)

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
    ylabels = ['Energy Price ($\$$/kW)',
                'Raw Mat ($\$$/mol-A)',
                'Prod Price ($\$$/mol-B)']
    for (axes, pari, ylabel) in zip(axes_list, range(num_pars), ylabels):
        # Plot the corresponding data.
        cost_pars[:, 0] = 60*cost_pars[:, 0]
        axes.plot(t, cost_pars[:len(t), pari])
        axes.set_ylabel(ylabel)
        axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlabel(xlabel)
    axes.set_xlim([np.min(t), np.max(t)])
    return [figure]

def main():
    """ Load the pickle file and plot. """

    # Get the data for plotting.
    cstr_flash_parameters = PickleTool.load(filename=
                                            "cstr_flash_parameters.pickle",
                                            type='read')
    cstr_flash_bbnntrain = PickleTool.load(filename=
                                         "cstr_flash_bbnntrain.pickle",
                                         type='read')
    cstr_flash_hybtrain = PickleTool.load(filename=
                                         "cstr_flash_hybtrain.pickle",
                                         type='read')


    # Collect data to plot open-loop predictions.
    bbnn_val_predictions = cstr_flash_bbnntrain['val_predictions']
    hyb_val_predictions = cstr_flash_hybtrain['val_predictions']
    valdata_list = [cstr_flash_parameters['training_data'][-1]]
    valdata_list += bbnn_val_predictions
    valdata_list += hyb_val_predictions
    (t, ulist, ylist, xlist) = get_plotting_array_list(simdata_list=
                                                    valdata_list[:1],
                                                plot_range = (10, 12*60+120))
    (t, ulist_train, 
     ylist_train, xlist_train) = get_plotting_array_list(simdata_list=
                                                     valdata_list[1:],
                                                     plot_range=(0, 12*60))
    ulist += ulist_train
    ylist += ylist_train
    xlist += xlist_train
    legend_names = ['Plant', 'Black-box', 'Hybrid']
    legend_colors = ['b', 'dimgrey', 'm']
    figures = []
    figures += CstrFlashPlots.plot_data(t=t, ulist=ulist, 
                                ylist=ylist, xlist=xlist, 
                                figure_size=PAPER_FIGSIZE, 
                                u_ylabel_xcoordinate=-0.1, 
                                y_ylabel_xcoordinate=-0.1, 
                                x_ylabel_xcoordinate=-0.2, 
                                plot_ulabel=False,
                                legend_names=legend_names, 
                                legend_colors=legend_colors, 
                                title_loc=(0.25, 0.9))

    # Plot validation metrics to show data requirements.
    #num_samples = cstr_flash_train['num_samples']
    #val_metrics = cstr_flash_train['val_metrics']
    #figures += plot_val_metrics(num_samples=num_samples, 
    #                            val_metrics=val_metrics, 
    #                            colors=['dimgray', 'm'], 
    #                            legends=['Black-box', 'Hybrid'])
    
    # # Plot the closed-loop simulation.
    # legend_names = ['Plant-EMPC', 'Plant-RTO-MPC', 
    #                 'BlackBox-RTO-MPC']
    # legend_colors = ['b', 'orange', 'dimgrey']
    # clDataList = cstr_flash_empc['clDataList']
    # clDataList += cstr_flash_empc_twotier['clDataList']
    # (t, ulist, ylist, xlist) = get_plotting_array_list(simdata_list=
    #                                    clDataList,
    #                                    plot_range = (0, 24*60))
    # figures += CstrFlashPlots.plot_data(t=t, ulist=ulist, 
    #                             ylist=ylist, xlist=xlist, 
    #                             figure_size=PAPER_FIGSIZE, 
    #                             u_ylabel_xcoordinate=-0.1, 
    #                             y_ylabel_xcoordinate=-0.1, 
    #                             x_ylabel_xcoordinate=-0.2, 
    #                             plot_ulabel=True,
    #                             legend_names=legend_names, 
    #                             legend_colors=legend_colors, 
    #                             title_loc=(0.18, 0.9), 
    #                             plot_y=True)
    
    # # # Plot the empc costs.
    # # figures += plot_cost_pars(t=t, 
    # #                           cost_pars=cstr_flash_empc['cost_pars'][:24*60, :])

    # # Plot the plant profit in time.
    # stageCostList = cstr_flash_empc['stageCostList']
    # stageCostList += cstr_flash_empc_twotier['stageCostList']
    # figures += plotAvgProfits(t=t,
    #                    stageCostList=stageCostList, 
    #                    legend_colors=legend_colors,
    #                    legend_names=legend_names)

    # Save PDF.
    with PdfPages('cstr_flash_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()