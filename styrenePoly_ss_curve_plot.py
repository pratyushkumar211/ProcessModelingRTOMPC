# [depends] %LIB%/hybridId.py %LIB%/plottingFuncs.py
# [depends] reac_parameters.pickle reac_bbnntrain.pickle
# [depends] reac_hybfullgbtrain.pickle reac_hybpartialgbtrain.pickle
# [depends] reac_rateanalysis.pickle reac_ssopt_curve.pickle
# [depends] reac_ssopt_optimizationanalysis.pickle
""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridId import PickleTool
from plottingFuncs import PRESENTATION_FIGSIZE, get_plotting_array_list
from plottingFuncs import plot_histogram

def plot_xudata(*, t, ylist, xlist, ulist, legend_names,
                    legend_colors, figure_size,
                    ylabel_xcoordinate, title_loc):
    """ Plot x and u data. 
        The states that are measured are plotted with measurement noise.
    """

    # Labels for the x and y axis.
    ylabels = [[r'$c_I$', r'$c_M$'],
               [r'$c_S$', r'$T$'],
               [r'$T_c$', r'$\lambda_0$'],
               [r'$\lambda_1$', r'$\lambda_2$']]

    # Create figure.
    nrow, ncol = 4, 2
    figure, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                  sharex=True, figsize=figure_size)

    # List to store handles for title labels.
    legend_handles = []

    # Loop through all the trajectories.
    for (y, x, u, color) in zip(ylist, xlist, ulist, legend_colors):
        
        # First plot the states.
        for row, col in itertools.product(range(nrow), range(ncol)):
            
            # Special case to plot the unmeasured states.  
            if row == 0 and col == 0:
                axes[row, col].plot(t, x[:, 0], color)
            elif row == 1 and col == 0:
                axes[row, col].plot(t, x[:, 2], color)
            else:
                axes[row, col].plot(t, y[:, yind_plot], color)
                yind_plot += 1

            # Axes labels.
            axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
            axes[row, col].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        
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

    # Return.
    return [figure]

def main():
    """ Load the pickle files and plot. """

    # Plant parameters.
    styrenePoly_parameters = PickleTool.load(filename=
                                             "styrenePoly_parameters.pickle",
                                             type='read')
    
    # List to store figures.
    figures = []

    # Plot training data.
    training_data = styrenePoly_parameters['training_data_withnoise'][:5]
    for data in training_data:

        (t, ulist, xlist, 
         ylist, plist) = get_plotting_array_list(simdata_list = [data],
                                                 plot_range=(2, 6*60+10))

        # xu data.
        figures += plot_xudata(t=t, ylist=ylist, 
                                xlist=xlist, ulist=ulist,
                                legend_names=None,
                                legend_colors=['b'], 
                                figure_size=PRESENTATION_FIGSIZE, 
                                ylabel_xcoordinate=-0.1, 
                                title_loc=None)

    # Get Black-Box and Hybrid model predictions.
    bbnn_predictions = reac_bbnntrain[1]['val_predictions']
    hybfullgb_predictions = reac_hybfullgbtrain[1]['val_predictions']
    hybpartialgb_predictions = reac_hybpartialgbtrain[1]['val_predictions']

    # Plot validation data.
    legend_names = ['Plant']
    legend_colors = ['b']
    valdata_plant = reac_parameters['training_data_withnoise'][-1]
    valdata_list = [valdata_plant]
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
                            figure_size=PRESENTATION_FIGSIZE, 
                            ylabel_xcoordinate=-0.12, 
                            title_loc=(0.17, 0.9))

    # Loop through all the figures to make the plots.
    with PdfPages('styrenePoly_plots.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()