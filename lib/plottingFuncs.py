import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import itertools

PRESENTATION_FIGSIZE = (6, 6)
PAPER_FIGSIZE = (5, 5)

def get_plotting_arrays(simdata, plot_range):
    """ Get data and return for plotting. """
    start, end = plot_range
    u = simdata.u[start:end, :]
    x = simdata.x[start:end, :]
    y = simdata.y[start:end, :]
    p = None#simdata.p[start:end, :]
    t = simdata.t[start:end]
    # Return u, x, y, p, t.
    return (u, x, y, p, t)

def get_plotting_array_list(*, simdata_list, plot_range):
    """ Get all data as lists. """
    ulist, xlist, ylist, plist = [], [], [], []
    for simdata in simdata_list:
        u, x, y, p, t = get_plotting_arrays(simdata, plot_range)
        ulist += [u]
        xlist += [x]
        ylist += [y]
        plist += [p]
    # Return lists.
    return (t, ulist, xlist, ylist, plist)

def plot_histogram(*, data_list, legend_colors, legend_names,
                      xlabel, ylabel, nBins, xlims, ylims, figure_size):
    """ Make a histogram. """
    
    # Create figures.
    figure, axes = plt.subplots(nrows=1, ncols=1, 
                                figsize=figure_size)

    # Loop over the errors.
    for data, color in zip(data_list, legend_colors):

        # Make the histogram.
        axes.hist(data, bins=nBins, range=xlims, color=color)

    # Legend. 
    if legend_names is not None:
        axes.legend(legend_names)

    # X and Y labels.
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    # X limits.
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)

    # Return the figure.
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

# class CstrFlashPlots:

#     ylabels = [r'$H_r \ (\textnormal{m})$',
#                r'$C_{Br} \ (\textnormal{mol/m}^3)$', 
#                r'$T_r \ (K)$',
#                r'$H_b \ (\textnormal{m})$',
#                r'$C_{Bb} \ (\textnormal{mol/m}^3)$',
#                r'$T_b \ (K)$']

#     xlabels = [r'$H_r \ (\textnormal{m})$',
#                r'$C_{Ar} \ (\textnormal{mol/m}^3)$', 
#                r'$C_{Br} \ (\textnormal{mol/m}^3)$',
#                r'$C_{Cr} \ (\textnormal{mol/m}^3)$',
#                r'$T_r \ (K)$',
#                r'$H_b \ (\textnormal{m})$',
#                r'$C_{Ab} \ (\textnormal{mol/m}^3)$',
#                r'$C_{Bb} \ (\textnormal{mol/m}^3)$',
#                r'$C_{Cb} \ (\textnormal{mol/m}^3)$',
#                r'$T_b \ (K)$']

#     ulabels = [r'$F \ (\textnormal{m}^3/\textnormal{min})$',
#                r'$D \ (\textnormal{m}^3/\textnormal{min})$']

#     def plot_inputs(t, ulist, figure_size, ylabel_xcoordinate, 
#                     plot_ulabel, legend_names, legend_colors, 
#                     title_loc):
#         """ Plot the training input data. """
#         nrow = len(CstrFlashPlots.ulabels)
#         (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
#                                       sharex=True, figsize=figure_size,
#                                       gridspec_kw=dict(wspace=0.5))
#         legend_handles = []
#         for (u, color) in zip(ulist, legend_colors):
#             for row in range(nrow):
#                 handle = axes[row].step(t, u[:, row], color, where='post')
#                 axes[row].set_ylabel(CstrFlashPlots.ulabels[row])
#                 axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
#             axes[row].set_xlabel('Time (hr)')
#             axes[row].set_xlim([np.min(t), np.max(t)])
#             legend_handles += handle
#         if plot_ulabel:
#             figure.legend(handles = legend_handles,
#                         labels = legend_names,
#                         loc = title_loc, ncol=len(legend_names))
#         return [figure]

#     def plot_outputs(t, ylist, figure_size, ylabel_xcoordinate,
#                      legend_names, legend_colors, title_loc):
#         """ Plot the training input data."""
#         nrow = len(CstrFlashPlots.ylabels)
#         (figure, axes) = plt.subplots(nrows=nrow, ncols=1,
#                                     sharex=True, figsize=figure_size,
#                                     gridspec_kw=dict(wspace=0.5))
#         legend_handles = []
#         for (y, color) in zip(ylist, legend_colors):
#             for row in range(nrow):
#                 handle = axes[row].plot(t, y[:, row], color)
#                 axes[row].set_ylabel(CstrFlashPlots.ylabels[row])
#                 axes[row].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
#             axes[row].set_xlabel('Time (hr)')
#             axes[row].set_xlim([np.min(t), np.max(t)])
#             legend_handles += handle
        
#         if legend_names is not None:
#             figure.legend(handles = legend_handles,
#                         labels = legend_names,
#                         loc = title_loc, ncol=len(legend_names))
#         return [figure]

#     def plot_states(t, xlist, figure_size, ylabel_xcoordinate,
#                     legend_names, legend_colors, title_loc):
#         """ Plot the training outputs. """
#         nrow, ncol = 5, 2
#         (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
#                                       sharex=True, figsize=figure_size,
#                                       gridspec_kw=dict(wspace=0.5))
#         legend_handles = []
#         for (x, color) in zip(xlist, legend_colors):
#             state_index = 0
#             for row, col in itertools.product(range(nrow), range(ncol)):
#                 handle = axes[row, col].plot(t, x[:, state_index], color)
#                 axes[row, col].set_ylabel(CstrFlashPlots.xlabels[state_index])
#                 axes[row, col].get_yaxis().set_label_coords(ylabel_xcoordinate, 
#                                                             0.5)
#                 if row == nrow - 1:
#                     axes[row, col].set_xlabel('Time (hr)')
#                     axes[row, col].set_xlim([np.min(t), np.max(t)])
#                 state_index += 1
#             legend_handles += handle
#         if legend_names is not None:
#             figure.legend(handles = legend_handles,
#                         labels = legend_names,
#                         loc = title_loc, ncol=len(legend_names))
#         return [figure]

#     @staticmethod
#     def plot_data(*, t, ulist, ylist, xlist,
#                      figure_size, u_ylabel_xcoordinate, 
#                      x_ylabel_xcoordinate, legend_names, 
#                      legend_colors, title_loc, y_ylabel_xcoordinate=None, 
#                      plot_y=True, plot_ulabel=True):
#         figures = []
#         figures += CstrFlashPlots.plot_inputs(t, ulist, figure_size, 
#                                u_ylabel_xcoordinate,
#                                plot_ulabel, legend_names, legend_colors, 
#                                title_loc)
#         if plot_y:
#             figures += CstrFlashPlots.plot_outputs(t, ylist, figure_size, 
#                                    y_ylabel_xcoordinate, legend_names, 
#                                    legend_colors, title_loc)
#         figures += CstrFlashPlots.plot_states(t, xlist, figure_size, 
#                                x_ylabel_xcoordinate, legend_names, 
#                                legend_colors, title_loc)

#         # Return the figure object.
#         return figures

#     @staticmethod
#     def plot_sscosts(*, us1, us2, sscost, figure_size):
#         """ Plot the profit curves. """

#         figure, axes = plt.subplots(nrows=1, ncols=1, 
#                                 subplot_kw=dict(projection="3d"))

#         xlabel = r'$F \ (\textnormal{m}^3/\textnormal{min})$'
#         ylabel = r'$D \ (\textnormal{m}^3/\textnormal{min})$'
#         zlabel = r'Cost ($\$ $)'

#         # Make the surface plot.
#         surf = axes.plot_surface(us1, us2, sscost, 
#                                     cmap=cm.viridis, 
#                                     shade=True)
#         #axes.view_init(10, -50)
        
#         figure.colorbar(surf)
#         figure.tight_layout()
#         axes.set_xlabel(xlabel)
#         axes.set_ylabel(ylabel)
#         axes.set_zlabel(zlabel, rotation=False)
#         return [figure]

def plotAvgCosts(*, t, stageCostList,
                      legend_colors, legend_names, 
                      figure_size=PAPER_FIGSIZE, 
                      ylabel_xcoordinate=-0.15):
    """ Plot the profit. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                                  sharex=True,
                                  figsize=figure_size,
                                  gridspec_kw=dict(left=0.15))
    xlabel = 'Time (hr)'
    ylabel = '$\Lambda_k$'
    for (cost, color) in zip(stageCostList, legend_colors):
        # Plot the corresponding data.
        axes.plot(t, cost, color)
    axes.legend(legend_names)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=True)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlim([np.min(t), np.max(t)])
    # Return.
    return [figure]

# def plot_val_metrics(*, num_samples, val_metrics, colors, legends, 
#                      figure_size=PAPER_FIGSIZE,
#                      ylabel_xcoordinate=-0.11, 
#                      left_label_frac=0.15):
#     """ Plot validation metric on open loop data. """
#     (figure, axes) = plt.subplots(nrows=1, ncols=1, 
#                                   sharex=True, 
#                                   figsize=figure_size, 
#                                   gridspec_kw=dict(left=left_label_frac))
#     xlabel = 'Hours of training samples'
#     ylabel = 'MSE'
#     num_samples = num_samples/60
#     for (val_metric, color) in zip(val_metrics, colors):
#         # Plot the corresponding data.
#         axes.semilogy(num_samples, val_metric, color)
#     axes.legend(legends)
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
#     axes.set_xlim([np.min(num_samples), np.max(num_samples)])
#     figure.suptitle('Mean squared error (MSE) - Validation data', 
#                     x=0.52, y=0.92)
#    # Return the figure object.
#     return [figure]