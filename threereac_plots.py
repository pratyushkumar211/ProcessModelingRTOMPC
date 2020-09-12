""" Script to plot the training data
    and grey-box + NN model predictions on validation data.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import matplotlib.pyplot as plt

def plot_training_data(*, training_data, plot_range,
                          figure_size=PRESENTATION_FIGSIZE,
                          ylabel_xcoordinate=-0.08, 
                          linewidth=0.8):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=5, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    ylabels = ['$T_{z}(^\circ$C)', '$\dot{Q}_c$ (kW)', '$i$']
    legend_colors = ['b', 'g']
    legend_handles = []
    for (cascadesim_data, 
         legend_color) in zip(cascadesim_datum, legend_colors):
        time = cascadesim_data.time/3600
        data_list = [(cascadesim_data.Tzone, cascadesim_data.Tzone_sp), 
                      cascadesim_data.upi, cascadesim_data.i]
        for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
            if ylabel == '$T_{z}(^\circ$C)':
                (Tzone, Tsp) = data
                legend_temp_handle = axes.plot(time, Tzone, legend_color)
                axes.plot(time, Tsp, '--' + legend_color)
                axes.plot(time, cascadesim_data.Tzone_lb, 'k')
                axes.plot(time, cascadesim_data.Tzone_ub, 'k')
            elif ylabel == '$\dot{Q}_c$ (kW)':
                axes.plot(time, data, legend_color)
                axes.plot(time, cascadesim_data.Qcool_lb, 'k')
                axes.plot(time, cascadesim_data.Qcool_ub, 'k')
            else:
                axes.plot(time, data, legend_color)
            axes.set_ylabel(ylabel, rotation=False)
            axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        legend_handles += legend_temp_handle 
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    figure.legend(handles = legend_handles,
                  labels = (r'$\alpha$ = 30',
                            r'$\alpha$ = 50'), 
                  loc = (0.32, 0.9), ncol=3)
    # Return the figure object.
    return [figure]

def plot_val_model_predictions(*, cascadesim_datum, plot_range,
                            figure_size=PRESENTATION_FIGSIZE,
                            ylabel_xcoordinate=-0.08, 
                            linewidth=0.8):
    """ Plot the performance loss economic MPC parameters."""
    (figure, axes_array) = plt.subplots(nrows=3, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size)
    ylabels = ['$T_{z}(^\circ$C)', '$\dot{Q}_c$ (kW)', '$i$']
    legend_colors = ['b', 'g']
    legend_handles = []
    for (cascadesim_data, 
         legend_color) in zip(cascadesim_datum, legend_colors):
        time = cascadesim_data.time/3600
        data_list = [(cascadesim_data.Tzone, cascadesim_data.Tzone_sp), 
                      cascadesim_data.upi, cascadesim_data.i]
        for (axes, data, ylabel) in zip(axes_array, data_list, ylabels):
            if ylabel == '$T_{z}(^\circ$C)':
                (Tzone, Tsp) = data
                legend_temp_handle = axes.plot(time, Tzone, legend_color)
                axes.plot(time, Tsp, '--' + legend_color)
                axes.plot(time, cascadesim_data.Tzone_lb, 'k')
                axes.plot(time, cascadesim_data.Tzone_ub, 'k')
            elif ylabel == '$\dot{Q}_c$ (kW)':
                axes.plot(time, data, legend_color)
                axes.plot(time, cascadesim_data.Qcool_lb, 'k')
                axes.plot(time, cascadesim_data.Qcool_ub, 'k')
            else:
                axes.plot(time, data, legend_color)
            axes.set_ylabel(ylabel, rotation=False)
            axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        legend_handles += legend_temp_handle 
    axes.set_xlabel('Time (hr)')
    axes.set_xlim([np.min(time), np.max(time)])
    figure.legend(handles = legend_handles,
                  labels = (r'$\alpha$ = 30',
                            r'$\alpha$ = 50'), 
                  loc = (0.32, 0.9), ncol=3)
    # Return the figure object.
    return [figure]