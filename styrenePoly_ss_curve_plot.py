import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

def plot_xudata(*,  xs, us, figure_size,
                    ylabel_xcoordinate, xlabel):
    """ Plot x and u data. 
        The states that are measured are plotted with measurement noise.
    """

    # Labels for the x and y axis.
    ylabels = [[r'$c_{Is}$', r'$c_{Ms}$'],
               [r'$c_{Ss}$', r'$T_s$'],
               [r'$T_{cs}$', r'$\lambda_{0s}$'],
               [r'$\lambda_{1s}$', r'$\lambda_{2s}$']]

    # Create figure.
    nrow, ncol = 4, 2
    figure, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                sharex=True, figsize=figure_size, 
                                gridspec_kw=dict(left=0.18, right=0.95,
                                                 wspace=0.85, hspace=0.6))

    # Which x axis to take the semilogy. 
    ysemilog_ind = [2]

    # Counter to keep track of the state being plotted. 
    yind_plot = 0

    # Iterate over the rows and columns.
    for row, col in itertools.product(range(nrow), range(ncol)):
        
        # Plot and increment counter to keep track of which state to plot.
        if yind_plot in ysemilog_ind:
            axes[row, col].semilogy(us, xs[:, yind_plot])
        else:
            axes[row, col].plot(us, xs[:, yind_plot])
        yind_plot += 1

        axes[row, col].set_xscale('log')

        # Axes labels.
        axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
        axes[row, col].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
        #axes[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #axes[row, col].yaxis.set_locator_params(nbins=2)

        # Overall asthetics of the x axis.
        if row == 3:
            axes[row, col].set_xlabel(xlabel)
            axes[row, col].set_xlim([np.min(us), np.max(us)])

    # Return.
    return [figure]

def main():
    """ Load the pickle files and plot. """

    # Load the steady-state data.
    with open("styrenePoly_ss_curve.pickle", "rb") as stream:
        ssCurveData_list = pickle.load(stream)

    # List of xlabels. 
    xlabels = [r'$Q_{Is}$', r'$Q_{Ms}$',
               r'$Q_{Ss}$', r'$Q_{cs}$']

    # Loop over the ss curve data. 
    figures = []
    for usi, ssCurveData in enumerate(ssCurveData_list):

        figures += plot_xudata(xs=ssCurveData['xs'], 
                               us=ssCurveData['us'][:, usi], 
                               figure_size=(5, 5), 
                               ylabel_xcoordinate=-0.55, xlabel=xlabels[usi])

    # Loop through all the figures to make the plots.
    with PdfPages('styrenePoly_ss_curve_plot.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

# Execute main.
main()