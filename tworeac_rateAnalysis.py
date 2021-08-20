# [depends] tworeac_parameters.pickle
# [depends] tworeac_hybtrain.pickle

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plottingFuncs import TwoReacPlots, PAPER_FIGSIZE
from hybridId import PickleTool
from BlackBoxFuncs import fnn

def getRNN(Ca, Cb, Cc, fNWeights, xuyscales):
    """ Plot the reaction rates. """

    # Assert sizes.
    assert Ca.shape[0] == Cb.shape[0]
    assert Cb.shape[0] == Cc.shape[0]

    # Get a concatenated set of state values.
    Ndata = Ca.shape[0]
    xvals = np.concatenate((Ca[:, np.newaxis], 
                            Cb[:, np.newaxis], 
                            Cc[:, np.newaxis]), axis=1)

    # Scale.
    xmean, xstd = xuyscales['yscale']
    Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
    umean, ustd = xuyscales['uscale']
    xvals = (xvals - xmean)/xstd

    # Get the NN outputs.
    r1NN_list = []
    r2NN_list = []
    for samp in range(Ndata):

        # Get the NN outputs.
        x = xvals[samp, :]
        nnOutput = fnn(x, fNWeights)
        r1NN = nnOutput[0:1]*Castd
        r2NN = nnOutput[1:2]*Ccstd

        # Append to list.
        r1NN_list += [r1NN]
        r2NN_list += [r2NN]

    # Reaction rates.
    r1NN = np.array(r1NN_list)
    r2NN = np.array(r2NN_list)

    # Return the rates computed by NNs.
    return r1NN, r2NN

def makeReactionPlot(x, r, ylabel, title, rNN=None, lw=0.5):
    """ Make the plots. """

    # Make a plot.
    figure, axes = plt.subplots(nrows=1, ncols=1, linewidth=lw,
                                sharex=True, figsize=(6, 4), 
                                gridspec_kw=dict(left=0.1, wspace=0.2))

    # Make plots.
    axes.plot(x, r, 'b', linewidth=lw)
    if rNN is not None:
        axes.plot(x, rNN, 'g', linewidth=lw)
        axes.legend(['True Reaction Rate', 'NN Reaction Rate'])


    # Plot the control input.
    axes.set_ylabel(ylabel, rotation=False)
    axes.set_xlim([np.min(x), np.max(x)])
    axes.set_title(label = title)

    # Return the figure.
    return [figure]

def main():
    """ Main function to be executed. """

    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')
    k1 = tworeac_parameters['plant_pars']['k1']
    k2 = tworeac_parameters['plant_pars']['k2']

    # Load pickle files.
    tworeac_hybtrain = PickleTool.load(filename=
                                      'tworeac_hybtrain_dyndata_3-16-2.pickle',
                                      type='read')
    fNWeights = tworeac_hybtrain['trained_weights'][-1]
    xuyscales = tworeac_hybtrain['xuyscales']

    # Create a figure list.
    figures = []

    # Get the neural network reaction rates.
    Ca = np.arange(0., 0.9, 1e-2)
    Cb = np.tile(0.4, (Ca.shape[0], ))
    Cc = np.tile(0.3, (Ca.shape[0], ))
    r1 = k1*Ca
    r1NN, _ = getRNN(Ca, Cb, Cc, fNWeights, xuyscales)
    r1NN = r1NN.squeeze()
    figures += makeReactionPlot(Ca, r1, '$r$', '$r = k_1 C_A$', rNN=r1NN)

    # Get the neural network reaction rates.
    # Cb = np.arange(0., 0.3, 1e-2)
    # Ca = np.tile(1.0, (Cb.shape[0], ))
    # Cc = np.tile(0.3, (Cb.shape[0], ))
    # r2 = k2*(Cb**3)
    # _, r2NN = getRNN(Ca, Cb, Cc, fNWeights, xuyscales)
    # r2NN = r2NN.squeeze()
    # figures += makeReactionPlot(Cb, r2, '$r$', '$r = k_2 C^3_B$', rNN=r2NN)

    # Make the plot.
    with PdfPages('tworeac_rateAnalysis.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()