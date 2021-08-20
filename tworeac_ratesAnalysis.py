# [depends] %LIB%/plottingFuncs.py %LIB%/hybridId.py
# [depends] %LIB%/BlackBoxFuncs.py %LIB%/TwoReacHybridFuncs.py
# [depends] %LIB%/linNonlinMPC.py %LIB%/tworeacFuncs.py
# [depends] tworeac_parameters.pickle
# [makes] pickle

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plottingFuncs import TwoReacPlots, PAPER_FIGSIZE
from hybridId import PickleTool
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from TwoReacHybridFuncs import get_hybrid_pars, hybrid_fxup, hybrid_hx
from linNonlinMPC import c2dNonlin, getSSOptimum, getXsYsSscost
from tworeacFuncs import cost_yup, plant_ode

def getNNRatePredictions(fNWeights, xvals, xuyscales):
    """ Get the neural network rate predictions. """




    return

def main():
    """ Main function to be executed. """

    # Load parameters.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')

    # Load Black-Box data.
    tworeac_hybtrain = PickleTool.load(filename=
                                      "tworeac_hybtrain_dyndata.pickle",
                                      type='read')
    fNWeights = tworeac_hybtrain['trained_weights'][-1]
    xuyscales = tworeac_hybtrain['xuyscales']
    
    # First get the Ca, Cb, and Cc range to analyze.
    Ndata = 1000
    lb = np.array([0., 0., 0.])
    ub = np.array([1., 0.5, 0.3])
    xvals = (ub - lb)*np.random.rand(Ndata, 3) + lb

    # Get the NN predictions.
    

    # Save.
    PickleTool.save(data_object=tworeac_bbnntrain,
                    filename='tworeac_ratesAnalysis.pickle')

main()