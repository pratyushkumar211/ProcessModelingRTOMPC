# [depends] tworeac_parameters.pickle
# [depends] tworeac_hybtrain.pickle

import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hybridId import PickleTool
from BlackBoxFuncs import fnn

def doRateAnalysis(*, xrange, yrange, zval, fNWeights, xuyscales, k, reaction):
    """ Function to do rate analysis. 
        x: Ca, y:Cb, z:Cc
    """

    # Get a concatenated set of state values.
    Nx = xrange.shape[0]
    Ny = yrange.shape[0]

    # Get a meshgrid of X and Y values.
    xGrid, yGrid = np.meshgrid(xrange, yrange)

    # Scale.
    xmean, xstd = xuyscales['yscale']
    Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
    umean, ustd = xuyscales['uscale']

    # Get the NN outputs.
    r = np.tile(np.nan, (Ny, Nx))
    rNN = np.tile(np.nan, (Ny, Nx))
    rError = np.tile(np.nan, (Ny, Nx))
    for i, j in itertools.product(range(Ny), range(Nx)):

        # Compute the true reaction rate.
        if reaction == 'First':
            r[i, j] = k*xGrid[i, j]
        else:
            r[i, j] = k*(yGrid[i, j]**3)

        # Get the NN outputs.
        x = np.array([xGrid[i, j], yGrid[i, j], zval])
        x = (x - xmean)/xstd
        nnOutput = fnn(x, fNWeights)

        # Get the reaction rates.
        if reaction == 'First':
            rNN[i, j] = nnOutput[0:1]*Castd
        else:
            rNN[i, j] = nnOutput[1:2]*Ccstd

        # Get the error.
        rError[i, j] = np.abs(rNN[i, j] - r[i, j])/r[i, j]

    # Return the data.
    return xGrid, yGrid, r, rNN, rError

def main():
    """ Main function to be executed. """

    # Load parameters and training data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                      'tworeac_hybtrain_dyndata.pickle',
                                      type='read')

    # Neural network weights and scaling.
    fNWeights = tworeac_hybtrain['trained_weights'][-1]
    xuyscales = tworeac_hybtrain['xuyscales']
    xmean, xstd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']

    # Rate constant parameters.
    k1 = tworeac_parameters['plant_pars']['k1']
    k2 = tworeac_parameters['plant_pars']['k2']

    # Get the neural network reaction rates.
    CcVals = [0.07, 0.09, 0.11, 0.13]
    CaRange = np.arange(0.25, 0.60, 1e-2)
    CbRange = np.arange(0.18, 0.28, 1e-2)

    # Loop over concentration of C values.
    r1, r1NN, r1Errors = [], [], []
    for CcVal in CcVals:
        
        # Do analysis.
        (xGrid, yGrid, 
         r, rNN, rErrors) = doRateAnalysis(xrange=CaRange, yrange=CbRange, 
                                           zval=CcVal, fNWeights=fNWeights, 
                                           xuyscales=xuyscales, k=k1, 
                                           reaction='First')

        # Store data in lists.
        r1 += [r]
        r1NN += [rNN]
        r1Errors += [rErrors]       

    # r1 Analysis data.
    r1Data = dict(r=r1, rNN=r1NN, rErrors=r1Errors, 
                  xGrid=xGrid, yGrid=yGrid, CcVals=CcVals)

    # Create a dictionary to save.
    rateAnalysisData = [r1Data]

    # Make the plot.
    PickleTool.save(data_object=rateAnalysisData,
                    filename='tworeac_rateAnalysis.pickle')

main()