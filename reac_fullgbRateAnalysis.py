# [depends] tworeac_parameters.pickle
# [depends] tworeac_hybtrain.pickle

import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hybridId import PickleTool
from BlackBoxFuncs import fnn

def doRateAnalysis(*, xrange, yrange, zval, 
                      fNWeights, xuyscales, k, reaction):
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

def getRateErrorsOnTrainingData(*, training_data_dyn, 
                                   fNWeights, xuyscales, 
                                   k1, k2):
    """ Get the relative errors on the training data. """

    # Scale.
    xmean, xstd = xuyscales['yscale']
    Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
    umean, ustd = xuyscales['uscale']

    # Loop over all the collected data.
    r1Errors, r2Errors = [], []
    y_list = []
    for data in training_data_dyn:

        # Get the number of time steps.
        Nt = data.t.shape[0]
        for t in range(Nt):
            
            # State at the current time.
            xt = data.y[t, :]
            
            # True rates.
            r1 = k1*xt[0]
            r2 = k2*(xt[1]**3)

            # NN rates.
            xt = (xt - xmean)/xstd
            nnOutput = fnn(xt, fNWeights)

            # Get the reaction rates.
            r1NN = nnOutput[0:1]*Castd
            r2NN = nnOutput[1:2]*Ccstd

            # Get the errors.
            r1Errors += [np.abs(r1 - r1NN)/r1]
            r2Errors += [np.abs(r2 - r2NN)/r2]

        # Get the ylists. 
        y_list += [data.y]

    # Make numpy arrays.
    errorsOnTrain = dict(r1=np.array(r1Errors), r2=np.array(r2Errors), 
                         ysamples=np.concatenate(y_list, axis=0))

    # Return the training data.
    return errorsOnTrain

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
    CcVals = [0.09, 0.4]
    CaRanges = [np.arange(0.20, 0.70, 1e-2), np.arange(0.1, 0.80, 1e-2)]
    CbRanges = [np.arange(0.15, 0.28, 1e-2), np.arange(0.1, 0.35, 1e-2)]
    
    # Loop over concentration of C values.
    r1, r1NN, r1Errors = [], [], []
    xGrids, yGrids = [], []
    for CaRange, CbRange, CcVal in zip(CaRanges, CbRanges, CcVals):
        
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
        xGrids += [xGrid]
        yGrids += [yGrid]

    # r1 Analysis data.
    r1Data = dict(r=r1, rNN=r1NN, rErrors=r1Errors, 
                  xGrids=xGrids, yGrids=yGrids, CcVals=CcVals)

    # Ranges of Ca and Cb.
    CaRanges = [np.arange(0.20, 0.70, 1e-2), np.arange(0.20, 0.70, 1e-2)]
    CbRanges = [np.arange(0.15, 0.28, 1e-2), np.arange(0.15, 0.35, 1e-2)]

    # Loop over concentration of C values.
    r2, r2NN, r2Errors = [], [], []
    xGrids, yGrids = [], []
    for CaRange, CbRange, CcVal in zip(CaRanges, CbRanges, CcVals):
        
        # Do analysis.
        (xGrid, yGrid, 
         r, rNN, rErrors) = doRateAnalysis(xrange=CaRange, yrange=CbRange, 
                                           zval=CcVal, fNWeights=fNWeights, 
                                           xuyscales=xuyscales, k=k2, 
                                           reaction='Second')

        # Store data in lists.
        r2 += [r]
        r2NN += [rNN]
        r2Errors += [rErrors]
        xGrids += [xGrid]
        yGrids += [yGrid]

    # r2 Analysis data.
    r2Data = dict(r=r2, rNN=r2NN, rErrors=r2Errors, 
                  xGrids=xGrids, yGrids=yGrids, CcVals=CcVals)

    # Errors in the rates collected over training data.
    training_data_dyn = tworeac_parameters['training_data_dyn']
    errorsOnTrain = getRateErrorsOnTrainingData(training_data_dyn=
                                                training_data_dyn, 
                                                fNWeights=fNWeights, 
                                                xuyscales=xuyscales, 
                                                k1=k1, k2=k2)

    # Create a dictionary to save.
    rateAnalysisData = [r1Data, r2Data, errorsOnTrain]

    # Make the plot.
    PickleTool.save(data_object=rateAnalysisData,
                    filename='tworeac_rateAnalysis.pickle')

main()