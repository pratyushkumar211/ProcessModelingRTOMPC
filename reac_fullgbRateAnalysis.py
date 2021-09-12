# [depends] reac_parameters.pickle
# [depends] reac_hybtrain.pickle
import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hybridId import PickleTool
from BlackBoxFuncs import fnn

def getRateErrorsOnTrainingData(*, training_data_dyn, r1Weights, r2Weights, 
                                   r3Weights, xuyscales, k1, k2f, k2b):
    """ Get the relative errors on the training data. """

    # Scale.
    xmean, xstd = xuyscales['xscale']
    ymean, ystd = xuyscales['yscale']
    Castd, Cbstd = ystd[0:1], ystd[1:2]
    umean, ustd = xuyscales['uscale']

    # Loop over all the collected data.
    r1Errors, r2Errors = [], []
    y_list = []
    for data in training_data_dyn:

        # Get the number of time steps.
        Nt = data.t.shape[0]
        for t in range(Nt):
            
            # State at the current time.
            xt = data.x[t, :]
            
            # True rates.
            r1 = k1*xt[0]
            r2 = k2f*(xt[1]**3) - k2b*(xt[2])

            # NN rates.
            xt = (xt - xmean)/xstd
            Ca, CbCc = xt[0:1], xt[1:3]

            # Get the reaction rates.
            r1NN = fnn(Ca, r1Weights)*Castd
            r2NN = fnn(CbCc, r2Weights)*Cbstd

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
    reac_parameters = PickleTool.load(filename=
                                         'reac_parameters.pickle',
                                         type='read')
    reac_hybtrain = PickleTool.load(filename=
                                      'reac_hybfullgbtrain_dyndata.pickle',
                                      type='read')
    
    # Get Weights and scales.
    xuyscales = reac_hybtrain['xuyscales']
    r1Weights = reac_hybtrain['trained_r1Weights'][-1]
    r2Weights = reac_hybtrain['trained_r2Weights'][-1]

    # Rate constants.
    k1 = reac_parameters['plant_pars']['k1']
    k2f = reac_parameters['plant_pars']['k2f']
    k2b = reac_parameters['plant_pars']['k2b']

    # Errors in the rates collected over training data.
    training_data_dyn = reac_parameters['training_data_dyn']
    errorsOnTrain = getRateErrorsOnTrainingData(training_data_dyn=
                                                training_data_dyn, 
                                                r1Weights=r1Weights,
                                                r2Weights=r2Weights,
                                                xuyscales=xuyscales,
                                                k1=k1, k2f=k2f, k2b=k2b)

    # Create a dictionary to save.
    rateAnalysisData = [errorsOnTrain]
    
    # Make the plot.
    PickleTool.save(data_object=rateAnalysisData,
                    filename='reac_fullgbRateAnalysis.pickle')

main()