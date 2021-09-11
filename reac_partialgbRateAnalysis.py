# [depends] reac_parameters.pickle
# [depends] reac_hybtrain.pickle
import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hybridId import PickleTool
from BlackBoxFuncs import fnn

def getRateErrorsOnTrainingData(*, training_data_dyn, r1Weights, r2Weights, 
                                   r3Weights, xuyscales, k1, k2, k3, 
                                   Np, tthrow):
    """ Get the relative errors on the training data. """

    # Scale.
    xmean, xstd = xuyscales['xscale']
    ymean, ystd = xuyscales['yscale']
    Castd, Cbstd = ystd[0:1], ystd[1:2]
    umean, ustd = xuyscales['uscale']

    # Sizes. 
    Ny, Nu = len(ymean), len(umean)

    # Loop over all the collected data.
    r1Errors, r2Errors, r3Errors = [], [], []
    r2r3LumpedErrors = []
    y_list = []
    for data in training_data_dyn:

        # Get the number of time steps.
        Nt = data.t.shape[0]
        for t in range(tthrow, Nt):
            
            # State at the current time.
            xt = data.x[t, :]
            Ca, Cb, Cc = xt[0], xt[1], xt[2]
            
            # True rates.
            r1 = k1*Ca
            r2 = k2*(Cb**3)
            r3 = k3*(Cc)

            # NN rates.
            yt = (xt[:2] - ymean)/ystd
            Ca, Cb = yt[0:1], yt[1:2]
            xpseq = (data.x[t-Np:t, :2] - ymean)/ystd
            xpseq = xpseq.reshape(Np*Ny, )
            upseq = (data.u[t-Np:t, :] - umean)/ustd
            upseq = upseq.reshape(Np*Nu, )
            z = np.concatenate((xpseq, upseq))

            # Get the reaction rates.
            r1NN = fnn(Ca, r1Weights)*Castd
            r2NN = fnn(Cb, r2Weights)*Cbstd
            r3NN = fnn(z, r3Weights)*Cbstd
            
            # Get the errors.
            r1Errors += [np.abs(r1 - r1NN)/r1]
            r2Errors += [np.abs(r2 - r2NN)/r2]
            r3Errors += [np.abs(r3 - r3NN)/r3]
            r2r3LumpedErrors += [np.abs(-3*r2+r3-(-3*r2NN+r3NN))/
                                 np.abs(-3*r2+r3)]

        # Get the ylists. 
        y_list += [data.y]

    # Make numpy arrays.
    errorsOnTrain = dict(r1=np.array(r1Errors), r2=np.array(r2Errors), 
                         r3=np.array(r3Errors), 
                         r2r3LumpedErrors=np.array(r2r3LumpedErrors),
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
                                      'reac_hybpartialgbtrain_dyndata.pickle',
                                      type='read')
    
    # Get Weights and scales.
    Np = 2
    tthrow = 10
    xuyscales = reac_hybtrain['xuyscales']
    r1Weights = reac_hybtrain['trained_r1Weights'][-1]
    r2Weights = reac_hybtrain['trained_r2Weights'][-1]
    r3Weights = reac_hybtrain['trained_r3Weights'][-1]

    # Rate constants.
    k1 = reac_parameters['plant_pars']['k1']
    k2 = reac_parameters['plant_pars']['k2']
    k3 = reac_parameters['plant_pars']['k3']

    # Errors in the rates collected over training data.
    training_data_dyn = reac_parameters['training_data_dyn']
    errorsOnTrain = getRateErrorsOnTrainingData(training_data_dyn=
                                                training_data_dyn, 
                                                r1Weights=r1Weights,
                                                r2Weights=r2Weights,
                                                r3Weights=r3Weights, 
                                                xuyscales=xuyscales, 
                                                k1=k1, k2=k2, k3=k3, 
                                                Np=Np, tthrow=tthrow)

    # Create a dictionary to save.
    rateAnalysisData = [errorsOnTrain]

    # Make the plot.
    PickleTool.save(data_object=rateAnalysisData,
                    filename='reac_partialgbRateAnalysis.pickle')

main()