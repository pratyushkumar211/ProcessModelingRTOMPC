# [depends] tworeac_parameters.pickle
# [depends] tworeac_hybtrain.pickle
import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hybridId import PickleTool
from BlackBoxFuncs import fnn

def getTrueR1R2(*, Ca, Cb, Cc, k1, k2f, k2b):
    """ Function to get the true reaction rates. """

    # Get r1.
    r1 = k1*Ca

    # Get r2. 
    r2 = k2f*(Cb**3) - k2b*Cc

    # Return.
    return r1, r2

def getFullGbR1R2(*, Ca, Cb, Cc, r1Weights, r2Weights, xuyscales):
    """ Function to get the rate laws for the full GB hybrid model. 
        Rates for this hybrid model are parametrized such that
        r1 = NN_r1(Ca), r2 = NN_r2(Cb, Cc)
    """

    # Get scaling.
    ymean, ystd = xuyscales['yscale']
    Camean, Cbmean = ystd[0:1], ystd[1:2]
    Castd, Cbstd = ystd[0:1], ystd[1:2]

    # Get r1.
    Ca = (Ca - Camean)/Castd
    r1 = fnn(Ca, r1Weights)*Castd 

    # Get r2.
    Cb = (Cb - Cbmean)/Cbstd
    Cc = (Cc - Cbmean)/Cbstd
    CbCc = np.concatenate((Cb, Cc))
    r2 = fnn(CbCc, r2Weights)*Cbstd 

    # Return.
    return r1, r2

def getPartialGbR1R2(*, Ca, Cb, z, r1Weights, r2Weights, xuyscales):
    """ Function to get the rate laws for the partial GB hybrid model. 
        Rates for this hybrid model are parametrized such that
        r1 = NN_r1(Ca), r2 = NN_r2(Cb, z)
    """

    # Get scaling.
    ymean, ystd = xuyscales['yscale']
    Camean, Cbmean = ystd[0:1], ystd[1:2]
    Castd, Cbstd = ystd[0:1], ystd[1:2]
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                             np.tile(umean, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                            np.tile(ustd, (Np, ))))

    # Get r1.
    Ca = (Ca - Camean)/Castd
    r1 = fnn(Ca, r1Weights)*Castd 

    # Get r2.
    Cb = (Cb - Cbmean)/Cbstd
    z = (z - zmean)/zstd
    r2Input = np.concatenate((Cb, z))
    r2 = fnn(r2Input, r2Weights)*Cbstd

    # Return.
    return r1, r2

def getFullGbRateErrorsInStateSpace(*, CaRange, CbRange, CcRange, 
                                     r1Weights, r2Weights, xuyscales, 
                                     k1, k2f, k2b):
    """ Get errors in the reaction rate laws of the 
        Full Grey-Box model in a chosen region of state-space. """


    # Return. 
    return 

def getFullGbRateErrorsOnGeneratedData(*, training_data, r1Weights,
                                          r2Weights, xuyscales, k1, k2f, k2b):
    """ Get errors in the reaction rate laws of the 
        full Grey-Box model on entire generated data. """ 

    # Return. 
    return 

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

    # Load parameters.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')
    tworeac_partialgbtrain = PickleTool.load(filename=
                                      'tworeac_partialgbtrain.pickle',
                                      type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                      'tworeac_hybtrain.pickle',
                                      type='read')
    
    # Neural network weights and scaling.
    fNWeights = tworeac_hybtrain['trained_weights'][-1]
    xuyscales = tworeac_hybtrain['xuyscales']
    xmean, xstd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']



    # Make the plot.
    PickleTool.save(data_object=rateAnalysisData,
                    filename='reac_rateanalysis.pickle')

main()