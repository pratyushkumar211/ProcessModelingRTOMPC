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
    Camean, Cbmean = ymean[0:1], ymean[1:2]
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

def getPartialGbR1R2(*, Ca, Cb, z, Np, r1Weights, r2Weights, xuyscales):
    """ Function to get the rate laws for the partial GB hybrid model. 
        Rates for this hybrid model are parametrized such that
        r1 = NN_r1(Ca), r2 = NN_r2(Cb, z)
    """

    # Get scaling.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    Camean, Cbmean = ymean[0:1], ymean[1:2]
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

    # Get an empty zero array. 
    zeroArray = np.zeros((1,))

    # Create a list to store errors in r1.  
    r1Errors = []

    # Get errors in r1.
    for i, Ca in enumerate(CaRange):

        # True rate. 
        r1, _ = getTrueR1R2(Ca=Ca, Cb=zeroArray, Cc=zeroArray, 
                            k1=k1, k2f=k2f, k2b=k2b)

        # NN rate.
        r1NN, _ = getFullGbR1R2(Ca=Ca, Cb=zeroArray, Cc=zeroArray, 
                                r1Weights=r1Weights, r2Weights=r2Weights, 
                                xuyscales=xuyscales)

        # Get the error.
        r1Errors += [np.abs(r1 - r1NN)/np.abs(r1)]

    # Get the errors in an array form. 
    r1Errors = np.array(r1Errors).squeeze()

    # Create empty arrays to store errors in r2.
    NCb = len(CbRange)
    NCc = len(CcRange)
    r2Errors = np.tile(np.nan, (NCc, NCb))
    for j, i in itertools.product(range(NCc), range(NCb)):

        # True rate.
        _, r2 = getTrueR1R2(Ca=zeroArray, Cb=CbRange[i], Cc=CcRange[j], 
                            k1=k1, k2f=k2f, k2b=k2b)

        # NN rate. 
        _, r2NN = getFullGbR1R2(Ca=zeroArray, Cb=CbRange[i], Cc=CcRange[j], 
                                r1Weights=r1Weights, r2Weights=r2Weights, 
                                xuyscales=xuyscales)

        # Get the error. 
        r2Errors[j, i] = np.abs(r2 - r2NN)/np.abs(r2)

    # Return. 
    return r1Errors, r2Errors

def getRateErrorsOnGeneratedData(*, training_data, Ntstart, Np, Ny, Nu, 
                                    fGbWeights, pGbWeights, yindices, 
                                    xuyscales, k1, k2f, k2b):
    """ Get errors in the reaction rate laws on the entire generated data. """ 

    # Create lists to store the errors.
    fGbErrors = dict(r1Errors=[], r2Errors = [])
    pGbErrors = dict(r1Errors=[], r2Errors = [])

    # Get r1 and r2 Weights. 
    fGbR1Weights, fGbR2Weights = fGbWeights
    pGbR1Weights, pGbR2Weights = pGbWeights

    # Loop over all the generate trjectories.
    for data in training_data:

        # Nuumber of time steps in the trajectory.
        Nt = data.t.shape[0]

        # Loop over time steps.
        for t in range(Ntstart, Nt):
            
            # State at the current time.
            xt = data.x[t, :]
            Ca, Cb, Cc = xt[0:1], xt[1:2], xt[2:3]
            
            # True rates.
            r1, r2 = getTrueR1R2(Ca=Ca, Cb=Cb, Cc=Cc, 
                                 k1=k1, k2f=k2f, k2b=k2b)

            # Full Gb rates.
            r1NN, r2NN = getFullGbR1R2(Ca=Ca, Cb=Cb, Cc=Cc, 
                                       r1Weights=fGbR1Weights, 
                                       r2Weights=fGbR2Weights, 
                                       xuyscales=xuyscales)

            # Get errors in the reaction rate laws.
            fGbErrors['r1Errors'] += [np.abs(r1 - r1NN)/r1]
            fGbErrors['r2Errors'] += [np.abs(r2 - r2NN)/r2]

            # Partial Gb rates.
            ypseq = data.y[t-Np:t, yindices].reshape(Np*Ny, )
            upseq = data.u[t-Np:t, :].reshape(Np*Nu, )
            z = np.concatenate((ypseq, upseq))
            r1NN, r2NN = getPartialGbR1R2(Ca=Ca, Cb=Cb, z=z, Np=Np,
                                       r1Weights=pGbR1Weights, 
                                       r2Weights=pGbR2Weights, 
                                       xuyscales=xuyscales)

            # Get errors in the reaction rate laws. 
            pGbErrors['r1Errors'] += [np.abs(r1 - r1NN)/r1]
            pGbErrors['r2Errors'] += [np.abs(r2 - r2NN)/r2]

    # Convert to one 1D arrays. 
    fGbErrors['r1Errors'] = np.array(fGbErrors['r1Errors']).squeeze()
    fGbErrors['r2Errors'] = np.array(fGbErrors['r2Errors']).squeeze()
    pGbErrors['r1Errors'] = np.array(pGbErrors['r1Errors']).squeeze()
    pGbErrors['r2Errors'] = np.array(pGbErrors['r2Errors']).squeeze()

    # Return. 
    return fGbErrors, pGbErrors

def main():
    """ Main function to be executed. """

    # Load parameters.
    reac_parameters = PickleTool.load(filename=
                                         'reac_parameters.pickle',
                                         type='read')
    reac_hybfullgbtrain = PickleTool.load(filename=
                                      'reac_hybfullgbtrain.pickle',
                                      type='read')
    reac_hybpartialgbtrain = PickleTool.load(filename=
                                      'reac_hybpartialgbtrain.pickle',
                                      type='read')

    # Get true reaction rate parameters.
    plant_pars = reac_parameters['plant_pars']
    k1 = plant_pars['k1']
    k2f = plant_pars['k2f']
    k2b = plant_pars['k2b']

    # Get all the training data. 
    Ntstart = reac_parameters['Ntstart']
    training_data = reac_parameters['training_data']

    # Trained Weights and scaling. 
    fGbWeights = [reac_hybfullgbtrain['r1Weights'], 
                  reac_hybfullgbtrain['r2Weights']]
    pGbWeights = [reac_hybpartialgbtrain['r1Weights'], 
                  reac_hybpartialgbtrain['r2Weights']]
    Np = reac_hybpartialgbtrain['Np']
    xuyscales = reac_hybfullgbtrain['xuyscales']

    # Get errors in reactions 1 and 2 for the full GB model. 
    CaRange = list(np.arange(1e-2, 1.5, 0.01)[:, np.newaxis])
    CbRange = list(np.arange(0.5, 0.6, 0.01)[:, np.newaxis])
    CcRange = list(np.arange(0.2, 0.5, 0.01)[:, np.newaxis])
    r2XGrid, r2YGrid = np.meshgrid(CbRange, CcRange)
    r1Errors, r2Errors = getFullGbRateErrorsInStateSpace(CaRange=CaRange, 
                            CbRange=CbRange, CcRange=CcRange, 
                            r1Weights=fGbWeights[0], r2Weights=fGbWeights[1], 
                            xuyscales=xuyscales, k1=k1, k2f=k2f, k2b=k2b)
    r1CaRange = np.array(CaRange).squeeze()

    # Save calculations as a dictionary to plot later.
    fGbErrorsInStateSpace = dict(r1Errors=r1Errors, r1CaRange=r1CaRange,
                                 r2Errors=r2Errors, r2XGrid=r2XGrid, 
                                 r2YGrid=r2YGrid)

    # Get errors in the reactions on the generated training data.
    yindices = plant_pars['yindices']
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    (fGbErrors, 
     pGbErrors) = getRateErrorsOnGeneratedData(training_data=training_data, 
                                Ntstart=Ntstart, Np=Np, Ny=Ny, Nu=Nu, 
                                fGbWeights=fGbWeights, pGbWeights=pGbWeights, 
                                yindices=yindices, xuyscales=xuyscales, k1=k1, 
                                k2f=k2f, k2b=k2b)
    errorsOnGenData = dict(fGbErrors=fGbErrors, pGbErrors=pGbErrors)

    # Save data.
    PickleTool.save(data_object=[fGbErrorsInStateSpace, errorsOnGenData],
                    filename='reac_rateanalysis.pickle')

main()