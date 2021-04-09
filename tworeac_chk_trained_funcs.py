# [depends] %LIB%/hybridid.py %LIB%/economicopt.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_bbtrain.pickle
# [depends] tworeac_kooptrain.pickle
# [makes] pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool, quick_sim
from BlackBoxFuncs import get_bbNN_pars, bbNN_fxu, bb_hx
from TwoReacHybridFuncs import (get_tworeacHybrid_pars,
                                tworeacHybrid_fxu, tworeacHybrid_hx)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    tworeac_bbNNtrain = PickleTool.load(filename='tworeac_bbNNtrain.pickle',
                                        type='read')
    tworeac_hybtrain = PickleTool.load(filename='tworeac_hybtrain.pickle',
                                        type='read')

    def check_bbNN(tworeac_bbNNtrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        plant_pars = tworeac_parameters['plant_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Np = tworeac_bbNNtrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state for forecasting.
        training_data = tworeac_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        bbNN_pars = get_bbNN_pars(train=tworeac_bbNNtrain, 
                                  plant_pars=plant_pars)
        fxu = lambda x, u: bbNN_fxu(x, u, bbNN_pars)
        hx = lambda x: bb_hx(x, bbNN_pars)

        # CHeck black-box model validation.
        bb_yval = tworeac_bbNNtrain['val_predictions'][-1].y
        bb_xpred, bb_ypred = quick_sim(fxu, hx, yz0, uval)
        breakpoint()
        # Return 
        return 

    def check_hybrid(tworeac_hybtrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        greybox_pars = tworeac_parameters['greybox_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Ny, Nu = greybox_pars['Ny'], greybox_pars['Nu']

        # Get initial state for forecasting.
        training_data = tworeac_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        x0 = training_data.x[tthrow, :]

        # Get the black-box model parameters and function handles.
        hyb_pars = get_tworeacHybrid_pars(train=tworeac_hybtrain, 
                                  greybox_pars=greybox_pars)
        fxu = lambda x, u: tworeacHybrid_fxu(x, u, hyb_pars)
        hx = lambda x: tworeacHybrid_hx(x)

        # CHeck black-box model validation.
        hyb_yval = tworeac_hybtrain['val_predictions'][-1].y
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, x0, uval)
        breakpoint()
        # Return 
        return 

    check_bbNN(tworeac_bbNNtrain, tworeac_parameters)
    check_hybrid(tworeac_hybtrain, tworeac_parameters)
    print("Hi")

main()

# def chk_koop_funcs(tworeac_kooptrain, tworeac_parameters):
#     """ Check Black-box functions. """
#     parameters = tworeac_parameters['parameters']

#     Get some sizes/parameters.
#     tthrow = 10
#     Np = tworeac_kooptrain['Np']
#     Ny, Nu = parameters['Ny'], parameters['Nu']

#     Get initial state.
#     uval = tworeac_parameters['training_data'][-1].u[tthrow:, :]
#     yp0seq = tworeac_parameters['training_data'][-1].y[tthrow-Np:tthrow, :]
#     yp0seq = yp0seq.reshape(Np*Ny, )
#     y0 = tworeac_parameters['training_data'][-1].y[tthrow, :]
#     up0seq = tworeac_parameters['training_data'][-1].u[tthrow-Np:tthrow, :]
#     up0seq = up0seq.reshape(Np*Nu, )
#     yz0 = np.concatenate((y0, yp0seq, up0seq))

#     Scale initial state and get the lifted state.
#     fN_weights = tworeac_kooptrain['trained_weights'][-1][:-2]
#     xuyscales = tworeac_kooptrain['xuyscales']
#     ymean, ystd = xuyscales['yscale']
#     umean, ustd = xuyscales['uscale']
#     yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
#                             np.tile(umean, (Np, ))))
#     yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
#                         np.tile(ustd, (Np, ))))
#     yz0 = (yz0 - yzmean)/yzstd
#     xkp0 = np.concatenate((yz0, fnn(yz0, fN_weights, 1.)))

#     Get the black-box model parameters and function handles.
#     koop_pars, koop_fxu, koop_hx = get_kooppars_fxu_hx(train=
#                                                     tworeac_kooptrain, 
#                                                     parameters=parameters)

#     CHeck black-box model validation.
#     koop_yval = tworeac_kooptrain['val_predictions'][-1].y
#     koop_xpred, koop_ypred = quick_sim(koop_fxu, koop_hx, xkp0, uval)
#     Return 
#     return 

# def check_iCNN(tworeac_iCNNtrain, tworeac_parameters):
#     """ Check Black-box functions. """

#     # Get plant parameters.
#     plant_pars = tworeac_parameters['plant_pars']

#     # Get some sizes/parameters.
#     tthrow = 10
#     Np = tworeac_iCNNtrain['Np']
#     Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

#     # Get initial state for forecasting.
#     training_data = tworeac_parameters['training_data'][-1]
#     uval = training_data.u[tthrow:, :]
#     y0 = training_data.y[tthrow, :]
#     yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
#     up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Ny, )
#     yz0 = np.concatenate((y0, yp0seq, up0seq))

#     # Get the black-box model parameters and function handles.
#     iCNN_pars = get_iCNN_pars(train=tworeac_iCNNtrain, 
#                               plant_pars=plant_pars)
#     fxu = lambda x, u: iCNN_fxu(x, u, iCNN_pars)
#     hx = lambda x: bb_hx(x, iCNN_pars)

#     # CHeck black-box model validation.
#     iCNN_yval = tworeac_iCNNtrain['val_predictions'][-1].y
#     iCNN_xpred, iCNN_ypred = quick_sim(fxu, hx, yz0, uval)
#     breakpoint()
#     # Return 
#     return 
