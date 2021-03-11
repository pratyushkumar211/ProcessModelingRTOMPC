# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/../tworeac_nonlin.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_train.pickle
# [makes] pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import mpctools as mpc
import casadi
import copy
import numpy as np
from hybridid import PickleTool, quick_sim
from economicopt import get_kooppars_fxu_hx, get_bbpars_fxu_hx
from economicopt import c2dNonlin, get_xuguess, fnn

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    tworeac_bbtrain = PickleTool.load(filename='tworeac_bbtrain.pickle',
                                      type='read')
    tworeac_kooptrain = PickleTool.load(filename='tworeac_kooptrain.pickle',
                                      type='read')

    def chk_bb_funcs(tworeac_bbtrain, tworeac_parameters):
        """ Check Black-box functions. """
        parameters = tworeac_parameters['parameters']

        # Get some sizes/parameters.
        tthrow = 10
        Np = tworeac_bbtrain['Np']
        Ny, Nu = parameters['Ny'], parameters['Nu']

        # Get
        uval = tworeac_parameters['training_data'][-1].u[tthrow:, :]
        yp0seq = tworeac_parameters['training_data'][-1].y[tthrow-Np:tthrow, :]
        yp0seq = yp0seq.reshape(Np*Ny, )
        up0seq = tworeac_parameters['training_data'][-1].u[tthrow-Np:tthrow, :]
        up0seq = up0seq.reshape(Np*Nu, )
        x0 = np.concatenate((yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=
                                                        tworeac_bbtrain, 
                                                        parameters=parameters) 

        # CHeck black-box model validation.
        bb_yval = tworeac_bbtrain['val_predictions'][-1].y
        bb_xpred, bb_ypred = quick_sim(blackb_fxu, blackb_hx, x0, uval)
        # Return 
        return 

    def chk_koop_funcs(tworeac_kooptrain, tworeac_parameters):
        """ Check Black-box functions. """
        parameters = tworeac_parameters['parameters']

        # Get some sizes/parameters.
        tthrow = 10
        Np = tworeac_kooptrain['Np']
        Ny, Nu = parameters['Ny'], parameters['Nu']

        # Get initial state.
        uval = tworeac_parameters['training_data'][-1].u[tthrow:, :]
        yp0seq = tworeac_parameters['training_data'][-1].y[tthrow-Np:tthrow, :]
        yp0seq = yp0seq.reshape(Np*Ny, )
        y0 = tworeac_parameters['training_data'][-1].y[tthrow, :]
        up0seq = tworeac_parameters['training_data'][-1].u[tthrow-Np:tthrow, :]
        up0seq = up0seq.reshape(Np*Nu, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Scale initial state and get the lifted state.
        fN_weights = tworeac_kooptrain['trained_weights'][-1][:-2]
        xuyscales = tworeac_kooptrain['xuyscales']
        ymean, ystd = xuyscales['yscale']
        umean, ustd = xuyscales['uscale']
        yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
                                np.tile(umean, (Np, ))))
        yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
                               np.tile(ustd, (Np, ))))
        yz0 = (yz0 - yzmean)/yzstd
        xkp0 = np.concatenate((yz0, fnn(yz0, fN_weights, 1.)))

        # Get the black-box model parameters and function handles.
        koop_pars, koop_fxu, koop_hx = get_kooppars_fxu_hx(train=
                                                        tworeac_kooptrain, 
                                                        parameters=parameters)

        # CHeck black-box model validation.
        koop_yval = tworeac_kooptrain['val_predictions'][-1].y
        koop_xpred, koop_ypred = quick_sim(koop_fxu, koop_hx, xkp0, uval)
        breakpoint()
        # Return 
        return 

    chk_bb_funcs(tworeac_bbtrain, tworeac_parameters)
    chk_koop_funcs(tworeac_kooptrain, tworeac_parameters)
    print("Hi")

main()