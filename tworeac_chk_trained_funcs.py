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
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    tworeac_bbtrain = PickleTool.load(filename='tworeac_bbtrain.pickle',
                                      type='read')

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
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=tworeac_bbtrain, 
                                                       parameters=parameters) 

    # CHeck black-box model validation.
    bb_yval = tworeac_bbtrain['val_predictions'][-1].y
    bb_xpred, bb_ypred = quick_sim(blackb_fxu, blackb_hx, x0, uval)

main()