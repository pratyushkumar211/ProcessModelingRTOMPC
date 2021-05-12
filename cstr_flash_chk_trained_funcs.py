# [depends] %LIB%/hybridid.py %LIB%/BlackBoxFuncs.py
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_bbnntrain.pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool, quick_sim
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from CstrFlashHybridFuncs import (get_CstrFlash_hybrid_pars, 
                                  CstrFlashHybrid_fxu, CstrFlashHybrid_hx)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                         'cstr_flash_parameters.pickle',
                                         type='read')
    cstr_flash_bbnntrain = PickleTool.load(filename=
                                     'cstr_flash_bbnntrain.pickle',
                                      type='read')
    cstr_flash_hybtrain = PickleTool.load(filename=
                                     'cstr_flash_hybtrain.pickle',
                                      type='read')

    def check_bbnn(cstr_flash_parameters, cstr_flash_bbnntrain):

        # Get some sizes/parameters.
        plant_pars = cstr_flash_parameters['plant_pars']
        tthrow = 10
        Np = cstr_flash_bbnntrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state for forecasting.
        training_data = cstr_flash_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Nu, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        bbnn_pars = get_bbnn_pars(train=cstr_flash_bbnntrain, 
                                  plant_pars=plant_pars)
        fxu = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
        hx = lambda x: bbnn_hx(x, bbnn_pars)

        # CHeck black-box model validation.
        bb_yval = cstr_flash_bbnntrain['val_predictions'][-1].y
        bb_xpred, bb_ypred = quick_sim(fxu, hx, yz0, uval)

        # Return.
        return 

    def check_hyb(cstr_flash_parameters, cstr_flash_hybtrain):

        # Get some sizes/parameters.
        greybox_pars = cstr_flash_parameters['greybox_pars']
        tthrow = 10
        Np = cstr_flash_hybtrain['Np']
        Ny, Nu = greybox_pars['Ny'], greybox_pars['Nu']

        # Get initial state for forecasting.
        training_data = cstr_flash_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Nu, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        hyb_pars = get_CstrFlash_hybrid_pars(train=cstr_flash_hybtrain, 
                                              greybox_pars=greybox_pars)
        fxu = lambda x, u: CstrFlashHybrid_fxu(x, u, hyb_pars)
        hx = lambda x: CstrFlashHybrid_hx(x, hyb_pars)

        # CHeck black-box model validation.
        hyb_yval = cstr_flash_hybtrain['val_predictions'][-1].y
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, yz0, uval)
        breakpoint()
        # Return 
        return 

    # Check the different types of functions.
    check_bbnn(cstr_flash_parameters, cstr_flash_bbnntrain)
    check_hyb(cstr_flash_parameters, cstr_flash_hybtrain)

main()