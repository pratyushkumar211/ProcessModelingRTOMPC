# [depends] %LIB%/hybridid.py %LIB%/BlackBoxFuncs.py
# [depends] %LIB%/CstrFlashHybridFuncs.py
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_bbnntrain.pickle
# [depends] cstr_flash_hybtrain.pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool, quick_sim
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from CstrFlashHybridFuncs import get_hybrid_pars, hybrid_fxup, hybrid_hx
from InputConvexFuncs import get_icnn_pars, icnn_lyu

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
    cstr_flash_icnntrain = PickleTool.load(filename=
                                     'cstr_flash_icnntrain.pickle',
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
        hyb_greybox_pars = cstr_flash_parameters['hyb_greybox_pars']
        ps = hyb_greybox_pars['ps']
        tthrow = 10
        Ny, Nu = hyb_greybox_pars['Ny'], hyb_greybox_pars['Nu']

        # Get initial state for forecasting.
        training_data = cstr_flash_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        x0 = training_data.x[tthrow, :]

        # Get the black-box model parameters and function handles.
        hyb_pars = get_hybrid_pars(train=cstr_flash_hybtrain, 
                                   hyb_greybox_pars=hyb_greybox_pars)
        fxu = lambda x, u: hybrid_fxup(x, u, ps, hyb_pars)
        hx = hybrid_hx

        # CHeck black-box model validation.
        hyb_yval = cstr_flash_hybtrain['val_predictions'][-1].y
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, x0, uval)
        breakpoint()
        # Return 
        return 

    def check_icnn(cstr_flash_parameters, cstr_flash_icnntrain):
        """ Function to check the ICNN implementation. """

        # Plant parameters.
        plant_pars = cstr_flash_parameters['plant_pars']

        # Get ICNN function handles.
        icnn_pars = get_icnn_pars(train=cstr_flash_icnntrain, 
                                  plant_pars=plant_pars)
        icnn_lu = lambda u: icnn_lyu(u, icnn_pars)

        # CHeck black-box model validation.
        lyup_val = cstr_flash_icnntrain['val_predictions'][-1]['lyup']
        uval = cstr_flash_icnntrain['val_predictions'][-1]['u']
        
        lyup_pred = []
        for u in uval:
            lyup_pred += [icnn_lu(u)]
        lyup_pred = np.array(lyup_pred).squeeze()
        breakpoint()
        #Return 
        return 

    # Check the different types of functions.
    check_bbnn(cstr_flash_parameters, cstr_flash_bbnntrain)
    check_hyb(cstr_flash_parameters, cstr_flash_hybtrain)
    check_icnn(cstr_flash_parameters, cstr_flash_icnntrain)

main()