# [depends] %LIB%/hybridid.py %LIB%/BlackBoxFuncs.py
# [depends] %LIB%/TwoReacHybridFuncs.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_bbnntrain.pickle
# [depends] tworeac_hybtrain.pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool, quick_sim
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from TwoReacHybridFuncs import (get_hybrid_pars,
                                hybrid_fxup, hybrid_hx)
# from KoopmanModelFuncs import get_KoopmanModel_pars, koop_fxu, koop_hx
from InputConvexFuncs import get_icnn_pars, icnn_lyu
from InputConvexFuncs import get_picnn_pars, picnn_lyup

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    tworeac_bbnntrain = PickleTool.load(filename='tworeac_bbnntrain.pickle',
                                        type='read')
    tworeac_hybtrain = PickleTool.load(filename='tworeac_hybtrain.pickle',
                                        type='read')
    tworeac_icnntrain = PickleTool.load(filename='tworeac_icnntrain.pickle',
                                        type='read')
    tworeac_picnntrain = PickleTool.load(filename='tworeac_picnntrain.pickle',
                                        type='read')

    def check_bbnn(tworeac_bbnntrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        plant_pars = tworeac_parameters['plant_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Np = tworeac_bbnntrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state for forecasting.
        training_data = tworeac_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        bbnn_pars = get_bbnn_pars(train=tworeac_bbnntrain, 
                                  plant_pars=plant_pars)
        fxu = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
        hx = lambda x: bbnn_hx(x, bbnn_pars)

        # CHeck black-box model validation.
        bb_yval = tworeac_bbnntrain['val_predictions'][-1].y
        bb_xpred, bb_ypred = quick_sim(fxu, hx, yz0, uval)
        breakpoint()
        # Return 
        return 

    def check_hybrid(tworeac_hybtrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        hyb_greybox_pars = tworeac_parameters['hyb_greybox_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Ny, Nu = hyb_greybox_pars['Ny'], hyb_greybox_pars['Nu']

        # Get initial state for forecasting.
        training_data = tworeac_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        x0 = training_data.x[tthrow, :]

        # Get the black-box model parameters and function handles.
        hyb_pars = get_hybrid_pars(train=tworeac_hybtrain, 
                                  hyb_greybox_pars=hyb_greybox_pars)
        ps = hyb_pars['ps']
        fxu = lambda x, u: hybrid_fxup(x, u, ps, hyb_pars)
        hx = lambda x: hybrid_hx(x)

        # CHeck black-box model validation.
        hyb_yval = tworeac_hybtrain['val_predictions'][-1].y
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, x0, uval)
        breakpoint()
        # Return 
        return 

    def check_koopman(tworeac_kooptrain, tworeac_parameters):
        """ Check Black-box functions. """
        plant_pars = tworeac_parameters['plant_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Np = tworeac_kooptrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state for forecasting.
        training_data = tworeac_parameters['training_data'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Scale initial state and get the lifted state.
        fNWeights = tworeac_kooptrain['trained_weights'][-1][:-2]
        xuyscales = tworeac_kooptrain['xuyscales']
        ymean, ystd = xuyscales['yscale']
        umean, ustd = xuyscales['uscale']
        yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
                                np.tile(umean, (Np, ))))
        yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
                            np.tile(ustd, (Np, ))))
        yz0 = (yz0 - yzmean)/yzstd
        xkp0 = np.concatenate((yz0, fnn(yz0, fNWeights, 1.)))

        # Get the black-box model parameters and function handles.
        koop_pars = get_KoopmanModel_pars(train=tworeac_kooptrain, 
                                          plant_pars=plant_pars)

        # Get function handles.
        fxu = lambda x, u: koop_fxu(x, u, koop_pars)
        hx = lambda x: koop_hx(x, koop_pars)

        # CHeck black-box model validation.
        koop_yval = tworeac_kooptrain['val_predictions'][-1].y
        koop_xpred, koop_ypred = quick_sim(fxu, hx, xkp0, uval)
        breakpoint()
        #Return 
        return 

    def check_icnn(tworeac_icnntrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Plant parameters.
        plant_pars = tworeac_parameters['plant_pars']

        # Get ICNN function handles.
        icnn_pars = get_icnn_pars(train=tworeac_icnntrain, 
                                  plant_pars=plant_pars)
        icnn_lu = lambda u: icnn_lyu(u, icnn_pars)

        # CHeck black-box model validation.
        lyup_val = tworeac_icnntrain['val_predictions'][-1]['lyup']
        uval = tworeac_icnntrain['val_predictions'][-1]['u']
        
        lyup_pred = []
        for u in uval:
            lyup_pred += [icnn_lu(u)]
        lyup_pred = np.array(lyup_pred).squeeze()
        breakpoint()
        #Return 
        return 

    def check_picnn(tworeac_picnntrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Plant parameters.
        plant_pars = tworeac_parameters['plant_pars']

        # Get ICNN function handles.
        picnn_pars = get_picnn_pars(train=tworeac_picnntrain, 
                                    plant_pars=plant_pars)
        picnn_lup = lambda u, p: picnn_lyup(u, p, picnn_pars)

        # CHeck black-box model validation.
        lyup_val = tworeac_picnntrain['val_predictions'][-1]['lyup']
        uval = tworeac_picnntrain['val_predictions'][-1]['u']
        pval = tworeac_picnntrain['val_predictions'][-1]['p']

        lyup_pred = []
        for u, p in zip(uval, pval):
            lyup_pred += [picnn_lup(u, p)]
        lyup_pred = np.array(lyup_pred).squeeze()
        breakpoint()
        #Return 
        return 

    check_bbnn(tworeac_bbnntrain, tworeac_parameters)
    check_icnn(tworeac_icnntrain, tworeac_parameters)
    check_picnn(tworeac_picnntrain, tworeac_parameters)
    #check_koopman(tworeac_kooptrain, tworeac_parameters)
    check_hybrid(tworeac_hybtrain, tworeac_parameters)
    print("Hi")

main()


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
