# [depends] %LIB%/hybridId.py %LIB%/BlackBoxFuncs.py
# [depends] %LIB%/TwoReacHybridFuncs.py
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool, quick_sim
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from ReacHybridFullGbFuncs import get_hybrid_pars, hybrid_fxup, hybrid_hx

def main():
    """ Main function to be executed. """

    # Load data.
    reac_parameters = PickleTool.load(filename=
                                      'reac_parameters.pickle',
                                      type='read')
    # tworeac_bbnntrain = PickleTool.load(filename=
    #                                     'tworeac_bbnntrain_dyndata.pickle',
    #                                     type='read')
    reac_hybfullgbtrain_dyndata = PickleTool.load(filename=
                                        'reac_hybfullgbtrain_dyndata.pickle',
                                        type='read')

    # def check_bbnn_ss(tworeac_bbnntrain, tworeac_parameters):
    #     """ Check Black-box functions. """

    #     # Get plant parameters.
    #     plant_pars = tworeac_parameters['plant_pars']

    #     # Get some sizes/parameters.
    #     Ny = plant_pars['Ny'] 
    #     Nu = plant_pars['Nu']

    #     # Get initial state for forecasting.
    #     training_data = tworeac_parameters['training_data_ss']
    #     Nval = 45
    #     uval = training_data.u[-Nval:, np.newaxis, :]
    #     uval_list = list(np.repeat(uval, 2, axis=1))
    #     x0_list = list(training_data.y[-Nval:, :])

    #     # Get the black-box model parameters and function handles.
    #     bbnn_pars = get_bbnn_pars(train=tworeac_bbnntrain, 
    #                               plant_pars=plant_pars)
    #     fxu = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    #     hx = lambda x: bbnn_hx(x, bbnn_pars)

    #     # CHeck black-box model validation.
    #     bb_yval = tworeac_bbnntrain['val_predictions'][-1].y
    #     bb_ypred_list = []
    #     for x0, uval in zip(x0_list, uval_list):
    #         _, bb_ypred = quick_sim(fxu, hx, x0, uval)
    #         bb_ypred_list += [bb_ypred]
    #     bb_ypred = np.asarray(bb_ypred_list)

    #     # Return.
    #     return 

    def check_bbnn(tworeac_bbnntrain, tworeac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        plant_pars = tworeac_parameters['plant_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Np = tworeac_bbnntrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state for forecasting.
        training_data = tworeac_parameters['training_data_dyn'][-1]
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
        # Return.
        return 

    def check_hybridfullgb(reac_hybtrain, reac_parameters):
        """ Check Hybrid full grey-box functions. """

        # Get plant parameters.
        hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Ny, Nu = hyb_fullgb_pars['Ny'], hyb_fullgb_pars['Nu']
        Np = reac_hybtrain['Np']

        # Get initial state for forecasting.
        training_data = reac_parameters['training_data_dyn'][-1]
        uval = training_data.u[tthrow:, :]
        x0 = training_data.y[tthrow, :]
        ypseq0 = training_data.y
        upseq0 = training_data.u

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

    # def check_koopman(tworeac_kooptrain, tworeac_parameters):
    #     """ Check Black-box functions. """
    #     plant_pars = tworeac_parameters['plant_pars']

    #     # Get some sizes/parameters.
    #     tthrow = 10
    #     Np = tworeac_kooptrain['Np']
    #     Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

    #     # Get initial state for forecasting.
    #     training_data = tworeac_parameters['training_data'][-1]
    #     uval = training_data.u[tthrow:, :]
    #     y0 = training_data.y[tthrow, :]
    #     yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
    #     up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Ny, )
    #     yz0 = np.concatenate((y0, yp0seq, up0seq))

    #     # Scale initial state and get the lifted state.
    #     fNWeights = tworeac_kooptrain['trained_weights'][-1][:-2]
    #     xuyscales = tworeac_kooptrain['xuyscales']
    #     ymean, ystd = xuyscales['yscale']
    #     umean, ustd = xuyscales['uscale']
    #     yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
    #                             np.tile(umean, (Np, ))))
    #     yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
    #                         np.tile(ustd, (Np, ))))
    #     yz0 = (yz0 - yzmean)/yzstd
    #     xkp0 = np.concatenate((yz0, fnn(yz0, fNWeights, 1.)))

    #     # Get the black-box model parameters and function handles.
    #     koop_pars = get_KoopmanModel_pars(train=tworeac_kooptrain, 
    #                                       plant_pars=plant_pars)

    #     # Get function handles.
    #     fxu = lambda x, u: koop_fxu(x, u, koop_pars)
    #     hx = lambda x: koop_hx(x, koop_pars)

    #     # CHeck black-box model validation.
    #     koop_yval = tworeac_kooptrain['val_predictions'][-1].y
    #     koop_xpred, koop_ypred = quick_sim(fxu, hx, xkp0, uval)
    #     breakpoint()
    #     #Return 
    #     return 

    # def check_icnn(tworeac_icnntrain, tworeac_parameters):
    #     """ Check Black-box functions. """

    #     # Plant parameters.
    #     plant_pars = tworeac_parameters['plant_pars']

    #     # Get ICNN function handles.
    #     icnn_pars = get_icnn_pars(train=tworeac_icnntrain, 
    #                               plant_pars=plant_pars)
    #     icnn_lu = lambda u: icnn_lyu(u, icnn_pars)

    #     # CHeck black-box model validation.
    #     lyup_val = tworeac_icnntrain['val_predictions'][-1]['lyup']
    #     uval = tworeac_icnntrain['val_predictions'][-1]['u']
        
    #     lyup_pred = []
    #     for u in uval:
    #         lyup_pred += [icnn_lu(u)]
    #     lyup_pred = np.array(lyup_pred).squeeze()
    #     breakpoint()
    #     #Return 
    #     return 

    # def check_picnn(tworeac_picnntrain, tworeac_parameters):
    #     """ Check Black-box functions. """

    #     # Plant parameters.
    #     plant_pars = tworeac_parameters['plant_pars']

    #     # Get ICNN function handles.
    #     picnn_pars = get_picnn_pars(train=tworeac_picnntrain, 
    #                                 plant_pars=plant_pars)
    #     picnn_lup = lambda u, p: picnn_lyup(u, p, picnn_pars)

    #     # CHeck black-box model validation.
    #     lyup_val = tworeac_picnntrain['val_predictions'][-1]['lyup']
    #     uval = tworeac_picnntrain['val_predictions'][-1]['u']
    #     pval = tworeac_picnntrain['val_predictions'][-1]['p']

    #     lyup_pred = []
    #     for u, p in zip(uval, pval):
    #         lyup_pred += [picnn_lup(u, p)]
    #     lyup_pred = np.array(lyup_pred).squeeze()
    #     breakpoint()
    #     #Return 
    #     return 

    #check_bbnn(tworeac_bbnntrain, tworeac_parameters)
    #check_icnn(tworeac_icnntrain, tworeac_parameters)
    #check_picnn(tworeac_picnntrain, tworeac_parameters)
    #check_koopman(tworeac_kooptrain, tworeac_parameters)
    check_hybridfullgb(reac_hybfullgbtrain_dyndata, reac_parameters)
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