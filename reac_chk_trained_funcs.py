# [depends] %LIB%/hybridId.py %LIB%/BlackBoxFuncs.py
# [depends] %LIB%/TwoReacHybridFuncs.py
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool, quick_sim
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx, fnn
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

        # Get scaling.
        xmean, xstd = reac_hybtrain['xuyscales']['xscale']
        umean, ustd = reac_hybtrain['xuyscales']['uscale']
        ymean, ystd = reac_hybtrain['xuyscales']['yscale']

        # Get initial state for forecasting.
        training_data = reac_parameters['training_data_dyn'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = (training_data.y[tthrow-Np:tthrow, :] - ymean)/ystd
        yp0seq = yp0seq.reshape(Np*Ny, )
        up0seq = (training_data.u[tthrow-Np:tthrow, :] - umean)/ustd
        up0seq = up0seq.reshape(Np*Nu, )
        z0 = np.concatenate((yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        hyb_pars = get_hybrid_pars(train=reac_hybtrain, 
                                   hyb_fullgb_pars=hyb_fullgb_pars)

        # # Get initial concentration of C.
        # Cc0 = fnn(z0, hyb_pars['estCWeights'])

        # # Initial state.
        # x0 = np.concatenate((y0, Cc0))

        # If initial state was chosen randomly.
        unmeasGbx0 = reac_hybtrain['unmeasGbx0_list'][-1][:, 0]
        unmeasGbx0 = unmeasGbx0*ystd[-1] + ymean[-1]
        x0 = np.concatenate((y0, unmeasGbx0))

        # Steady state disturbance.
        ps = hyb_pars['ps']
        fxu = lambda x, u: hybrid_fxup(x, u, ps, hyb_pars)
        hx = lambda x: hybrid_hx(x, hyb_pars)

        # CHeck black-box model validation.
        hyb_yval = reac_hybtrain['val_predictions'][-1].y
        hyb_xval = reac_hybtrain['val_predictions'][-1].x
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, x0, uval)
        breakpoint()
        # Return.
        return 


    #check_bbnn(tworeac_bbnntrain, tworeac_parameters)
    check_hybridfullgb(reac_hybfullgbtrain_dyndata, reac_parameters)
    print("Hi")

main()