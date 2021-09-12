# [depends] %LIB%/hybridId.py %LIB%/BlackBoxFuncs.py
# [depends] %LIB%/TwoReacHybridFuncs.py
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool, quick_sim
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx, fnn

from ReacHybridFullGbFuncs import get_hybrid_pars as get_fullhybrid_pars
from ReacHybridFullGbFuncs import hybrid_fxup as fullhybrid_fxup
from ReacHybridFullGbFuncs import hybrid_hx as fullhybrid_hx

from ReacHybridPartialGbFuncs import get_hybrid_pars as get_partialhybrid_pars
from ReacHybridPartialGbFuncs import hybrid_fxup as partialhybrid_fxup
from ReacHybridPartialGbFuncs import hybrid_hx as partialhybrid_hx

def main():
    """ Main function to be executed. """

    # Load data.
    reac_parameters = PickleTool.load(filename=
                                      'reac_parameters.pickle',
                                      type='read')
    reac_bbnntrain = PickleTool.load(filename=
                                        'reac_bbnntrain_dyndata.pickle',
                                        type='read')
    reac_hybfullgbtrain_dyndata = PickleTool.load(filename=
                                        'reac_hybfullgbtrain_dyndata.pickle',
                                        type='read')
    reac_hybpartialgbtrain_dyndata = PickleTool.load(filename=
                                        'reac_hybpartialgbtrain_dyndata.pickle',
                                        type='read')

    def check_bbnn(reac_bbnntrain, reac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        plant_pars = reac_parameters['plant_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Np = reac_bbnntrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state for forecasting.
        training_data = reac_parameters['training_data_dyn'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Nu, )
        yz0 = np.concatenate((y0, yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        bbnn_pars = get_bbnn_pars(train=reac_bbnntrain, 
                                  plant_pars=plant_pars)
        fxu = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
        hx = lambda x: bbnn_hx(x, bbnn_pars)

        # CHeck black-box model validation.
        bb_yval = reac_bbnntrain['val_predictions'][-1].y
        bb_xpred, bb_ypred = quick_sim(fxu, hx, yz0, uval)
        breakpoint()
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

        # Get initial z0.
        yp0seq = (training_data.y[tthrow-Np:tthrow, :] - ymean)/ystd
        yp0seq = yp0seq.reshape(Np*Ny, )
        up0seq = (training_data.u[tthrow-Np:tthrow, :] - umean)/ustd
        up0seq = up0seq.reshape(Np*Nu, )
        z0 = np.concatenate((yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        hyb_pars = get_fullhybrid_pars(train=reac_hybtrain, 
                                   hyb_fullgb_pars=hyb_fullgb_pars)

        # If initial state was set using a NN.
        # Cbmean, Cbstd = ymean[-1:], ystd[-1:]
        # Cc0 = fnn(z0, hyb_pars['estCWeights'])*Cbstd + Cbmean
        # x0 = np.concatenate((y0, Cc0))

        # If initial state was chosen randomly.
        unmeasGbx0 = reac_hybtrain['unmeasGbx0_list'][-1][:, 0]
        unmeasGbx0 = unmeasGbx0*ystd[-1] + ymean[-1]
        x0 = np.concatenate((y0, unmeasGbx0))

        # Steady state disturbance.
        ps = hyb_pars['ps']
        fxu = lambda x, u: fullhybrid_fxup(x, u, ps, hyb_pars)
        hx = lambda x: fullhybrid_hx(x, hyb_pars)

        # CHeck black-box model validation.
        hyb_yval = reac_hybtrain['val_predictions'][-1].y
        hyb_xval = reac_hybtrain['val_predictions'][-1].x
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, x0, uval)
        breakpoint()
        # Return.
        return 

    def check_hybridpartialgb(reac_hybtrain, reac_parameters):
        """ Check Hybrid full grey-box functions. """

        # Get plant parameters.
        hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']

        # Get some sizes/parameters.
        tthrow = 10
        Ny, Nu = hyb_partialgb_pars['Ny'], hyb_partialgb_pars['Nu']
        Np = reac_hybtrain['Np']

        # Get scaling.
        umean, ustd = reac_hybtrain['xuyscales']['uscale']
        ymean, ystd = reac_hybtrain['xuyscales']['yscale']
        xzmean = np.concatenate((np.tile(ymean, (Np + 1, )), 
                             np.tile(umean, (Np, ))))
        xzstd = np.concatenate((np.tile(ystd, (Np + 1, )), 
                            np.tile(ustd, (Np, ))))


        # Get initial state for forecasting.
        training_data = reac_parameters['training_data_dyn'][-1]
        uval = training_data.u[tthrow:, :]
        y0 = training_data.y[tthrow, :]
        yp0seq = training_data.y[tthrow-Np:tthrow, :].reshape(Np*Ny, )
        up0seq = training_data.u[tthrow-Np:tthrow, :].reshape(Np*Nu, )
        xz0 = np.concatenate((y0, yp0seq, up0seq))
        
        # Get the black-box model parameters and function handles.
        hyb_pars = get_partialhybrid_pars(train=reac_hybtrain, 
                                          hyb_partialgb_pars=hyb_partialgb_pars)

        # Steady state disturbance.
        ps = hyb_pars['ps']
        fxu = lambda x, u: partialhybrid_fxup(x, u, ps, hyb_pars)
        hx = lambda x: partialhybrid_hx(x, hyb_pars)

        # CHeck black-box model validation.
        hyb_yval = reac_hybtrain['val_predictions'][-1].y
        hyb_xval = reac_hybtrain['val_predictions'][-1].x
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, xz0, uval)
        breakpoint()
        # Return.
        return 

    check_bbnn(reac_bbnntrain, reac_parameters)
    check_hybridfullgb(reac_hybfullgbtrain_dyndata, reac_parameters)
    check_hybridpartialgb(reac_hybpartialgbtrain_dyndata, reac_parameters)
    print("Hi")

main()