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
                                        'reac_bbnntrain.pickle',
                                        type='read')
    reac_hybfullgbtrain = PickleTool.load(filename=
                                        'reac_hybfullgbtrain.pickle',
                                        type='read')
    reac_hybpartialgbtrain = PickleTool.load(filename=
                                        'reac_hybpartialgbtrain.pickle',
                                        type='read')

    def check_bbnn(reac_bbnntrain, reac_parameters):
        """ Check Black-box functions. """

        # Get plant parameters.
        plant_pars = reac_parameters['plant_pars']

        # Extract out the trained information. 
        reac_bbnntrain = reac_bbnntrain[0]

        # Sizes.
        Ntstart = reac_parameters['Ntstart']
        Np = reac_bbnntrain['Np']
        Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

        # Get initial state.
        training_data = reac_parameters['training_data_nonoise'][-1]
        uval = training_data.u[Ntstart:, :]
        yp0seq = training_data.y[Ntstart-Np:Ntstart, :].reshape(Np*Ny, )
        up0seq = training_data.u[Ntstart-Np:Ntstart, :].reshape(Np*Nu, )
        z0 = np.concatenate((yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        bbnn_pars = get_bbnn_pars(train=reac_bbnntrain, 
                                  plant_pars=plant_pars)
        fxu = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
        hx = lambda x: bbnn_hx(x, bbnn_pars)

        # Check black-box model validation.
        bb_yval = reac_bbnntrain['val_predictions'].y
        bb_xpred, bb_ypred = quick_sim(fxu, hx, z0, uval)
        breakpoint()
        # Return.
        return 

    def check_hybridfullgb(reac_hybtrain, reac_parameters):
        """ Check Hybrid full grey-box functions. """

        # Get plant parameters.
        plant_pars = reac_parameters['plant_pars']
        hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']

        # Extract out the trained information. 
        reac_hybtrain = reac_hybtrain[0]

        # Sizes.
        Ntstart = reac_parameters['Ntstart']
        Ny, Nu = hyb_fullgb_pars['Ny'], hyb_fullgb_pars['Nu']
        Np = reac_hybtrain['Np']

        # Get scaling.
        xmean, xstd = reac_hybtrain['xuyscales']['xscale']
        umean, ustd = reac_hybtrain['xuyscales']['uscale']
        ymean, ystd = reac_hybtrain['xuyscales']['yscale']

        # Get initial state for forecasting.
        training_data = reac_parameters['training_data_nonoise'][-1]
        uval = training_data.u[Ntstart:, :]
        y0 = training_data.y[Ntstart, :]

        # Get initial z0.
        yp0seq = (training_data.y[Ntstart-Np:Ntstart, :] - ymean)/ystd
        yp0seq = yp0seq.reshape(Np*Ny, )
        up0seq = (training_data.u[Ntstart-Np:Ntstart, :] - umean)/ustd
        up0seq = up0seq.reshape(Np*Nu, )
        z0 = np.concatenate((yp0seq, up0seq))

        # Get the black-box model parameters and function handles.
        hyb_pars = get_fullhybrid_pars(train=reac_hybtrain, 
                                       hyb_fullgb_pars=hyb_fullgb_pars, 
                                       plant_pars=plant_pars)

        # If initial state was set using a NN.
        Cbmean, Cbstd = ymean[-1:], ystd[-1:]
        Cc0 = fnn(z0, hyb_pars['estC0Weights'])*Cbstd + Cbmean
        x0 = np.concatenate((y0, Cc0))
        
        # Steady state disturbance.
        ps = hyb_pars['ps']
        fxu = lambda x, u: fullhybrid_fxup(x, u, ps, hyb_pars)
        hx = lambda x: fullhybrid_hx(x, hyb_pars)

        # Check black-box model validation.
        hyb_yval = reac_hybtrain['val_predictions'].y
        hyb_xval = reac_hybtrain['val_predictions'].x
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, x0, uval)
        breakpoint()
        # Return.
        return 

    def check_hybridpartialgb(reac_hybtrain, reac_parameters):
        """ Check Hybrid full grey-box functions. """

        # Get plant parameters.
        plant_pars = reac_parameters['plant_pars']
        hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']

        # Extract out the trained information. 
        reac_hybtrain = reac_hybtrain[0]

        # Sizes.
        Ntstart = reac_parameters['Ntstart']
        Ny, Nu = hyb_partialgb_pars['Ny'], hyb_partialgb_pars['Nu']
        Np = reac_hybtrain['Np']

        # Get scaling.
        umean, ustd = reac_hybtrain['xuyscales']['uscale']
        ymean, ystd = reac_hybtrain['xuyscales']['yscale']
        xzmean = np.concatenate((np.tile(ymean, (Np + 1, )), 
                                 np.tile(umean, (Np, ))))
        xzstd = np.concatenate((np.tile(ystd, (Np + 1, )), 
                                np.tile(ustd, (Np, ))))

        # Get initial state.
        training_data = reac_parameters['training_data_nonoise'][-1]
        uval = training_data.u[Ntstart:, :]
        y0 = training_data.y[Ntstart, :]
        yp0seq = training_data.y[Ntstart-Np:Ntstart, :].reshape(Np*Ny, )
        up0seq = training_data.u[Ntstart-Np:Ntstart, :].reshape(Np*Nu, )
        xz0 = np.concatenate((y0, yp0seq, up0seq))
        
        # Get the black-box model parameters and function handles.
        hyb_pars = get_partialhybrid_pars(train=reac_hybtrain, 
                                          hyb_partialgb_pars=hyb_partialgb_pars,plant_pars=plant_pars)

        # Steady state disturbance.
        ps = hyb_pars['ps']
        fxu = lambda x, u: partialhybrid_fxup(x, u, ps, hyb_pars)
        hx = lambda x: partialhybrid_hx(x, hyb_pars)

        # CHeck black-box model validation.
        hyb_yval = reac_hybtrain['val_predictions'].y
        hyb_xval = reac_hybtrain['val_predictions'].x
        hyb_xpred, hyb_ypred = quick_sim(fxu, hx, xz0, uval)
        breakpoint()
        # Return.
        return 

    check_bbnn(reac_bbnntrain, reac_parameters)
    check_hybridfullgb(reac_hybfullgbtrain, reac_parameters)
    check_hybridpartialgb(reac_hybpartialgbtrain, reac_parameters)
    print("Hi")

main()