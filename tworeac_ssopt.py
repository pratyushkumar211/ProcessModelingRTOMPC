# [depends] %LIB%/plottingFuncs.py %LIB%/hybridId.py
# [depends] %LIB%/BlackBoxFuncs.py %LIB%/TwoReacHybridFuncs.py
# [depends] %LIB%/linNonlinMPC.py %LIB%/tworeacFuncs.py
# [depends] tworeac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plottingFuncs import TwoReacPlots, PAPER_FIGSIZE
from hybridId import PickleTool
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from TwoReacHybridFuncs import get_hybrid_pars, hybrid_fxup, hybrid_hx
from linNonlinMPC import c2dNonlin, getSSOptimum, getXsYsSscost
from tworeacFuncs import cost_yup, plant_ode

def get_xuguess(*, model_type, plant_pars):
    """ Get x and u guesses depending on model type. """
    
    us = plant_pars['us']

    if model_type == 'Plant':

        xs = plant_pars['xs']

    elif model_type == 'Black-Box-NN':

        xs = plant_pars['xs']

    else:

        xs = plant_pars['xs']

    # Return as dict.
    return dict(x=xs, u=us)

def main():
    """ Main function to be executed. """

    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')
    tworeac_bbnntrain = PickleTool.load(filename=
                                    'tworeac_bbnntrain_dyndata.pickle',
                                      type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                      'tworeac_hybtrain_dyndata.pickle',
                                      type='read')

    # Get plant and grey-box parameters. 
    plant_pars = tworeac_parameters['plant_pars']
    hyb_greybox_pars = tworeac_parameters['hyb_greybox_pars']

    # Get cost function handle.
    p = [100, 800]
    lyu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    bbnn_pars = get_bbnn_pars(train=tworeac_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Get the black-box model parameters and function handles.
    hyb_pars = get_hybrid_pars(train=tworeac_hybtrain, 
                               hyb_greybox_pars=hyb_greybox_pars)
    ps = hyb_pars['ps']
    hybrid_f = lambda x, u: hybrid_fxup(x, u, ps, hyb_pars)
    hybrid_h = lambda x: hybrid_hx(x)
    
    # Get the plant function handle.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_f = c2dNonlin(plant_fxu, Delta)
    plant_h = lambda x: x[plant_pars['yindices']]

    # Lists to loop over for different models.
    model_types = ['Plant']
    fxu_list = [plant_f, bbnn_f, hybrid_f]
    hx_list = [plant_h, bbnn_h, hybrid_h]
    par_list = [plant_pars, bbnn_pars, hyb_pars]

    # Loop over the different models and obtain SS optimums.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # Get Guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars)
        
        # Get the steady state optimum.
        xs, us, ys, opt_sscost = getSSOptimum(fxu=fxu, hx=hx, lyu=lyu, 
                                              parameters=model_pars, 
                                              guess=xuguess)

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
    
    breakpoint()
    
    # Get a linspace of steady-state u values.
    ulb, uub = plant_pars['ulb'], plant_pars['uub']
    us_list = list(np.linspace(ulb, uub, 100))

    # Lists to store Steady-state cost.
    sscosts = []

    # Loop over all the models.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # List to store SS costs for one model.
        model_sscost = []

        # Compute SS cost.
        for us in us_list:
            
            _, _, sscost = getXsYsSscost(fxu=fxu, hx=hx, lyu=lyu, 
                                         us=us, parameters=model_pars, 
                                         xguess=plant_pars['xs'])
            model_sscost += [sscost]
        
        model_sscost = np.asarray(model_sscost)
        sscosts += [model_sscost]

    # Get us as rank 1 array.
    us = np.asarray(us_list)[:, 0]

    # Create data object and save.
    tworeac_ssopt = dict(us=us, sscosts=sscosts)

    # Save.
    PickleTool.save(data_object=tworeac_ssopt,
                    filename='tworeac_ssopt.pickle')


main()