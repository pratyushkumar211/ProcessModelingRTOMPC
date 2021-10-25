# [depends] %LIB%/plottingFuncs.py %LIB%/hybridId.py
# [depends] %LIB%/BlackBoxFuncs.py %LIB%/ReacHybridFuncs.py
# [depends] %LIB%/linNonlinMPC.py %LIB%/reacFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool
from linNonlinMPC import (c2dNonlin, getSSOptimum, 
                          getXsYsSscost, getXsUsSSCalcGuess)
from reacFuncs import cost_lxup_noCc, cost_lxup_withCc, plant_ode

# Functions for Black-Box, full hybrid, and partial hybrid models.
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from ReacHybridFullGbFuncs import get_hybrid_pars as get_fhyb_pars
from ReacHybridFullGbFuncs import hybrid_fxup as fhyb_fxup
from ReacHybridFullGbFuncs import hybrid_hx as fhyb_hx
from ReacHybridPartialGbFuncs import get_hybrid_pars as get_phyb_pars
from ReacHybridPartialGbFuncs import hybrid_fxup as phyb_fxup
from ReacHybridPartialGbFuncs import hybrid_hx as phyb_hx

def getSSOptimums(*, model_types, fxu_list, hx_list, 
                     par_list, plant_pars, cost_lxup):
    """ Get steady-state optimums of all the 
        models with the specified stage cost. 
    """

    # Lists to store optimization results. 
    xs_list, us_list, optSscost_list = [], [], []

    # Initial guess for input. 
    uguess = np.array([1.5])

    # Loop over the different models and obtain the SS optimums.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # Get Guess.
        xuguess = getXsUsSSCalcGuess(model_type=model_type, fxu=fxu, hx=hx,
                                     model_pars=model_pars, plant_pars=plant_pars, us=uguess)

        # Get the steady state optimum.
        xs, us, ys, optSscost = getSSOptimum(fxu=fxu, hx=hx, 
                                             lxu=cost_lxup, 
                                             parameters=model_pars, 
                                             guess=xuguess)

        # Save.
        xs_list += [xs]
        us_list += [us]
        optSscost_list += [optSscost]

    # Return.
    return xs_list, us_list, optSscost_list

def printOptimums(model_type, xs_list, us_list):
    """ Quick function to print the optimums. """
    for model, us in zip(model_type, us_list):
        print(model + " us: " + str(us))
    

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

    # Extract out the training data for analysis. 
    # Change the index if need to switch between data with and without noise.
    reac_bbnntrain = reac_bbnntrain[1]
    reac_hybfullgbtrain = reac_hybfullgbtrain[1]
    reac_hybpartialgbtrain = reac_hybpartialgbtrain[1]

    # Get plant and hybrid model parameters.
    plant_pars = reac_parameters['plant_pars']
    hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']
    hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']

    # Plant function handles.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_f = c2dNonlin(plant_fxu, Delta)
    plant_h = lambda x: x[plant_pars['yindices']]

    # Black-Box NN function handles.
    bbnn_pars = get_bbnn_pars(train=reac_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Full GB Hybrid model function handles.
    fhyb_pars = get_fhyb_pars(train=reac_hybfullgbtrain, 
                              hyb_fullgb_pars=hyb_fullgb_pars, 
                              plant_pars=plant_pars)
    ps = fhyb_pars['ps']
    fhyb_f = lambda x, u: fhyb_fxup(x, u, ps, fhyb_pars)
    fhyb_h = lambda x: fhyb_hx(x, fhyb_pars)
    
    # Get the partial hybrid model parameters and function handles.
    phyb_pars = get_phyb_pars(train=reac_hybpartialgbtrain, 
                              hyb_partialgb_pars=hyb_partialgb_pars, 
                              plant_pars=plant_pars)
    ps = phyb_pars['ps']
    phyb_f = lambda x, u: phyb_fxup(x, u, ps, phyb_pars)
    phyb_h = lambda x: phyb_hx(x, phyb_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN', 
                   'Hybrid-FullGb', 'Hybrid-PartialGb']
    fxu_list = [plant_f, bbnn_f, fhyb_f, phyb_f]
    hx_list = [plant_h, bbnn_h, fhyb_h, phyb_h]
    par_list = [plant_pars, bbnn_pars, fhyb_pars, phyb_pars]

    # Get the optimums for the cost without a Cc term.
    p = [100, 900]
    cost_lxup = lambda x, u: cost_lxup_noCc(x, u, p)
    (cost1_xs_list, cost1_us_list, 
     cost1_optSscost_list) = getSSOptimums(model_types=model_types, 
                                           fxu_list=fxu_list, 
                                           hx_list=hx_list,
                                           par_list=par_list,
                                           plant_pars=plant_pars,
                                           cost_lxup=cost_lxup)

    # Get the optimums for the cost with a Cc term.
    model_types = ['Plant', 'Hybrid-FullGb']
    fxu_list = [plant_f, fhyb_f]
    hx_list = [plant_h, fhyb_h]
    par_list = [plant_pars, fhyb_pars]
    p = [100, 1000, 100]
    cost_lxup = lambda x, u: cost_lxup_withCc(x, u, p)
    (cost2_xs_list, cost2_us_list, 
     cost2_optSscost_list) = getSSOptimums(model_types=model_types,
                                           fxu_list=fxu_list,
                                           hx_list=hx_list,
                                           par_list=par_list,
                                           plant_pars=plant_pars,
                                           cost_lxup=cost_lxup)

    # Print the optimums to look over in the terminal.
    # Cost without a Cc contribution.
    print("Cost Type 1")
    model_types = ['Plant', 'Black-Box-NN', 
                   'Hybrid-FullGb', 'Hybrid-PartialGb']
    printOptimums(model_types, cost1_xs_list, cost1_us_list)

    # Cost with a Cc contribution.
    print("Cost Type 2")
    model_types = ['Plant', 'Hybrid-FullGb']
    printOptimums(model_types, cost2_xs_list, cost2_us_list)
    
main()