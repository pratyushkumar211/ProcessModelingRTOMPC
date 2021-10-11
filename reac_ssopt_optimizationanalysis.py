# [depends] %LIB%/plottingFuncs.py %LIB%/hybridId.py
# [depends] %LIB%/BlackBoxFuncs.py %LIB%/ReacHybridFuncs.py
# [depends] %LIB%/linNonlinMPC.py %LIB%/reacFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plottingFuncs import ReacPlots, PAPER_FIGSIZE
from hybridId import PickleTool
from linNonlinMPC import c2dNonlin, getSSOptimum, getXsYsSscost
from reacFuncs import cost_yup, plant_ode

# Import function handles for Black-Box, Full Grey-Box and Hybrid Grey-Box.
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from ReacHybridFullGbFuncs import get_hybrid_pars as get_fhyb_pars
from ReacHybridFullGbFuncs import hybrid_fxup as fhyb_fxup
from ReacHybridFullGbFuncs import hybrid_hx as fhyb_hx
from ReacHybridPartialGbFuncs import get_hybrid_pars as get_phyb_pars
from ReacHybridPartialGbFuncs import hybrid_fxup as phyb_fxup
from ReacHybridPartialGbFuncs import hybrid_hx as phyb_hx

def get_xuguess(*, model_type, plant_pars, model_pars):
    """ Get x and u guesses depending on model type. """
    
    us = plant_pars['us']

    if model_type == 'Plant':

        xs = plant_pars['xs']

    elif model_type == 'Black-Box-NN':
        
        Np = model_pars['Np']
        Ny = model_pars['Ny']
        xs = np.concatenate((np.tile(plant_pars['xs'][:Ny], (Np)), 
                             np.tile(plant_pars['us'], (Np))))

    elif model_type == 'Hyb-FGb':

        xs = plant_pars['xs']

    elif model_type == 'Hyb-PGb':

        Np = model_pars['Np']
        Ny = model_pars['Ny']
        xs = np.concatenate((np.tile(plant_pars['xs'][:Ny], (Np+1)), 
                             np.tile(plant_pars['us'], (Np))))

    else:
        None

    # Return as dict.
    return dict(x=xs, u=us)

def getBestSSOptimum(*, fxu, hx, lyu, model_pars, Nguess):
    """ Solve the steady-state optimization for multiple initial guesses. 
        and return the best solution. 

        i.e, Heuristic method to find the global optimization.    
    """

    # Create empty lists to store solutions.
    xs_list, us_list, ys_list, sscost_list = [], [], [], []

    # Loop over all the guesses.
    for _ in range(Nguess):

        # Generate a random us within the input constraint
        # limit and corresponding xs.

        getSSOptimum(fxu=fxu, hx=hx, lyu=lyu, 
                     parameters=model_pars, guess=xuguess)

    # Compare the steady-state cost values. 

    # Return.
    return xs, us, ys, sscost

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
    
    # Partial GB Hybrid model and function handles.
    phyb_pars = get_phyb_pars(train=reac_hybpartialgbtrain, 
                              hyb_partialgb_pars=hyb_partialgb_pars, 
                              plant_pars=plant_pars)
    ps = phyb_pars['ps']
    phyb_f = lambda x, u: phyb_fxup(x, u, ps, phyb_pars)
    phyb_h = lambda x: phyb_hx(x, phyb_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN', 'Hyb-FGb', 'Hyb-PGb']
    fxu_list = [plant_f, bbnn_f, fhyb_f, phyb_f]
    hx_list = [plant_h, bbnn_h, fhyb_h, phyb_h]
    par_list = [plant_pars, bbnn_pars, fhyb_pars, phyb_pars]

    # Generate parameters at which to do the optimization analysis.
    plb = np.array([50, 500])
    pub = np.array([200, 1500])
    Npvals = 10
    pvals = list((pub-plb)*np.random.rand(Npvals, 2) + plb)

    # Create lists to store the all the optimization results/sub gaps.    
    xs_list, us_list, subgaps_list = [], [], []

    # Loop over all the models.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # Generate lists to store the optimization results 
        # and suboptimality gaps for a model.
        model_xs = []
        model_us = []
        model_subgaps = []

        # Loop over all the parameter values.
        for p in pvals:
            
            # Get guess.
            xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, 
                              model_pars=model_pars)
            xguess = xuguess['x']

            # Get the steady state optimum.
            xs, us, ys, opt_sscost = getSSOptimum(fxu=fxu, hx=hx, lyu=lyu, 
                                                  parameters=model_pars,
                                                  guess=xuguess)

            # Store result.
            model_xs += [xs]
            model_us += [us]
            model_sscost += [sscost]

            # Compute suboptimality gaps if the model is not a plant model.



        # Model steady states and costs.        
        model_xs = np.asarray(model_xs)
        model_us = np.asarray(model_us)
        model_sscost = np.asarray(model_sscost)

        # Store steady states and costs in lists.
        xs_list += [model_xs]
        sscosts += [model_sscost]

    # Get us as rank 1 array.
    us = np.asarray(us_list)[:, 0]

    # Create data object and save.
    reac_ssopt = dict(us=us, xs=xs_list, sscosts=sscosts)

    # Save.
    PickleTool.save(data_object=reac_ssopt,
                    filename='reac_ssopt_optimizationanalysis.pickle')

main()