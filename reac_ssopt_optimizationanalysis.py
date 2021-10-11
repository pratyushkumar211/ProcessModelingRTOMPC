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

def get_XGuess(*, model_type, fxu, hx, us, model_pars, plant_pars):
    """ Get x and u guesses depending on model type. """
    if model_type == 'Plant':
        xs = plant_pars['xs']
    elif model_type == 'Black-Box-NN':
        Np = model_pars['Np']
        Ny = model_pars['Ny']
        xs = np.concatenate((np.tile(plant_pars['xs'][:Ny], (Np,)), 
                             np.tile(plant_pars['us'], (Np, ))))
    elif model_type == 'Hyb-FGb':
        xs = plant_pars['xs']
    elif model_type == 'Hyb-PGb':
        Np = model_pars['Np']
        Ny = model_pars['Ny']
        xs = np.concatenate((np.tile(plant_pars['xs'][:Ny], (Np+1, )), 
                             np.tile(plant_pars['us'], (Np, ))))
    else:
        None
    # Solve a steady-state equality problem to get 
    # an updated xs corresponding to exact equality constraint.
    xs, _, _ = getXsYsSscost(fxu=fxu, hx=hx, us=us, 
                             parameters=model_pars, xguess=xs)
    # Return.
    return xs

def getBestSSOptimum(*, fxu, hx, lxu, model_pars, 
                        plant_pars, Nguess):
    """ Solve the steady-state optimization for multiple initial guesses. 
        and return the best solution. 

        i.e, Heuristic method to find the global optimum.   
    """

    # Create empty lists to store solutions.
    xs_list, us_list, ssCost_list = [], [], []

    # Input constraint limits. 
    ulb, uub = model_pars['ulb'], model_pars['uub']

    # Create variables to store the best xs, us, and sscosts. 
    bestXs, bestUs, bestSsCost = None, None, None

    # Solve optimization problems for multiple initial guesses.
    for i in range(Nguess):

        # Generate a random us within the input constraint
        # limit and corresponding xs.
        us = (uub - ulb)*np.random.rand(Nu) + ulb

        # Get the corresponding xs guess.
        xs = getXGuess(model_type=model_type, fxu=fxu, hx=hx, us=us, 
                       model_pars=model_pars, plant_pars=plant_pars)

        # XUguess dictionary. 
        xuguess = dict(x=xs, u=us)

        # Solve the optimization with that initial guess.
        xs, us, _, ssCost = getSSOptimum(fxu=fxu, hx=hx, lxu=lxu,
                                         parameters=model_pars,
                                         guess=xuguess)

        # Update the best solution. 
        if i > 0 and ssCost[0] < np.max(ssCost_list):
            bestIndex = np.argmax(ssCost_list)
            bestXs = xs_list[bestIndex]
            bestUs = us_list[bestIndex]
            bestSsCost = ssCost_list[bestIndex]
        elif i == 0:
            bestXs = xs
            bestUs = us
            bestSsCost = ssCost
        else:
            pass

    # Return.
    return bestXs, bestUs, bestSsCost

def doOptimizationAnalysis(*, model_type, fxu_list, hx_list, par_list, 
                              lxu, plb, pub, Npvals, Nguess):
    """ Generate a random set of cost parameters, compute optimum of all the
        models and deviation of the optimum input from the plant optimum and the
        suboptimality gap. 
    """

    # Create lists to store the all the optimization results/sub gaps.
    xs_list, us_list = [], []
    optCosts_list, subGaps_list = [], []

    # Generate a set of cost parameters.
    pvals = list((pub-plb)*np.random.rand(Npvals, len(plb)) + plb)

    # Extract the plant function handles. 
    plant_f, plant_h, plant_pars = fxu_list[0], hx_list[0], par_list[0]

    # Loop over all the models.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list,
                                                 hx_list, par_list):

        # Generate lists to store the optimization results 
        # and suboptimality gaps for a model.
        model_xs, model_us = [], []
        model_optCosts, model_subGaps = [], []

        # Loop over all the parameter values.
        for i, p in enumerate(pvals):
            
            # Get the best steady-state optimum.
            # Best refers to simply solve the optimization multiple times.
            xs, us, ys, optCost = getBestSSOptimum(fxu=fxu, hx=hx, lxu=lxu, 
                                                   model_pars=model_pars, 
                                                   plant_pars=plant_pars,
                                                   Nguess=Nguess)

            # Store result.
            model_xs += [xs]
            model_us += [us]
            model_optCosts += [optCost]

            # Compute suboptimality gaps if the model is not a plant model.
            if model_type != 'Plant':
                
                # Compute the cost incurred when the plant is operated at 
                # the model's optimum.
                _, _, plantSsCost = getXsYsSscost(fxu=plant_f, 
                                                  hx=plant_h, us=us, 
                                                  parameters=plant_pars, 
                                                  xguess=plant_pars['xs'])

                # Plant optimum cost. 
                plantOptCost = optCosts_list[0][i, :]

                # Compute the suboptimality gap. 
                subGap = np.abs(plantOptCost - plantSsCost)
                subGap = subGap/np.abs(plantOptCost)
                model_subGaps += [subGap]

            else:

                model_subGaps += [np.array([np.nan])]

        # Store steady-state xs, us, optimum costs, and suboptimality gaps.
        # in lists. 
        xs_list += [np.array(model_xs)]
        us_list += [np.array(model_us)]
        optCosts_list += [np.array(model_optCosts)]
        subGaps_list += [np.array(model_subGaps)]

    # Create data object and save.
    reac_ssopt = dict(xs=xs_list, us=us_list, 
                      optCosts=optCosts_list, subGaps=subGaps_list)

    # Return. 
    return reac_ssopt

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

    ## Optimization analysis for the cost type 1 without a Cc contribution.
    # Get lists of model types.
    model_types = ['Plant', 'Black-Box-NN', 'Hyb-FGb', 'Hyb-PGb']
    fxu_list = [plant_f, bbnn_f, fhyb_f, phyb_f]
    hx_list = [plant_h, bbnn_h, fhyb_h, phyb_h]
    par_list = [plant_pars, bbnn_pars, fhyb_pars, phyb_pars]
    # Lower and upper bounds of cost parameters. 
    plb = np.array([50, 500])
    pub = np.array([200, 1500])
    Npvals = 10

    reac_optanalysis = doOptimizationAnalysis(model_types=model_types, 
                                        fxu_list=fxu_list, hx_list=hx_list, 
                                        par_list=par_list, plb=plb, pub=pub, 
                                        Npvals=Npvals, Nguess=Nguess)


    ## Optimization analysis for the cost type 1 with a Cc contribution.

    # Save.
    PickleTool.save(data_object=reac_ssopt,
                    filename='reac_ssopt_optimizationanalysis.pickle')

main()