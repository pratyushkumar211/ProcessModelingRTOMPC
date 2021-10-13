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
from plottingFuncs import PAPER_FIGSIZE
from hybridId import PickleTool
from linNonlinMPC import (c2dNonlin, getSSOptimum,
                          getXsYsSscost, getXsUsSSCalcGuess)
from reacFuncs import cost_lxup_noCc, cost_lxup_withCc, plant_ode

# Import respective function handles.
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from ReacHybridFullGbFuncs import get_hybrid_pars as get_fhyb_pars
from ReacHybridFullGbFuncs import hybrid_fxup as fhyb_fxup
from ReacHybridFullGbFuncs import hybrid_hx as fhyb_hx
from ReacHybridPartialGbFuncs import get_hybrid_pars as get_phyb_pars
from ReacHybridPartialGbFuncs import hybrid_fxup as phyb_fxup
from ReacHybridPartialGbFuncs import hybrid_hx as phyb_hx

def get_xguess(*, model_type, plant_pars, model_pars):
    """ Get x guess based on model type. """    
    if model_type == 'Plant':
        xs = plant_pars['xs']
    elif model_type == 'Black-Box-NN':
        Np = model_pars['Np']
        Ny = model_pars['Ny']
        yindices = plant_pars['yindices']
        xs = np.concatenate((np.tile(plant_pars['xs'][yindices], (Np, )), 
                             np.tile(plant_pars['us'], (Np, ))))
    elif model_type == 'Hybrid-FullGb':
        xs = plant_pars['xs']
    elif model_type == 'Hybrid-PartialGb':
        Np = model_pars['Np']
        Ny = model_pars['Ny']
        yindices = plant_pars['yindices']
        xs = np.concatenate((np.tile(plant_pars['xs'][yindices], (Np+1, )), 
                             np.tile(plant_pars['us'], (Np, ))))
    else:
        None
    # Return.
    return xs

def getCostCurveData(*, Nus, model_types, fxu_list, hx_list, 
                        par_list, plant_pars, cost_lxup):
    """ Vary the steady-state input (us) and compute the corresponding
        xs and cost value (\ell).
    """

    # Get a linspace of steady-state u values.
    ulb, uub = plant_pars['ulb'], plant_pars['uub']
    us_list = list(np.linspace(ulb, uub, Nus))

    # Create lists to store xs and sscosts. 
    xs_list, sscosts = [], []

    # Loop over all the models.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # Lists to store xs and sscost for a given model.
        model_xs = []
        model_sscost = []

        # Compute SS cost.
        for us in us_list:
            
            # Get guess.
            xguess = get_xguess(model_type=model_type, 
                                plant_pars=plant_pars, 
                                model_pars=model_pars)
            
            # Get the xs and sscost.
            xs, _, sscost = getXsYsSscost(fxu=fxu, hx=hx, lxu=cost_lxup, 
                                          us=us, parameters=model_pars, 
                                          xguess=xguess)
            model_xs += [xs]
            model_sscost += [sscost]

        # Model steady states and costs.
        model_xs = np.asarray(model_xs)
        model_sscost = np.asarray(model_sscost)

        # Store steady states and costs in lists.
        xs_list += [model_xs]
        sscosts += [model_sscost]

    # Convert to a rank 1 array.
    us = np.asarray(us_list)[:, 0]

    # Create a dictionary and save.
    reac_ssopt = dict(us=us, xs=xs_list, sscosts=sscosts)

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

    # Extract out the training data for analysis. 
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

    # Cost function handle.
    p = [100, 900]
    cost_lxup = lambda x, u: cost_lxup_noCc(x, u, p)
    # Number of ss inputs at which to compute the cost.
    Nus = 100
    reac_ssopt = getCostCurveData(Nus=Nus, model_types=model_types,
                                  fxu_list=fxu_list, hx_list=hx_list,
                                  par_list=par_list, plant_pars=plant_pars, 
                                  cost_lxup=cost_lxup)

    # Append NaNs for plotting the steady-state cost curves of the 
    # Black-Box NN and Hybrid Partial Gb model.
    yindices = plant_pars['yindices']
    reac_ssopt['xs'][1] = np.concatenate((reac_ssopt['xs'][1][:, yindices], 
                                          np.tile(np.nan, (Nus, 1))), axis=-1)
    reac_ssopt['xs'][3] = np.concatenate((reac_ssopt['xs'][3][:, yindices], 
                                          np.tile(np.nan, (Nus, 1))), axis=-1)
    reac_ssopt_list = [reac_ssopt]

    # Do the cost curve computation for the other cost function.
    # Model/fxu/hx/and par-lists. 
    model_types = ['Plant', 'Hybrid-FullGb']
    fxu_list = [plant_f, fhyb_f]
    hx_list = [plant_h, fhyb_h]
    par_list = [plant_pars, fhyb_pars]
    # Cost function handle.
    p = [100, 1000, 100]
    cost_lxup = lambda x, u: cost_lxup_withCc(x, u, p)
    # Number of ss inputs at which to compute the cost.
    Nus = 100
    reac_ssopt = getCostCurveData(Nus=Nus, model_types=model_types,
                                  fxu_list=fxu_list, hx_list=hx_list,
                                  par_list=par_list, plant_pars=plant_pars, 
                                  cost_lxup=cost_lxup)
    
    # Save data. 
    reac_ssopt_list += [reac_ssopt]

    # Save.
    PickleTool.save(data_object=reac_ssopt_list,
                    filename='reac_ssopt_curve.pickle')

main()