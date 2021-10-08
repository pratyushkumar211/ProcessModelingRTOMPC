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

# Functions for Black-Box, full hybrid, and partial hybrid models.
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

    # Get cost function handle.
    p = [100, 1200]
    lyu = lambda y, u: cost_yup(y, u, p)

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
    model_types = ['Plant', 'Black-Box-NN', 'Hyb-FGb', 'Hyb-PGb']
    fxu_list = [plant_f, bbnn_f, fhyb_f, phyb_f]
    hx_list = [plant_h, bbnn_h, fhyb_h, phyb_h]
    par_list = [plant_pars, bbnn_pars, fhyb_pars, phyb_pars]

    # Loop over the different models and obtain SS optimums.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # Get Guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, 
                              model_pars=model_pars)
        
        # Get the steady state optimum.
        xs, us, ys, opt_sscost = getSSOptimum(fxu=fxu, hx=hx, lyu=lyu, 
                                              parameters=model_pars, 
                                              guess=xuguess)

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
        
main()
