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
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from linNonlinMPC import c2dNonlin, getSSOptimum, getXsYsSscost
from reacFuncs import cost_yup, plant_ode

from ReacHybridFullGbFuncs import get_hybrid_pars as get_fullhybrid_pars
from ReacHybridFullGbFuncs import hybrid_fxup as fullhybrid_fxup
from ReacHybridFullGbFuncs import hybrid_hx as fullhybrid_hx

from ReacHybridPartialGbFuncs import get_hybrid_pars as get_partialhybrid_pars
from ReacHybridPartialGbFuncs import hybrid_fxup as partialhybrid_fxup
from ReacHybridPartialGbFuncs import hybrid_hx as partialhybrid_hx

def get_xuguess(*, model_type, plant_pars, model_pars):
    """ Get x and u guesses depending on model type. """
    
    us = plant_pars['us']

    if model_type == 'Plant':

        xs = plant_pars['xs']

    elif model_type == 'Black-Box-NN':
        
        Np = model_pars['Np']
        Ny = model_pars['Ny']
        xs = np.concatenate((np.tile(plant_pars['xs'][:Ny], (Np+1)), 
                             np.tile(plant_pars['us'], (Np))))

    elif model_type == 'Hybrid-1':

        xs = plant_pars['xs']

    elif model_type == 'Hybrid-2':

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

    # Get plant and grey-box parameters.
    plant_pars = reac_parameters['plant_pars']
    hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']
    hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']

    # Get cost function handle.
    p = [100, 1000]
    lyu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    bbnn_pars = get_bbnn_pars(train=reac_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Get the Full hybrid model parameters and function handles.
    fullhyb_pars = get_fullhybrid_pars(train=reac_hybfullgbtrain, 
                                   hyb_fullgb_pars=hyb_fullgb_pars)
    ps = fullhyb_pars['ps']
    fullhybrid_f = lambda x, u: fullhybrid_fxup(x, u, ps, fullhyb_pars)
    fullhybrid_h = lambda x: fullhybrid_hx(x, fullhyb_pars)
    
    # Get the partial hybrid model parameters and function handles.
    partialhyb_pars = get_partialhybrid_pars(train=reac_hybpartialgbtrain, 
                                        hyb_partialgb_pars=hyb_partialgb_pars)
    ps = partialhyb_pars['ps']
    partialhybrid_f = lambda x, u: partialhybrid_fxup(x, u, ps, partialhyb_pars)
    partialhybrid_h = lambda x: partialhybrid_hx(x, partialhyb_pars)

    # Get the plant function handle.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_f = c2dNonlin(plant_fxu, Delta)
    plant_h = lambda x: x[plant_pars['yindices']]

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN', 'Hybrid-1', 'Hybrid-2']
    fxu_list = [plant_f, bbnn_f, fullhybrid_f, partialhybrid_f]
    hx_list = [plant_h, bbnn_h, fullhybrid_h, partialhybrid_h]
    par_list = [plant_pars, bbnn_pars, fullhyb_pars, partialhyb_pars]

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
