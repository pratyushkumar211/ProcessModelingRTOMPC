# [depends] %LIB%/hybridid.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/TwoReacHybridFuncs.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_hybtrain.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import numpy as np
from hybridid import PickleTool
from cstr_flash_funcs import cost_yup
from economicopt import get_sscost
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from CstrFlashHybridFuncs import (get_CstrFlash_hybrid_pars, 
                                  CstrFlashHybrid_fxu, 
                                  CstrFlashHybrid_hx)

def generate_data(*, fxu, hx, cost_yu, parameters, seed=10, 
                     xguess=None):
    """ Function to generate data to train the ICNN. """

    # Set numpy seed.
    np.random.seed(seed)

    # Get a list of random inputs.
    Nu = parameters['Nu']
    ulb, uub = parameters['ulb'], parameters['uub']
    Ndata = 4500
    us_list = list((uub-ulb)*np.random.rand(Ndata, Nu) + ulb)

    # Get the corresponding steady state costs.
    ss_costs = []
    for us in us_list:
        ss_cost = get_sscost(fxu=fxu, hx=hx, lyu=cost_yu, 
                             us=us, parameters=parameters, 
                             xguess=xguess)
        ss_costs += [ss_cost]
    lyu = np.array(ss_costs)
    u = np.array(us_list)
    
    # Get rid of NaNs in sscost.
    u = np.delete(u, np.where(np.isnan(lyu)), axis=0)
    lyu = np.delete(lyu, np.where(np.isnan(lyu)), axis=0)

    # Return.
    return u, lyu

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                        'cstr_flash_parameters.pickle',
                                         type='read')
    cstr_flash_bbnntrain = PickleTool.load(filename=
                                        'cstr_flash_bbnntrain.pickle',
                                         type='read')
    cstr_flash_hybtrain = PickleTool.load(filename=
                                        'cstr_flash_hybtrain.pickle',
                                         type='read')

    # Get plant and grey-box parameters.
    greybox_pars = cstr_flash_parameters['greybox_pars']
    plant_pars = cstr_flash_parameters['plant_pars']

    # Get the black-box model parameters and function handles.
    bbnn_pars = get_bbnn_pars(train=cstr_flash_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Get hybrid model parameters and function handles.
    hyb_pars = get_CstrFlash_hybrid_pars(train=cstr_flash_hybtrain, 
                                         greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: CstrFlashHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: CstrFlashHybrid_hx(x, hyb_pars)

    # Cost.
    p = [30, 2200, 10000]
    cost_yu = lambda y, u: cost_yup(y, u, p, plant_pars)
    
    # Get xGuess.
    ys = plant_pars['xs'][plant_pars['yindices']]
    us = np.array([10., 5.])
    Np = hyb_pars['Np']
    xguess = np.concatenate((np.tile(ys, (Np+1, )), 
                             np.tile(us, (Np, ))))


    # Generate data.
    u, lyup = generate_data(fxu=hyb_fxu, hx=hyb_hx, 
                            cost_yu=cost_yu,
                            parameters=hyb_pars, 
                            xguess=xguess)
    
    # Get data in dictionary.
    cstr_flash_icnndata = dict(u=u, lyup=lyup)

    # Save data.
    PickleTool.save(data_object=cstr_flash_icnndata,
                    filename='cstr_flash_icnndata.pickle')

main()