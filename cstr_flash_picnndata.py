# [depends] %LIB%/hybridid.py %LIB%/cstr_flash_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/CstrFlashHybridFuncs.py
# [depends] %LIB%/InputConvexFuncs.py
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
from economicopt import get_xs_sscost
from CstrFlashHybridFuncs import get_hybrid_pars, hybrid_fxu, hybrid_hx
from InputConvexFuncs import generate_picnn_data

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                        'cstr_flash_parameters.pickle',
                                         type='read')
    # cstr_flash_bbnntrain = PickleTool.load(filename=
    #                                     'cstr_flash_bbnntrain.pickle',
    #                                      type='read')
    cstr_flash_hybtrain = PickleTool.load(filename=
                                        'cstr_flash_hybtrain.pickle',
                                         type='read')

    # Get plant and grey-box parameters.
    hyb_greybox_pars = cstr_flash_parameters['hyb_greybox_pars']
    plant_pars = cstr_flash_parameters['plant_pars']

    # Get the black-box model parameters and function handles.
    # bbnn_pars = get_bbnn_pars(train=cstr_flash_bbnntrain, 
    #                           plant_pars=plant_pars)
    # bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    # bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Get hybrid model parameters and function handles.
    hyb_pars = get_hybrid_pars(train=cstr_flash_hybtrain, 
                               hyb_greybox_pars=hyb_greybox_pars)
    hyb_fxu = lambda x, u: hybrid_fxu(x, u, hyb_pars)
    hyb_hx = hybrid_hx

    # Cost.
    cost_yup = lambda y, u: cost_yup(y, u, p, plant_pars)

    # Range of economic parameters.

    # Get xGuess.
    ys = plant_pars['xs'][plant_pars['yindices']]
    us = np.array([10., 5.])
    xguess = ys

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