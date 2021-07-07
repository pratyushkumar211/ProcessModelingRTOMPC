# [depends] %LIB%/hybridid.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/TwoReacHybridFuncs.py
# [depends] %LIB%/InputConvexFuncs.py %LIB%/BlackBoxFuncs.py
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
from tworeac_funcs import cost_yup
from TwoReacHybridFuncs import (hybrid_fxu, hybrid_hx,
                                get_hybrid_pars)         
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from InputConvexFuncs import generate_icnn_data

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    tworeac_bbNNtrain = PickleTool.load(filename=
                                        'tworeac_bbnntrain.pickle',
                                         type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                        'tworeac_hybtrain.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']
    hyb_greybox_pars = tworeac_parameters['hyb_greybox_pars']

    # Get the Hybrid model parameters and function handles.
    hyb_pars = get_hybrid_pars(train=tworeac_hybtrain, 
                               hyb_greybox_pars=hyb_greybox_pars)
    hyb_fxu = lambda x, u: hybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: hybrid_hx(x)

    # Get the cost function handle.
    p = [100, 300]
    cost_yu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    # bbnn_pars = get_bbnn_pars(train=tworeac_bbNNtrain, 
    #                           plant_pars=plant_pars)
    # bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    # bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Generate data.
    xguess = plant_pars['xs']
    Ndata = 2000
    u, lyup = generate_icnn_data(fxu=hyb_fxu, hx=hyb_hx, cost_yu=cost_yu, 
                                 parameters=hyb_pars, Ndata=Ndata, 
                                 xguess=xguess)
    
    # Get data in dictionary.
    tworeac_icnndata = dict(u=u, lyup=lyup)

    # Save data.
    PickleTool.save(data_object=tworeac_icnndata,
                    filename='tworeac_icnndata.pickle')

main()