# [depends] %LIB%/hybridid.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/TwoReacHybridFuncs.py
# [depends] %LIB%/InputConvexFuncs.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_hybtrain.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hybridid import PickleTool 
from tworeac_funcs import cost_yup
from economicopt import get_xs_sscost
from InputConvexFuncs import generate_picnn_data
from TwoReacHybridFuncs import (hybrid_fxup, hybrid_hx,
                                get_hybrid_pars)         
# from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    # tworeac_bbNNtrain = PickleTool.load(filename=
    #                                     'tworeac_bbnntrain.pickle',
    #                                      type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                        'tworeac_hybtrain.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']
    hyb_greybox_pars = tworeac_parameters['hyb_greybox_pars']

    # Get the Hybrid model parameters and function handles.
    hyb_pars = get_hybrid_pars(train=tworeac_hybtrain, 
                               hyb_greybox_pars=hyb_greybox_pars)
    ps = hyb_pars['ps']
    hyb_fxu = lambda x, u: hybrid_fxup(x, u, ps, hyb_pars)
    hyb_hx = lambda x: hybrid_hx(x)

    # Get the black-box model parameters and function handles.
    # bbnn_pars = get_bbnn_pars(train=tworeac_bbNNtrain, 
    #                           plant_pars=plant_pars)
    # bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    # bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Range of economic parameters.
    plb = np.array([100., 100.])
    pub = np.array([100., 600.])
    Nsamp_us = 50
    Nsamp_p = 50

    # Generate data.
    p, u, lyup = generate_picnn_data(fxup=hyb_fxu, hx=hyb_hx, 
                                    model_pars=hyb_pars, cost_yup=cost_yup, 
                                    Nsamp_us=Nsamp_us, plb=plb, pub=pub,
                                    Nsamp_p=Nsamp_p) 
    
    # Get data in dictionary.
    p = p[:, 1:2]
    tworeac_picnndata = dict(p=p, u=u, lyup=lyup)

    # Save data.
    PickleTool.save(data_object=tworeac_picnndata,
                    filename='tworeac_picnndata.pickle')

main()