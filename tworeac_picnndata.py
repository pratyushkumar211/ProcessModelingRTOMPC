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
from TwoReacHybridFuncs import (tworeacHybrid_fxu,
                                tworeacHybrid_hx,
                                get_tworeacHybrid_pars)         
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
    greybox_pars = tworeac_parameters['greybox_pars']

    # Get the Hybrid model parameters and function handles.
    hyb_pars = get_tworeacHybrid_pars(train=tworeac_hybtrain, 
                                      greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: tworeacHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: tworeacHybrid_hx(x)

    # Get the black-box model parameters and function handles.
    # bbnn_pars = get_bbnn_pars(train=tworeac_bbNNtrain, 
    #                           plant_pars=plant_pars)
    # bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    # bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Generate data.
    p, u, lyup = generate_picnn_data(fxup=hyb_fxup, hx=hx, 
                                    model_pars=hyb_pars, cost_yup=cost_yup, 
                                    Nsamp_us=Nsamp_us, plb=plb, pub=pub,
                                    Nsamp_p=Nsamp_p) 
    
    # Get data in dictionary.
    tworeac_icnndata = dict(u=u, lyup=lyup)

    # Save data.
    PickleTool.save(data_object=tworeac_icnndata,
                    filename='tworeac_icnndata.pickle')

main()