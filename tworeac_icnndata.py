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
from tworeac_funcs import cost_yup
from economicopt import get_sscost
from TwoReacHybridFuncs import (tworeacHybrid_fxu,
                                tworeacHybrid_hx,
                                get_tworeacHybrid_pars)         

def generate_data(*, hyb_fxu, hyb_hx, hyb_pars, seed=10):
    """ Function to generate data to train the ICNN. """

    # Set numpy seed.
    np.random.seed(seed)
    
    # Get cost function.
    p = [100, 200]
    cost_yu = lambda y, u: cost_yup(y, u, p)

    # Get a list of random inputs.
    Nu = hyb_pars['Nu']
    ulb, uub = hyb_pars['ulb'], hyb_pars['uub']
    Ndata = 1000
    us_list = list((uub-ulb)*np.random.rand(Ndata, Nu) + ulb)

    # Get the corresponding steady state costs.
    ss_costs = []
    for us in us_list:
        ss_cost = get_sscost(fxu=hyb_fxu, hx=hyb_hx, lyu=cost_yu, 
                             us=us, parameters=hyb_pars)
        ss_costs += [ss_cost]
    lyu = np.array(ss_costs)
    u = np.array(us_list)

    # Return.
    return u, lyu

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                        'tworeac_hybtrain.pickle',
                                         type='read')
    greybox_pars = tworeac_parameters['greybox_pars']

    # Get the black-box model parameters and function handles.
    hyb_pars = get_tworeacHybrid_pars(train=tworeac_hybtrain, 
                                      greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: tworeacHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: tworeacHybrid_hx(x)

    # Generate data.
    u, lyup = generate_data(hyb_fxu=hyb_fxu, hyb_hx=hyb_hx, 
                            hyb_pars=hyb_pars)    
    
    # Get data in dictionary.
    tworeac_icnndata = dict(u=u, lyup=lyup)

    # Save data.
    PickleTool.save(data_object=tworeac_icnndata,
                    filename='tworeac_icnndata.pickle')

main()