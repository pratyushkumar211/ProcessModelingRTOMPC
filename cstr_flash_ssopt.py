# [depends] %LIB%/hybridid.py %LIB%/cstr_flash_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/plotting_funcs.py
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_bbtrain.pickle
""" Script to use the trained hybrid model for 
    steady-state optimization.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import casadi
import copy
import matplotlib.pyplot as plt
import mpctools as mpc
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import PickleTool, measurement
from economicopt import get_bbpars_fxu_hx, get_sscost, get_kooppars_fxu_hx
from economicopt import get_ss_optimum, get_xuguess, c2dNonlin
from cstr_flash_funcs import cost_yup, plant_ode, greybox_ode
from plotting_funcs import TwoReacPlots, PAPER_FIGSIZE

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                        'cstr_flash_parameters.pickle',
                                         type='read')
    plant_pars = cstr_flash_parameters['plant_pars']
    cstr_flash_bbtrain = PickleTool.load(filename='cstr_flash_bbtrain.pickle',
                                      type='read')

    # Get cost function handle.
    p = [10, 2000, 14000]
    lyu = lambda y, u: cost_yup(y, u, p, plant_pars)

    # Get the black-box model parameters and function handles.
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=
                                                       cstr_flash_bbtrain, 
                                                       parameters=plant_pars)

    # Get the plant function handle.
    Delta = plant_pars['Delta']
    ps = plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Lists to loop over for different models.
    model_types = ['plant', 'black-box']
    fxu_list = [plant_fxu, blackb_fxu]
    hx_list = [plant_hx, blackb_hx]
    par_list = [plant_pars, bb_pars]
    Nps = [None, bb_pars['Np']]

    # Loop over the different models, and obtain SS optimums.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np, 
                              Nx = model_pars['Nx'])

        # Get the steady state optimum.
        xs, us, ys = get_ss_optimum(fxu=fxu, hx=hx, lyu=lyu, 
                                    parameters=model_pars, guess=xuguess)

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
        print('ys: ' + str(ys))


main()