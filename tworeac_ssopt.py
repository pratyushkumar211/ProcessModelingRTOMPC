# [depends] %LIB%/hybridid.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/plotting_funcs.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_bbtrain.pickle
# [depends] tworeac_kooptrain.pickle
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
from tworeac_funcs import cost_yup, plant_ode, greybox_ode
from plotting_funcs import TwoReacPlots, PAPER_FIGSIZE

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    tworeac_bbtrain = PickleTool.load(filename='tworeac_bbtrain.pickle',
                                      type='read')
    tworeac_kooptrain = PickleTool.load(filename='tworeac_kooptrain.pickle',
                                      type='read')

    # Get cost function handle.
    p = [100, 200]
    lyu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=tworeac_bbtrain, 
                                                       parameters=parameters)

    # Get the Koopman model parameters and function handles.
    koop_pars, koop_fxu, koop_hx = get_kooppars_fxu_hx(train=tworeac_kooptrain, 
                                                       parameters=parameters)

    # Get the plant function handle.
    Delta = parameters['Delta']
    ps = parameters['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, parameters)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, parameters)

    # Get the grey-box function handle.
    gb_fxu = lambda x, u: greybox_ode(x, u, ps, parameters)
    gb_fxu = c2dNonlin(gb_fxu, Delta)
    gb_pars = copy.deepcopy(parameters)
    gb_pars['Nx'] = len(parameters['gb_indices'])

    # Lists to loop over for different models.
    model_types = ['plant', 'grey-box', 'black-box', 'Koopman']
    fxu_list = [plant_fxu, gb_fxu, blackb_fxu, koop_fxu]
    hx_list = [plant_hx, plant_hx, blackb_hx, koop_hx]
    par_list = [parameters, gb_pars, bb_pars, koop_pars]
    Nps = [None, None, bb_pars['Np'], koop_pars['Np']]

    # Loop over the different models, and obtain SS optimums.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=parameters, Np=Np, 
                              Nx = model_pars['Nx'])

        # Get the steady state optimum.
        xs, us, ys = get_ss_optimum(fxu=fxu, hx=hx, lyu=lyu, 
                                    parameters=model_pars, guess=xuguess)

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
        print('ys: ' + str(ys))

    # Get a linspace of steady-state u values.
    ulb, uub = parameters['ulb'], parameters['uub']
    us_list = list(np.linspace(ulb, uub, 100))

    # Lists to store Steady-state cost.
    sscosts = []

    # Loop over all the models.
    for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
                                                 hx_list, par_list):

        # List to store SS costs for one model.
        model_sscost = []

        # Compute SS cost.
        for us in us_list:

            sscost = get_sscost(fxu=fxu, hx=hx, lyu=lyu, 
                                us=us, parameters=model_pars)
            model_sscost += [sscost]
        
        model_sscost = np.asarray(model_sscost)
        sscosts += [model_sscost]

    # Get us as rank 1 array.
    us = np.asarray(us_list)[:, 0]

    legend_names = ['Plant', 'Grey-box', 'Black-box', 'Koopman']
    legend_colors = ['b', 'g', 'dimgrey', 'm']
    figures = TwoReacPlots.plot_sscosts(us=us, sscosts=sscosts, 
                                        legend_colors=legend_colors, 
                                        legend_names=legend_names, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.12, 
                                        left_label_frac=0.15, 
                                        font_size=12)

    # Finally plot.
    with PdfPages('tworeac_ssopt.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()