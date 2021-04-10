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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotting_funcs import TwoReacPlots, PAPER_FIGSIZE
from hybridid import PickleTool, measurement
from BlackBoxFuncs import get_bbNN_pars, bbNN_fxu, bb_hx
from TwoReacHybridFuncs import (get_tworeacHybrid_pars, tworeacHybrid_fxu, 
                               tworeacHybrid_hx)
from economicopt import (get_ss_optimum, get_xuguess, c2dNonlin, 
                         get_sscost)
from tworeac_funcs import cost_yup, plant_ode

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']
    greybox_pars = tworeac_parameters['greybox_pars']
    tworeac_bbNNtrain = PickleTool.load(filename='tworeac_bbNNtrain.pickle',
                                      type='read')
    tworeac_hybtrain = PickleTool.load(filename='tworeac_hybtrain.pickle',
                                      type='read')

    # Get cost function handle.
    p = [100, 220]
    lyu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    bbNN_pars = get_bbNN_pars(train=tworeac_bbNNtrain, 
                              plant_pars=plant_pars)
    bbNN_Fxu = lambda x, u: bbNN_fxu(x, u, bbNN_pars)
    bbNN_hx = lambda x: bb_hx(x, bbNN_pars)

    # Get the black-box model parameters and function handles.
    hyb_pars = get_tworeacHybrid_pars(train=tworeac_hybtrain, 
                                      greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: tworeacHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: tworeacHybrid_hx(x)

    # Get the plant function handle.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN', 'Hybrid']
    fxu_list = [plant_fxu, bbNN_Fxu, hyb_fxu]
    hx_list = [plant_hx, bbNN_hx, hyb_hx]
    par_list = [plant_pars, bbNN_pars, hyb_pars]
    Nps = [None, bbNN_pars['Np'], None]

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

    # Get a linspace of steady-state u values.
    ulb, uub = plant_pars['ulb'], plant_pars['uub']
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

    legend_names = ['Plant', 'Black-box-NN', 'Hybrid']
    legend_colors = ['b', 'dimgrey', 'm']
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