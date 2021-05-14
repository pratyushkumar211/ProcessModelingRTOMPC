# [depends] %LIB%/plotting_funcs.py %LIB%/hybridid.py
# [depends] %LIB%/BlackBoxFuncs.py %LIB%/TwoReacHybridFuncs.py
# [depends] %LIB%/economicopt.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/InputConvexFuncs.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_bbNNtrain.pickle
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
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from TwoReacHybridFuncs import (get_tworeacHybrid_pars, 
                                tworeacHybrid_fxu, 
                               tworeacHybrid_hx)
from economicopt import (get_ss_optimum, c2dNonlin, 
                         get_sscost)
from tworeac_funcs import cost_yup, plant_ode
from InputConvexFuncs import get_ss_optimum as get_icnn_ss_optimum
from InputConvexFuncs import get_icnn_pars, icnn_lyu

def get_xuguess(*, model_type, plant_pars, Np=None):
    """ Get x, u guess depending on model type. """
    us = plant_pars['us']
    if model_type == 'Plant' or model_type == 'Hybrid':
        xs = plant_pars['xs']
    elif model_type == 'Black-Box-NN':
        yindices = plant_pars['yindices']
        ys = plant_pars['xs'][yindices]
        us = np.array([0.5])
        xs = np.concatenate((np.tile(ys, (Np+1, )), 
                             np.tile(us, (Np, ))))
    elif model_type == 'ICNN':
        us = np.array([1.5])
        xs = None
    else:
        pass
    # Return as dict.
    return dict(x=xs, u=us)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters.pickle',
                                         type='read')
    tworeac_bbnntrain = PickleTool.load(filename=
                                    'tworeac_bbnntrain.pickle',
                                      type='read')
    tworeac_icnntrain = PickleTool.load(filename=
                                    'tworeac_icnntrain.pickle',
                                      type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                      'tworeac_hybtrain.pickle',
                                      type='read')

    # Get plant and grey-box parameters. 
    plant_pars = tworeac_parameters['plant_pars']
    greybox_pars = tworeac_parameters['greybox_pars']

    # Get cost function handle.
    p = [100, 200]
    lyu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    bbnn_pars = get_bbnn_pars(train=tworeac_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Get the black-box model parameters and function handles.
    hyb_pars = get_tworeacHybrid_pars(train=tworeac_hybtrain, 
                                      greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: tworeacHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: tworeacHybrid_hx(x)

    # Get ICNN pars and function.
    icnn_pars = get_icnn_pars(train=tworeac_icnntrain, plant_pars=plant_pars)
    icnn_lu = lambda u: icnn_lyu(u, icnn_pars)
    
    # Get the plant function handle.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN', 'ICNN']
    fxu_list = [plant_fxu, bbnn_f, None]
    hx_list = [plant_hx, bbnn_h, None]
    par_list = [plant_pars, bbnn_pars, None]
    Nps = [None, bbnn_pars['Np'], None]

    # Loop over the different models, and obtain SS optimums.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np)
        
        # Get the steady state optimum.
        if model_type != 'ICNN':
            xs, us, ys = get_ss_optimum(fxu=fxu, hx=hx, lyu=lyu, 
                                        parameters=model_pars, guess=xuguess)
        else:
            us = get_icnn_ss_optimum(lyu=icnn_lu, parameters=icnn_pars, 
                                      uguess=xuguess['u'])
        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))

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
            
            if model_type != 'ICNN':
                sscost = get_sscost(fxu=fxu, hx=hx, lyu=lyu, 
                                    us=us, parameters=model_pars)
            else:
                sscost = icnn_lu(us)
            model_sscost += [sscost]
        
        model_sscost = np.asarray(model_sscost)
        sscosts += [model_sscost]

    # Get us as rank 1 array.
    us = np.asarray(us_list)[:, 0]

    legend_names = model_types
    legend_colors = ['b', 'dimgrey', 'tomato']
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