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
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import PickleTool, measurement
from economicopt import get_sscost, get_ss_optimum, c2dNonlin, get_xuguess
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from cstr_flash_funcs import cost_yup, plant_ode, greybox_ode
from plotting_funcs import CstrFlashPlots, PAPER_FIGSIZE

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                        'cstr_flash_parameters.pickle',
                                         type='read')
    plant_pars = cstr_flash_parameters['plant_pars']
    cstr_flash_bbnntrain = PickleTool.load(filename=
                                     'cstr_flash_bbnntrain.pickle',
                                      type='read')

    # Get cost function handle.
    p = [10, 2000, 18000]
    lyu = lambda y, u: cost_yup(y, u, p, plant_pars)

    # Get the plant function handle.
    Delta = plant_pars['Delta']
    ps = plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Get the black-box model parameters and function handles.
    bbnn_pars = get_bbnn_pars(train=cstr_flash_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN']
    fxu_list = [plant_fxu, bbnn_f]
    hx_list = [plant_hx, bbnn_h]
    par_list = [plant_pars, bbnn_pars]
    Nps = [None, bbnn_pars['Np']]

    # Loop over the different models, and obtain SS optimums.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np)

        # Get the steady state optimum.
        xs, us, ys = get_ss_optimum(fxu=fxu, hx=hx, lyu=lyu, 
                                    parameters=model_pars, guess=xuguess)

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
        print('ys: ' + str(ys))

    # # Get a linspace of steady-state u values.
    # Nu = plant_pars['Nu']
    # ulb, uub = plant_pars['ulb'], plant_pars['uub']

    # # Get lists.
    # us1_list = list(np.linspace(ulb[0], uub[0], 2))
    # us2_list = list(np.linspace(ulb[1], uub[1], 2))

    # # Lists to store Steady-state cost.
    # sscosts = []

    # # Loop over all the models.
    # for (model_type, fxu, hx, model_pars) in zip(model_types, fxu_list, 
    #                                              hx_list, par_list):

    #     # List to store SS costs for one model.
    #     model_sscost = []

    #     # Get guess.
    #     xuguess = get_xuguess(model_type=model_type, 
    #                           plant_pars=plant_pars, Np=Np)

    #     # Compute SS cost.
    #     for us1, us2 in itertools.product(us1_list, us2_list):
            
    #         # Get the vector input.
    #         us = np.array([us1, us2])
    #         if model_type != 'ICNN':
    #             sscost = get_sscost(fxu=fxu, hx=hx, lyu=lyu, 
    #                                 us=us, parameters=model_pars, 
    #                                 findRoot=True,
    #                                 xguess=xuguess['x'])
    #         else:
    #             sscost = icnn_lu(us)
    #         model_sscost += [sscost]

    #     model_sscost = np.asarray(model_sscost).squeeze()
    #     model_sscost = model_sscost.reshape(len(us2_list), len(us1_list))
    #     sscosts += [model_sscost]

    # # Get mesh.
    # us1 = np.asarray(us1_list)
    # us2 = np.asarray(us2_list)
    # us1, us2 = np.meshgrid(us1, us2)
    # legend_names = model_types
    # legend_colors = ['b', 'm']
    # figures = CstrFlashPlots.plot_sscosts(us1=us1, us2=us2, 
    #                                     sscosts=sscosts[:1], 
    #                                     legend_colors=legend_colors[:1], 
    #                                     legend_names=legend_names, 
    #                                     figure_size=PAPER_FIGSIZE, 
    #                                     ylabel_xcoordinate=-0.12, 
    #                                     left_label_frac=0.15)
    # figures += CstrFlashPlots.plot_sscosts(us1=us1, us2=us2, 
    #                                     sscosts=sscosts[1:], 
    #                                     legend_colors=legend_colors[1:], 
    #                                     legend_names=legend_names, 
    #                                     figure_size=PAPER_FIGSIZE, 
    #                                     ylabel_xcoordinate=-0.12, 
    #                                     left_label_frac=0.15)

    # # Finally plot.
    # with PdfPages('cstr_flash_ssopt.pdf', 'w') as pdf_file:
    #     for fig in figures:
    #         pdf_file.savefig(fig)

main()