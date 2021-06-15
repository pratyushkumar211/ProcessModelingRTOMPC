# [depends] %LIB%/hybridid.py %LIB%/cstr_flash_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/BlackBoxFuncs.py
# [depends] %LIB%/CstrFlashHybridFuncs.py %LIB%/InputConvexFuncs.py
# [depends] %LIB%/plotting_funcs.py 
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_bbnntrain.pickle
# [depends] cstr_flash_hybtrain.pickle
# [makes] pickle
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
from economicopt import get_xs_sscost, get_ss_optimum, c2dNonlin
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from CstrFlashHybridFuncs import (get_CstrFlash_hybrid_pars, 
                                  CstrFlashHybrid_fxu, 
                                  CstrFlashHybrid_hx)
from cstr_flash_funcs import cost_yup, plant_ode
from plotting_funcs import CstrFlashPlots, PAPER_FIGSIZE
from InputConvexFuncs import get_icnn_pars, icnn_lyu
from InputConvexFuncs import get_ss_optimum as get_icnn_ss_optimum

def get_xuguess(*, model_type, plant_pars, Np=None):
    """ Get x, u guess depending on model type. """
    
    # Steady state control.
    us = plant_pars['us']

    if model_type == 'Plant':
        xs = plant_pars['xs']
    elif model_type == 'Black-Box-NN' or model_type == 'Hybrid':
        yindices = plant_pars['yindices']
        ys = plant_pars['xs'][yindices]
        us = np.array([15., 2.])
        xs = np.concatenate((np.tile(ys, (Np+1, )), 
                             np.tile(us, (Np, ))))
    elif model_type == 'ICNN':
        us = np.array([10., 8.])
        xs = None
    else:
        pass
    # Return as dict.
    return dict(x=xs, u=us)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                        'cstr_flash_parameters.pickle',
                                         type='read')
    cstr_flash_bbnntrain = PickleTool.load(filename=
                                     'cstr_flash_bbnntrain.pickle',
                                      type='read')
    cstr_flash_hybtrain = PickleTool.load(filename=
                                     'cstr_flash_hybtrain.pickle',
                                      type='read')
    # cstr_flash_icnntrain = PickleTool.load(filename=
    #                                  'cstr_flash_icnntrain.pickle',
    #                                   type='read')

    # Get plant and grey-box model parameters.
    plant_pars = cstr_flash_parameters['plant_pars']
    greybox_pars = cstr_flash_parameters['greybox_pars']

    # Get cost function handle.
    p = [500, 300, 12000]
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

    # Get Hybrid model parameters and function handles.
    hyb_pars = get_CstrFlash_hybrid_pars(train=cstr_flash_hybtrain, 
                                         greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: CstrFlashHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: CstrFlashHybrid_hx(x)

    # Get ICNN parameters and function.
    # icnn_pars = get_icnn_pars(train=cstr_flash_icnntrain, plant_pars=plant_pars)
    # icnn_lu = lambda u: icnn_lyu(u, icnn_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Black-Box-NN', 'Hybrid']
    fxu_list = [plant_fxu, bbnn_f, hyb_fxu, None]
    hx_list = [plant_hx, bbnn_h, hyb_hx, None]
    par_list = [plant_pars, bbnn_pars, hyb_pars, None]
    Nps = [None, 0, 0, None]

    # Loop over the different models, and obtain SS optimums.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess.
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np)

        # Get the steady state optimum.
        if model_type != 'ICNN':
            xs, us, ys, opt_sscost = get_ss_optimum(fxu=fxu, hx=hx, lyu=lyu, 
                                        parameters=model_pars, guess=xuguess)
        else:
            us = get_icnn_ss_optimum(lyu=icnn_lu, parameters=icnn_pars, 
                                      uguess=xuguess['u'])

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
    
    # # Get a linspace of steady-state u values.
    # Nu = plant_pars['Nu']
    # ulb, uub = plant_pars['ulb'], plant_pars['uub']

    # # Get lists.
    # us1_list = list(np.linspace(ulb[0], uub[0], 4))
    # us2_list = list(np.linspace(ulb[1], uub[1], 4))

    # # Lists to store Steady-state cost.
    # sscosts = []

    # # Loop over all the models.
    # for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
    #                                              hx_list, par_list, Nps):

    #     # List to store SS costs for one model.
    #     model_sscost = []

    #     # Get guess.
    #     xuguess = get_xuguess(model_type=model_type, 
    #                           plant_pars=plant_pars, Np=Np)

    #     # Compute SS cost.
    #     for us2, us1 in itertools.product(us2_list, us1_list):
            
    #         # Get the vector input.
    #         us = np.array([us1, us2])
    #         if model_type != 'ICNN':
    #             xs, sscost = get_xs_sscost(fxu=fxu, hx=hx, lyu=lyu, 
    #                                 us=us, parameters=model_pars, 
    #                                 xguess=xuguess['x'], 
    #                                 lbx=np.zeros((model_pars['Nx'], )), 
    #                                 ubx=np.tile(np.inf, (model_pars['Nx'], )))
    #         else:
    #             sscost = icnn_lu(us)
    #         model_sscost += [sscost]

    #     model_sscost = np.asarray(model_sscost).squeeze()
    #     model_sscost = model_sscost.reshape(len(us1_list), len(us2_list))
    #     sscosts += [model_sscost]

    # # Get mesh.
    # us1 = np.asarray(us1_list)
    # us2 = np.asarray(us2_list)
    # us1, us2 = np.meshgrid(us1, us2)

    # # Get dictionary to save data.
    # cstr_flash_ssopt = dict(us1=us1, us2=us2, sscosts=sscosts)

    # # Save data.
    # PickleTool.save(data_object=cstr_flash_ssopt,
    #                 filename='cstr_flash_ssopt.pickle')

main()