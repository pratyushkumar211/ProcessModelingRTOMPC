# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/cstr_flash_funcs.py %LIB%/economicopt.py
# [depends] %LIB%/plotting_funcs.py
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_bbtrain.pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import mpctools as mpc
import casadi
import copy
import numpy as np
from hybridid import PickleTool, SimData, measurement
from linNonlinMPC import TwoTierMPController
from cstr_flash_funcs import plant_ode, greybox_ode
from cstr_flash_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotting_funcs import PAPER_FIGSIZE, CstrFlashPlots

def get_openloop_sol(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """
    
    # Some sizes.
    Np = 3
    Nx, Nu, Ny = model_pars['Nx'], model_pars['Nu'], model_pars['Ny']
    tSsOptFreq = 120

    # Get cost as a function of yup.
    lyup = lambda y, u, p: cost_yup(y, u, p, model_pars)

    # Steady states/guess.
    xs, us = xuguess['x'], xuguess['u']

    # MPC Regulator parameters.
    Nmpc = 120
    ulb, uub = model_pars['ulb'], model_pars['uub']
    Q = np.eye(Nx)*np.diag(1/xs**2)
    R = 1e-3*np.eye(Nu)*np.diag(1/us**2)
    S = np.eye(Nu)*np.diag(1/us**2)
    
    # Initial parameters. 
    empcPars = np.tile(np.array([[10, 2000, 13000]]), (Nmpc, 1))

    # Extened Kalman Filter parameters. 
    xhatPrior = xs[:, np.newaxis]
    Qw = 1e-4*np.eye(Nx)
    Rv = 1e-4*np.eye(Ny)
    covxPrior = Qw

    # Return the Two Tier controller.
    controller = TwoTierMPController(fxu=fxu, hx=hx, lyup=lyup, 
                                     empcPars=empcPars, tSsOptFreq=tSsOptFreq,
                                     Nx=Nx, Nu=Nu, Ny=Ny,
                                     xs=xs, us=us, Q=Q, R=R, S=S, ulb=ulb, 
                                     uub=uub, Nmpc=Nmpc, xhatPrior=xhatPrior, 
                                     covxPrior=covxPrior, Qw=Qw, Rv=Rv)

    # Get the open-loop solution.
    useq = controller.useq[0]
    xseq, yseq = [], []
    xt = controller.x0[0][:, 0]
    for t in range(Nmpc):
        yseq += [hx(xt)]
        xseq += [xt]
        xt = fxu(xt, useq[t, :])
    xseq = np.array(xseq)
    yseq = np.array(yseq)
    t = np.arange(0, Nmpc, 1)

    # Return the open-loop sol.
    return (t, useq, xseq, yseq)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                         'cstr_flash_parameters.pickle',
                                         type='read')
    plant_pars = cstr_flash_parameters['plant_pars']
    cstr_flash_bbtrain = PickleTool.load(filename='cstr_flash_bbtrain.pickle',
                                     type='read')

    # Get the black-box model parameters and function handles.
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=
                                                       cstr_flash_bbtrain, 
                                                       parameters=plant_pars)
    # Add some more parameters to bb_pars.
    bb_pars['ps'] = plant_pars['ps']
    bb_pars['Td'] = plant_pars['Td']
    bb_pars['pho'] = plant_pars['pho']
    bb_pars['Cp'] = plant_pars['Cp']
    bb_pars['kb'] = plant_pars['kb']

    # Get the plant function handle.
    Delta = plant_pars['Delta']
    ps = plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Get the grey-box function handle.
    #gb_fxu = lambda x, u: greybox_ode(x, u, ps, parameters)
    #gb_fxu = c2dNonlin(gb_fxu, Delta)
    #gb_pars = copy.deepcopy(parameters)
    #gb_pars['Nx'] = len(parameters['gb_indices'])

    # Lists to loop over for the three problems.  
    model_types = ['plant', 'black-box']
    fxu_list = [plant_fxu, blackb_fxu]
    hx_list = [plant_hx, blackb_hx]
    par_list = [plant_pars, bb_pars]
    Nps = [None, bb_pars['Np']]
    
    # Lists to store solutions.
    ulist, xlist = [], []

    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np)

        # Get the steady state optimum.
        t, useq, xseq, yseq = get_openloop_sol(fxu, hx, model_pars, xuguess)

        # Store. 
        ulist += [useq]
        if model_type != 'plant':
            xseq = np.insert(yseq, [1, 2, 4, 5], 
                             np.nan*np.ones((yseq.shape[0], 1)), axis=1)
        xlist += [xseq]

    # Get figure.
    t = t*Delta/60
    legend_names = ['Plant', 'Black-box']
    legend_colors = ['b', 'dimgrey']
    figures = CstrFlashPlots.plot_data(t=t, ulist=ulist, 
                                ylist=None, xlist=xlist, 
                                figure_size=PAPER_FIGSIZE, 
                                u_ylabel_xcoordinate=-0.1, 
                                y_ylabel_xcoordinate=-0.1, 
                                x_ylabel_xcoordinate=-0.2, 
                                plot_ulabel=True,
                                legend_names=legend_names, 
                                legend_colors=legend_colors, 
                                title_loc=(0.25, 0.9), 
                                plot_y=False)

    # Finally plot.
    with PdfPages('cstr_flash_openloop_twotier.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)



main()