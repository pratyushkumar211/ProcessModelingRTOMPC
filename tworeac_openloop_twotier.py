# [depends] %LIB%/hybridid.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/economicopt.py %LIB%/plotting_funcs.py
# [depends] %LIB%/linNonlinMPC.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_bbtrain.pickle
# [depends] tworeac_kooptrain.pickle
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
from tworeac_funcs import plant_ode, greybox_ode, get_parameters
from tworeac_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotting_funcs import PAPER_FIGSIZE, TwoReacPlots

def get_openloop_sol(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Np = 2
    Nx, Nu, Ny = model_pars['Nx'], model_pars['Nu'], model_pars['Ny']
    tSsOptFreq = 120
    
    # Steady state values.
    xs = xuguess['x'] #np.array([0.01, 2, 0.01])
    us = xuguess['u']
    
    # MPC Regulator parameters.
    Nmpc = 60
    ulb = model_pars['ulb']
    uub = model_pars['uub']
    Q = np.eye(Nx)
    R = 1e-3*np.eye(Nu)
    S = np.eye(Nu)

    # SS Opt parameters.
    empcPars = np.tile(np.array([[100, 200]]), (Nmpc, 1))

    # Extened Kalman Filter parameters. 
    xhatPrior = xs
    Qw = 1e-4*np.eye(Nx)
    Rv = 1e-4*np.eye(Ny)
    covxPrior = Qw

    # Return the Two Tier controller.
    controller = TwoTierMPController(fxu=fxu, hx=hx, lyup=cost_yup, 
                                     empcPars=empcPars, tSsOptFreq=tSsOptFreq,
                                     Nx=Nx, Nu=Nu, Ny=Ny, 
                                     xs=xs, us=us, Q=Q, R=R, S=S, ulb=ulb, 
                                     uub=uub, Nmpc=Nmpc, xhatPrior=xhatPrior, 
                                     covxPrior=covxPrior, Qw=Qw, Rv=Rv)
    
    # Get the open-loop solution.
    useq = controller.useq[0]
    xseq, yseq = [], []
    xt = controller.x0[0]
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
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    tworeac_bbtrain = PickleTool.load(filename='tworeac_bbtrain.pickle',
                                      type='read')

    # Get the black-box model parameters and function handles.
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=tworeac_bbtrain, 
                                                       parameters=parameters)

    # Get the plant function handle.
    Delta = parameters['Delta']
    ps = parameters['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, parameters)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, parameters)

    # Lists to loop over for the three problems.  
    model_types = ['plant', 'black-box']
    fxu_list = [plant_fxu, blackb_fxu]
    hx_list = [plant_hx, blackb_hx]
    par_list = [parameters, bb_pars]
    Nps = [None, bb_pars['Np']]
    
    # Lists to store solutions.
    ulist, xlist = [], []

    # Loop over the models.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=parameters, Np=Np, Nx=model_pars['Nx'])

        # Get the steady state optimum.
        t, useq, xseq, yseq = get_openloop_sol(fxu, hx, model_pars, xuguess)

        # Store. 
        ulist += [useq]
        if model_type != 'plant':
            xseq = np.insert(yseq, [2], 
                             np.nan*np.ones((yseq.shape[0], 1)), axis=1)
        xlist += [xseq]

    # Get figure.
    t = t*Delta/60
    legend_names = ['Plant', 'Black-box']
    legend_colors = ['b', 'dimgrey']
    figures = TwoReacPlots.plot_xudata(t=t, xlist=xlist, ulist=ulist,
                                        legend_names=legend_names,
                                        legend_colors=legend_colors, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.1, 
                                        title_loc=(0.05, 0.9), 
                                        font_size=12)

    # Finally plot.
    with PdfPages('tworeac_openloop_twotier.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()