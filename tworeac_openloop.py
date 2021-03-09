# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/../tworeac_nonlin.py
# [depends] tworeac_parameters.pickle
# [depends] tworeac_train.pickle
# [makes] pickle
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
from linNonlinMPC import NonlinearEMPCRegulator
from tworeac_funcs import plant_ode, greybox_ode, get_parameters
from tworeac_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotting_funcs import PAPER_FIGSIZE, TwoReacPlots

def get_openloop_sol(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get the stage cost.
    lxup = lambda x, u, p: cost_yup(hx(x), u, p)
    
    # Some sizes. 
    Np = 2
    Nx, Nu = model_pars['Nx'], model_pars['Nu']
    Nmpc = 60

    # Initial parameters. 
    init_empc_pars = np.tile(np.array([[100, 180]]), (Nmpc, 1))

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Convert fxu to casadi func.
    fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

    # Return the NN controller.
    regulator = NonlinearEMPCRegulator(fxu=fxu, lxup=lxup, Nx=Nx, Nu=Nu, Np=Np, 
                                       Nmpc=Nmpc, ulb=ulb, uub=uub, 
                                       init_guess=xuguess, 
                                       init_empc_pars=init_empc_pars)

    # Get the open-loop solution.
    useq = regulator.useq[0]
    xseq = regulator.xseq[0][:-1, :]
    yseq = []
    for t in range(Nmpc):
        yseq += [hx(xseq[t, :])]
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

    # Get the grey-box function handle.
    gb_fxu = lambda x, u: greybox_ode(x, u, ps, parameters)
    gb_fxu = c2dNonlin(gb_fxu, Delta)
    gb_pars = copy.deepcopy(parameters)
    gb_pars['Nx'] = len(parameters['gb_indices'])

    # Lists to loop over for the three problems.  
    model_types = ['plant', 'grey-box', 'black-box']
    fxu_list = [plant_fxu, gb_fxu, blackb_fxu]
    hx_list = [plant_hx, plant_hx, blackb_hx]
    par_list = [parameters, gb_pars, bb_pars]
    Nps = [None, None, bb_pars['Np']]
    
    # Lists to store solutions.
    ulist, xlist = [], []


    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=parameters, Np=Np)

        # Get the steady state optimum.
        t, useq, xseq, yseq = get_openloop_sol(fxu, hx, model_pars, xuguess)

        # Store. 
        ulist += [useq]
        if model_type != 'plant':
            xseq = np.insert(yseq, [2], 
                             np.nan*np.ones((yseq.shape[0], 1)), axis=1)
        xlist += [xseq]

    # Get figure.
    t = t*Delta
    legend_names = ['Plant', 'Grey-box', 'Black-box']
    legend_colors = ['b', 'g', 'dimgrey']
    figures = TwoReacPlots.plot_xudata(t=t, xlist=xlist, ulist=ulist,
                                        legend_names=legend_names,
                                        legend_colors=legend_colors, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.1, 
                                        title_loc=(0.15, 0.9))

    # Finally plot.
    with PdfPages('tworeac_openloop.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)



main()