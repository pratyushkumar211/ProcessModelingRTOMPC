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
from linNonlinMPC import NonlinearEMPCRegulator
from tworeac_funcs import plant_ode, greybox_ode, get_parameters
from tworeac_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
from economicopt import get_kooppars_fxu_hx, fnn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotting_funcs import PAPER_FIGSIZE, TwoReacPlots

def get_openloop_sol(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes. 
    Np = 2
    Nx, Nu = model_pars['Nx'], model_pars['Nu']
    Nmpc = 60

    # Get the stage cost.
    lxup = lambda x, u, p: cost_yup(hx(x), u, p)
    lxup = mpc.getCasadiFunc(lxup, [Nx, Nu, Np], ["x", "u", "p"])

    # Initial parameters. 
    t0EmpcPars = np.tile(np.array([[100, 200]]), (Nmpc, 1))

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Convert fxu to casadi func.
    fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

    # Return the NN controller.
    regulator = NonlinearEMPCRegulator(fxu=fxu, lxup=lxup, Nx=Nx, Nu=Nu, Np=Np, 
                                       Nmpc=Nmpc, ulb=ulb, uub=uub, 
                                       t0Guess=xuguess, 
                                       t0EmpcPars=t0EmpcPars)
    
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

def get_koopman_xkp0(train, parameters):

    # Get initial state.
    Np = train['Np']
    us = parameters['us']
    yindices = parameters['yindices']
    ys = parameters['xs'][yindices]
    yz0 = np.concatenate((np.tile(ys, (Np+1, )), 
                          np.tile(us, (Np, ))))

    # Scale initial state and get the lifted state.
    fN_weights = train['trained_weights'][-1][:-2]
    xuyscales = train['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
                            np.tile(umean, (Np, ))))
    yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
                            np.tile(ustd, (Np, ))))
    yz0 = (yz0 - yzmean)/yzstd
    xkp0 = np.concatenate((yz0, fnn(yz0, fN_weights, 1.)))

    # Return.
    return xkp0

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

    # Get the black-box model parameters and function handles.
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=tworeac_bbtrain, 
                                                       parameters=parameters) 

    # Get the Koopman model parameters and function handles.
    koop_pars, koop_fxu, koop_hx = get_kooppars_fxu_hx(train=tworeac_kooptrain, 
                                                       parameters=parameters)
    xkp0 = get_koopman_xkp0(tworeac_kooptrain, parameters)

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
    model_types = ['plant', 'grey-box', 'black-box', 'Koopman']
    fxu_list = [plant_fxu, gb_fxu, blackb_fxu, koop_fxu]
    hx_list = [plant_hx, plant_hx, blackb_hx, koop_hx]
    par_list = [parameters, gb_pars, bb_pars, koop_pars]
    Nps = [None, None, bb_pars['Np'], koop_pars['Np']]
    
    # Lists to store solutions.
    ulist, xlist = [], []

    # Loop over the models.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=parameters, Np=Np, Nx=model_pars['Nx'])

        # Update guess for the Koopman model.
        if model_type == 'Koopman':
            xuguess['x'] = xkp0

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
    legend_names = ['Plant', 'Grey-box', 'Black-box', 'Koopman']
    legend_colors = ['b', 'g', 'dimgrey', 'm']
    figures = TwoReacPlots.plot_xudata(t=t, xlist=xlist, ulist=ulist,
                                        legend_names=legend_names,
                                        legend_colors=legend_colors, 
                                        figure_size=PAPER_FIGSIZE, 
                                        ylabel_xcoordinate=-0.1, 
                                        title_loc=(0.05, 0.9), 
                                        font_size=12)

    # Finally plot.
    with PdfPages('tworeac_openloop.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)



main()