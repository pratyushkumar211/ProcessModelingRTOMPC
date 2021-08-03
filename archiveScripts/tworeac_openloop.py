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
from tworeac_funcs import cost_yup, plant_ode
from economicopt import c2dNonlin, get_xuguess
from BlackBoxFuncs import get_bbNN_pars, bbNN_fxu, bb_hx
from TwoReacHybridFuncs import (get_tworeacHybrid_pars, 
                                tworeacHybrid_fxu, 
                               tworeacHybrid_hx)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotting_funcs import PAPER_FIGSIZE, TwoReacPlots

def get_openloop_sol(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes. 
    Np = 2
    Nx, Nu = model_pars['Nx'], model_pars['Nu']
    Nmpc = 120

    # Get the stage cost.
    lxup = lambda x, u, p: cost_yup(hx(x), u, p)
    lxup = mpc.getCasadiFunc(lxup, [Nx, Nu, Np], ["x", "u", "p"])

    # Initial parameters. 
    t0EmpcPars = np.tile(np.array([[100, 180]]), (Nmpc, 1))

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Convert fxu to casadi func.
    fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

    # Return the NN controller.
    regulator = NonlinearEMPCRegulator(fxu=fxu, lxup=lxup, Nx=Nx, Nu=Nu, 
                                       Np=Np, 
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

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']
    greybox_pars = tworeac_parameters['greybox_pars']
    tworeac_bbNNtrain = PickleTool.load(filename=
                                    'tworeac_bbNNtrain.pickle',
                                      type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                      'tworeac_hybtrain.pickle',
                                      type='read')

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
    Delta = plant_pars['Delta']
    ps = plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Lists to loop over for different models.
    model_types = ['Plant', 'Hybrid']
    fxu_list = [plant_fxu, hyb_fxu]
    hx_list = [plant_hx, hyb_hx]
    par_list = [plant_pars, hyb_pars]
    Nps = [None, None]
    
    # Lists to store solutions.
    ulist, xlist = [], []

    # Loop over the models.
    for (model_type, fxu, 
         hx, model_pars, Np) in zip(model_types, fxu_list, 
                                    hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np, 
                              Nx=model_pars['Nx'])

        # Get the steady state optimum.
        t, useq, xseq, yseq = get_openloop_sol(fxu, hx, 
                                                model_pars, xuguess)

        # Store. 
        ulist += [useq]
        xlist += [xseq]

    # Get figure.
    t = t*Delta/60
    legend_names = model_types
    legend_colors = ['b', 'm']
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