# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/../tworeac_nonlin_funcs.py
# [depends] tworeac_parameters_nonlin.pickle
# [depends] tworeac_train_nonlin.pickle
# [makes] pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import casadi
import numpy as np
from hybridid import (PickleTool, SimData, _resample_fast, c2dNonlin,
                     interpolate_yseq, koopman_func, 
                     get_koopman_pars_check_func)
from linNonlinMPC import (NonlinearPlantSimulator, NonlinearEMPCController, 
                         online_simulation)
from tworeac_nonlin_funcs import plant_ode, greybox_ode, measurement
from tworeac_nonlin_funcs import get_model, get_hybrid_pars, hybrid_func
from tworeac_nonlin_funcs import get_economic_opt_pars
from tworeac_nonlin_funcs import get_hybrid_pars_check_func

def get_controller(model_func, model_pars, model_type,
                   cost_pars, mhe_noise_tuning, 
                   regulator_guess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get the sizes.
    if model_type == 'grey-box':
        Nx = model_pars['Ng']
    else:
        Nx = model_pars['Nx']
    Nu, Ny = model_pars['Nu'], model_pars['Ny']
    Nd = Ny

    # Get state space and input models.
    if model_type == 'koopman':
        fxu = lambda x, u: model_func(x, u, model_pars)
        ymean, ystd = model_pars['xuyscales']['yscale']
        hx = lambda x: x[:Ny]*ystd + ymean
    else:
        ps = model_pars['ps']
        Delta = model_pars['Delta']
        fxu = lambda x, u: model_func(x, u, ps, model_pars)
        fxu = c2dNonlin(fxu, Delta)
        hx = lambda x: measurement(x, model_pars)

    # Get the stage cost.
    lyup = lambda y, u, p: stage_cost(y, u, p)
    
    # Get the disturbance models.
    Bd = np.zeros((Nx, Nd))
    if model_type == 'plant' or model_type == 'koopman':
        Bd[0, 0] = 1.
        Bd[1, 1] = 1.
    else:
        Ng = model_pars['Ng']
        Bd[:Ng, :Nd] = np.eye(Nd)
    Cd = np.zeros((Ny, Nd))

    # Get steady states.
    if model_type == 'grey-box':
        xs = model_pars['xs'][:2]
    else:
        xs = model_pars['xs']
    us = model_pars['us']
    ds = np.zeros((Nd,))
    ys = hx(xs)

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Fictitious noise covariances for MHE.
    Qwx, Qwd, Rv = mhe_noise_tuning

    # Horizon lengths.
    Nmpc = 60
    Nmhe = 30

    # Return the NN controller.
    return NonlinearEMPCController(fxu=fxu, hx=hx,
                                   lyup=lyup, Bd=Bd, Cd=Cd,
                                   Nx=Nx, Nu=Nu, Ny=Ny, Nd=Nd,
                                   xs=xs, us=us, ds=ds, ys=ys,
                                   empc_pars=cost_pars,
                                   ulb=ulb, uub=uub, Nmpc=Nmpc,
                                   Qwx=Qwx, Qwd=Qwd, Rv=Rv, Nmhe=Nmhe,
                                   guess=regulator_guess), hx

def get_mhe_noise_tuning(model_type, model_par):
    # Get MHE tuning.
    if model_type == 'plant' or model_type =='koopman':
        Qwx = 1e-6*np.eye(model_par['Nx'])
        Qwd = 1e-2*np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    if model_type == 'grey-box':
        Qwx = 1e-6*np.eye(model_par['Ng'])
        Qwd = 1e-2*np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    return (Qwx, Qwd, Rv)

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