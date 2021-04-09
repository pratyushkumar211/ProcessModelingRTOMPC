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
import mpctools as mpc
import casadi
import copy
import numpy as np
from hybridid import PickleTool, SimData, measurement, get_model
from linNonlinMPC import TwoTierMPController
from tworeac_funcs import plant_ode, greybox_ode, get_parameters
from tworeac_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
from economicopt import get_kooppars_fxu_hx, fnn, get_koopman_ss_xkp0
from economicopt import online_simulation

def getController(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Np, Nx, Nu, Ny = 2, model_pars['Nx'], model_pars['Nu'], model_pars['Ny']
    Nmhe, Nmpc = 15, 60

    # MHE tuning.
    tSsOptFreq = 120

    # Initial parameters. 
    empcPars = np.repeat(np.array([[100, 180], 
                                   [100, 150], 
                                   [100, 120], 
                                   [100, 250], 
                                   [100, 250]]), 120, axis=0)

    # Steady states/guess.
    xs, us = xuguess['x'], xuguess['u']

    # MPC Regulator parameters.
    Nmpc = 60
    ulb, uub = model_pars['ulb'], model_pars['uub']
    Q = np.eye(Nx)
    R = 1e-3*np.eye(Nu)
    S = np.eye(Nu)

    # Extened Kalman Filter parameters. 
    xhatPrior = xs[:, np.newaxis]
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
    
    # Return Controller.
    return controller

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
    model_types = ['plant', 'black-box']
    fxu_list = [plant_fxu, blackb_fxu]
    hx_list = [plant_hx, blackb_hx]
    par_list = [parameters, bb_pars]
    Nps = [None, bb_pars['Np']]
    
    # Get disturbances.
    disturbances = np.repeat(parameters['ps'], 8*60, axis=0)[:, np.newaxis]

    # Lists to store solutions.
    clDataList, stageCostList = [], []

    # Loop over the models.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=parameters, Np=Np, Nx=model_pars['Nx'])

        # Get controller.
        controller = getController(fxu, hx, model_pars, xuguess)

        # Get plant.
        plant = get_model(ode=plant_ode, parameters=parameters)

        # Run closed-loop simulation.
        clData, avgStageCosts = online_simulation(plant, controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=8*60, disturbances=disturbances,
                                    stdout_filename='tworeac_empc_twotier.txt')

        # Store data. 
        clDataList += [clData]
        stageCostList += [avgStageCosts]

    # Save data.
    PickleTool.save(data_object=dict(clDataList=clDataList,
                                     stageCostList=stageCostList),
                    filename='tworeac_empc_twotier.pickle')

main()