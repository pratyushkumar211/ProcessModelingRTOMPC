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
from cstr_flash_funcs import plant_ode, greybox_ode
from cstr_flash_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
from economicopt import online_simulation

def getController(fxu, hx, model_pars, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Np, Nx, Nu, Ny = 3, model_pars['Nx'], model_pars['Nu'], model_pars['Ny']
    tSsOptFreq = 360

    # Get cost as a function of yup.
    lyup = lambda y, u, p: cost_yup(y, u, p, model_pars)

    # Initial parameters. 
    empcPars = np.repeat(np.array([[10, 2000, 13000], 
                                   [10, 2000, 14000], 
                                   [10, 2000, 20000],
                                   [10, 2000, 12000],
                                   [10, 2000, 12000]]), 360, axis=0)

    # Steady states/guess.
    xs, us = xuguess['x'], xuguess['u']

    # MPC Regulator parameters.
    Nmpc = 120
    ulb, uub = model_pars['ulb'], model_pars['uub']
    Q = np.eye(Nx)*np.diag(1/xs**2)
    R = np.eye(Nu)*np.diag(1/us**2)
    S = 1e-3*np.eye(Nu)*np.diag(1/us**2)

    # Extened Kalman Filter parameters. 
    xhatPrior = xs[:, np.newaxis]
    Qw = 1e-8*np.eye(Nx)
    Rv = np.eye(Ny)
    covxPrior = Qw

    # Return the Two Tier controller.
    controller = TwoTierMPController(fxu=fxu, hx=hx, lyup=lyup, 
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

    # Lists to loop over for the three problems.  
    model_types = ['plant', 'black-box']
    fxu_list = [plant_fxu, blackb_fxu]
    hx_list = [plant_hx, blackb_hx]
    par_list = [plant_pars, bb_pars]
    Nps = [None, bb_pars['Np']]
        
    # Get disturbances.
    disturbances = np.repeat(plant_pars['ps'][np.newaxis, :], 24*60, axis=0)

    # Lists to store solutions.
    clDataList, stageCostList = [], []

    # Loop over the models.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np, Nx=model_pars['Nx'])

        # Get controller.
        controller = getController(fxu, hx, model_pars, xuguess)

        # Get plant.
        plant = get_model(ode=plant_ode, parameters=plant_pars)

        # Run closed-loop simulation.
        clData, avgStageCosts = online_simulation(plant, controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=24*60, disturbances=disturbances,
                                stdout_filename='cstr_flash_empc_twotier.txt')

        # Store data. 
        clDataList += [clData]
        stageCostList += [avgStageCosts]

    # Save data.
    PickleTool.save(data_object=dict(clDataList=clDataList,
                                     stageCostList=stageCostList),
                    filename='cstr_flash_empc_twotier.pickle')

main()