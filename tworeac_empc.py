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
from linNonlinMPC import NonlinearEMPCController
from tworeac_funcs import plant_ode, greybox_ode, get_parameters
from tworeac_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
from economicopt import get_kooppars_fxu_hx, fnn, get_koopman_ss_xkp0
from economicopt import online_simulation

def getController(fxu, hx, model_pars, mheTuning, distModel, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Np, Nx, Nu, Ny = 2, model_pars['Nx'], model_pars['Nu'], model_pars['Ny']
    Nmhe, Nmpc = 15, 60

    # MHE tuning.
    Qwx, Qwd, Rv = mheTuning

    # Disturbance model.
    Bd, Cd = distModel

    # Initial parameters. 
    empcPars = np.repeat(np.array([[100, 180], 
                                   [100, 150], 
                                   [100, 120], 
                                   [100, 250], 
                                   [100, 250]]), 120, axis=0)

    # Get upper and lower bounds.
    ulb, uub = model_pars['ulb'], model_pars['uub']

    # Steady states/guess.
    xs, us = xuguess['x'], xuguess['u']
    ds = np.zeros((Ny, ))

    # Return the NN controller.
    controller = NonlinearEMPCController(fxu=fxu, hx=hx, lyup=cost_yup, 
                                         Bd=Bd, Cd=Cd, Nx=Nx, Nu=Nu, Ny=Ny,
                                         Nd=Ny, xs=xs, us=us, ds=ds, 
                                         empcPars=empcPars, ulb=ulb, uub=uub,
                                         Nmpc=Nmpc, Qwx=Qwx, Qwd=Qwd, Rv=Rv,
                                         Nmhe=Nmhe)
    
    # Return Controller.
    return controller

def getEstimatorTuning(model_type, model_pars):
    """ Function to get estimation tuning. """

    # Get sizes.
    Nx, Ny = model_pars['Nx'], model_pars['Ny'] 

    # Get MHE tuning.
    if model_type == 'plant':
        
        Qwx, Qwd, Rv = 1e-6*np.eye(Nx), 1e-2*np.eye(Ny), 1e-3*np.eye(Ny)

    if model_type =='Koopman':

        Qwx, Qwd, Rv = 1e-6*np.eye(Nx), 1e-2*np.eye(Ny), 1e-3*np.eye(Ny)

    if model_type == 'grey-box':

        Qwx, Qwd, Rv = 1e-6*np.eye(Nx), 1e-2*np.eye(Ny), 1e-3*np.eye(Ny)

    # Return variances. 
    return (Qwx, Qwd, Rv)

def getDistModel(model_type, model_pars):
    """ Get the disturbance model. """

    # Get sizes.
    Nx, Ny = model_pars['Nx'], model_pars['Ny'] 

    # Same Cd matrix for the three models.
    Cd = np.zeros((Ny, Ny))
    
    # Get disturbance model.
    if model_type == 'plant':
        
        Bd = np.array([[0, 0], 
                       [1, 0],
                       [0, 1]])

    if model_type == 'Koopman':
        
        Bd = np.zeros((Nx, Ny))
        Bd[1, 0] = 1
        Bd[2, 1] = 1

    if model_type == 'grey-box':
        
        Bd = np.array([[0, 0], 
                       [1, 0],
                       [0, 1]])

    # Disturbance model.
    return Bd, Cd

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    tworeac_kooptrain = PickleTool.load(filename='tworeac_kooptrain.pickle',
                                      type='read')

    # Get the Koopman model parameters and function handles.
    koop_pars, koop_fxu, koop_hx = get_kooppars_fxu_hx(train=tworeac_kooptrain, 
                                                       parameters=parameters)
    xkp0 = get_koopman_ss_xkp0(tworeac_kooptrain, parameters)

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
    model_types = ['plant', 'grey-box', 'Koopman']
    fxu_list = [plant_fxu, gb_fxu, koop_fxu]
    hx_list = [plant_hx, plant_hx, koop_hx]
    par_list = [parameters, gb_pars, koop_pars]
    Nps = [None, None, koop_pars['Np']]
    
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

        # Get MHE tuning.
        mheTuning = getEstimatorTuning(model_type, model_pars)

        # Get Disturbance model.
        distModel = getDistModel(model_type, model_pars)

        # Update guess for the Koopman model.
        if model_type == 'Koopman':
            xuguess['x'] = xkp0

        # Get controller.
        controller = getController(fxu, hx, model_pars,
                                   mheTuning, distModel, xuguess)

        # Get plant.
        plant = get_model(ode=plant_ode, parameters=parameters)

        # Run closed-loop simulation.
        clData, avgStageCosts = online_simulation(plant, controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=8*60, disturbances=disturbances,
                                         stdout_filename='tworeac_empc.txt')

        # Store data. 
        clDataList += [clData]
        stageCostList += [avgStageCosts]

    # Save data.
    PickleTool.save(data_object=dict(clDataList=clDataList,
                                     stageCostList=stageCostList),
                    filename='tworeac_empc.pickle')

main()