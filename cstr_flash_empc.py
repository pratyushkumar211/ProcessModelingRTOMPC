# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_train.pickle
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
from cstr_flash_funcs import plant_ode, greybox_ode
from cstr_flash_funcs import cost_yup
from economicopt import get_bbpars_fxu_hx, c2dNonlin, get_xuguess
from economicopt import online_simulation

def getController(fxu, hx, model_pars, mheTuning, distModel, xuguess):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Np, Nx, Nu, Ny = 3, model_pars['Nx'], model_pars['Nu'], model_pars['Ny']
    Nmhe, Nmpc = 30, 120

    # MHE tuning.
    Qwx, Qwd, Rv = mheTuning

    # Disturbance model.
    Bd, Cd = distModel

    # Initial parameters. 
    empcPars = np.repeat(np.array([[10, 2000, 13000], 
                                   [10, 2000, 15000], 
                                   [10, 2000, 20000],
                                   [10, 2000, 14000],
                                   [10, 2000, 14000]]), 360, axis=0)

    # Get the stage cost.
    lyup = lambda y, u, p: cost_yup(y, u, p, model_pars)

    # Get upper and lower bounds.
    ulb, uub = model_pars['ulb'], model_pars['uub']

    # Steady states/guess.
    xs, us = xuguess['x'], xuguess['u']
    ds = np.zeros((Ny, ))

    # Return the NN controller.
    controller = NonlinearEMPCController(fxu=fxu, hx=hx, lyup=lyup, 
                                         Bd=Bd, Cd=Cd, Nx=Nx, Nu=Nu, Ny=Ny,
                                         Nd=Ny, xs=xs, us=us, ds=ds, 
                                         empcPars=empcPars, ulb=ulb, uub=uub,
                                         Nmpc=Nmpc, Qwx=Qwx, Qwd=Qwd, Rv=Rv,
                                         Nmhe=Nmhe)
    
    # Return Controller.
    return controller

def getEstimatorTuning(model_type, model_pars, xuguess):
    """ Function to get estimation tuning. """

    # Get sizes.
    Nx, Ny = model_pars['Nx'], model_pars['Ny'] 
    xs = xuguess['x']
    ys = measurement(xs, model_pars)

    # Get MHE tuning.
    if model_type == 'plant':
        
        Qwx = 1e-8*np.eye(Nx)
        Qwd = 1e-8*np.eye(Ny)
        Rv = np.eye(Ny)

    # Return variances. 
    return (Qwx, Qwd, Rv)

def getDistModel(model_type, model_pars):
    """ Get the disturbance model. """

    # Get sizes.
    Nx, Ny = model_pars['Nx'], model_pars['Ny'] 

    # Same Cd matrix for the three models.
    Cd = np.eye(Ny)
    
    # Get disturbance model.
    if model_type == 'plant':
        
        Bd = np.zeros((Nx, Ny))

    # Disturbance model.
    return Bd, Cd

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                         'cstr_flash_parameters.pickle',
                                         type='read')
    plant_pars = cstr_flash_parameters['plant_pars']

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
    model_types = ['plant']
    fxu_list = [plant_fxu]
    hx_list = [plant_hx]
    par_list = [plant_pars]
    Nps = [None]
    
    # Get disturbances.
    disturbances = np.repeat(plant_pars['ps'][:, np.newaxis], 24*60, axis=0)

    # Lists to store solutions.
    clDataList, stageCostList = [], []

    # Loop over the models.
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=plant_pars, Np=Np, Nx=model_pars['Nx'])

        # Get MHE tuning.
        mheTuning = getEstimatorTuning(model_type, model_pars, xuguess)

        # Get Disturbance model.
        distModel = getDistModel(model_type, model_pars)

        # Get controller.
        controller = getController(fxu, hx, model_pars,
                                   mheTuning, distModel, xuguess)

        # Get plant.
        plant = get_model(ode=plant_ode, parameters=plant_pars)

        # Run closed-loop simulation.
        clData, avgStageCosts = online_simulation(plant, controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=24*60, disturbances=disturbances,
                                         stdout_filename='cstr_flash_empc.txt')

        # Store data. 
        clDataList += [clData]
        stageCostList += [avgStageCosts]

    # Save data.
    PickleTool.save(data_object=dict(clDataList=clDataList,
                                     stageCostList=stageCostList),
                    filename='cstr_flash_empc.pickle')

main()