# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/../cstr_flash_funcs.py
# [depends] cstr_flash_parameters.pickle
# [depends] cstr_flash_train.pickle
# [makes] pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import casadi
import copy
import numpy as np
from hybridid import (PickleTool, SimData, c2dNonlin)
from linNonlinMPC import (NonlinearPlantSimulator, RTOController, 
                         online_simulation)
from cstr_flash_funcs import plant_ode, greybox_ode, hybrid_func, measurement
from cstr_flash_funcs import get_hybrid_pars_check_func, get_economic_opt_pars
from cstr_flash_funcs import get_model, stage_cost

def get_controller(model_func, model_pars, model_type, opt_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get models parameters/sizes.
    ps = model_pars['ps']
    Delta = model_pars['Delta']
    (Nu, Ny) = (model_pars['Nu'], model_pars['Ny'])
    Np = opt_pars.shape[1]
    Ntstep_solve = 2*60

    # State-space model (discrete time).
    hx = lambda x: measurement(x, model_pars)
    if model_type == 'hybrid':
        fxu = lambda x, u: model_func(x, u, model_pars)
    else:
        fxu = lambda x, u: model_func(x, u, ps, model_pars)
        fxu = c2dNonlin(fxu, Delta)

    # Get the stage cost.
    if model_type == 'plant':
        lyup_yindices = [4, 6, 7]
        Nx = model_pars['Nx']
    elif model_type == 'grey-box':
        lyup_yindices = [4, 6, 7]
        Nx = model_pars['Ng']
    else:
        lyup_yindices = [4, 6, 7]
        Nx = model_pars['Nx']
    lyup = lambda y, u, p: stage_cost(y, u, p, model_pars, lyup_yindices)
    
    # Get steady states.
    xs = model_pars['xs']
    us = model_pars['us']
    init_guess = dict(x=xs, u=us)

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Return the NN controller.
    return RTOController(fxu=fxu, hx=hx,
                         lyup=lyup,
                         Nx=Nx, Nu=Nu, Np=Np,
                         ulb=ulb, uub=uub,
                         init_guess=init_guess,
                         opt_pars=opt_pars,
                         Ntstep_solve=Ntstep_solve)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                            'cstr_flash_parameters.pickle',
                                            type='read')
    cstr_flash_train = PickleTool.load(filename=
                                            'cstr_flash_train.pickle',
                                            type='read')

    # Get parameters for the real-time optimization.
    plant_pars = cstr_flash_parameters['plant_pars']
    greybox_pars = cstr_flash_parameters['greybox_pars']
    opt_pars, disturbances = get_economic_opt_pars(num_days=2,
                                         sample_time=plant_pars['Delta'], 
                                         plant_pars=plant_pars)

    # Check the hybrid func and get parameters.
    greybox_processed_data = cstr_flash_parameters['greybox_processed_data']
    hybrid_pars = get_hybrid_pars_check_func(greybox_pars=greybox_pars, 
                                train=cstr_flash_train,
                                greybox_processed_data=greybox_processed_data)

    # Run simulations for different model.
    cl_data_list, avg_stage_costs_list = [], []
    model_odes = [plant_ode, greybox_ode, hybrid_func]
    model_pars = [plant_pars, greybox_pars, hybrid_pars]
    model_types = ['plant', 'grey-box', 'hybrid']
    plant_lyup = lambda y, u, p: stage_cost(y, u, p, plant_pars, [4, 6, 7])
    regulator_guess = None
    for (model_ode, model_par,
         model_type) in zip(model_odes, model_pars, model_types):
        plant = get_model(parameters=plant_pars, plant=True)
        controller = get_controller(model_ode, model_par, model_type, opt_pars)
        cl_data, avg_stage_costs = online_simulation(plant,
                                         controller,
                                         plant_lyup=plant_lyup,
                                         Nsim=24*60, disturbances=disturbances,
                                         stdout_filename='cstr_flash_rto.txt')
        cl_data_list += [cl_data]
        avg_stage_costs_list += [avg_stage_costs]
    
    # Save data.
    PickleTool.save(data_object=dict(cl_data_list=cl_data_list,
                                     cost_pars=opt_pars,
                                     disturbances=disturbances,
                                     avg_stage_costs=avg_stage_costs_list),
                    filename='cstr_flash_rto.pickle')

main()