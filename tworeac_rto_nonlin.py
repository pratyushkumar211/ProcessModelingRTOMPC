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
from hybridid import (PickleTool, SimData, c2dNonlin)
from linNonlinMPC import (NonlinearPlantSimulator, RTOController, 
                          online_simulation)
from tworeac_nonlin_funcs import plant_ode, greybox_ode, measurement
from tworeac_nonlin_funcs import get_model, get_hybrid_pars, hybrid_func
from tworeac_nonlin_funcs import get_economic_opt_pars, stage_cost
from tworeac_nonlin_funcs import get_hybrid_pars_check_func

def get_controller(model_func, model_pars, model_type, opt_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get a few parameters.
    ps = model_pars['ps']
    Delta = model_pars['Delta']
    Nu, Ny = model_pars['Nu'], model_pars['Ny']
    Np = opt_pars.shape[1]
    Ntstep_solve = 2*60

    # Get the state dimension.
    if model_type == 'grey-box':
        Nx = model_pars['Ng']
    else:
        Nx = model_pars['Nx']

    # State-space model (discrete time).
    hx = lambda x: measurement(x)
    if model_type == 'hybrid':
        fxu = lambda x, u: model_func(x, u, model_pars)
    else:
        fxu = lambda x, u: model_func(x, u, ps, model_pars)
        fxu = c2dNonlin(fxu, Delta)

    # Get the stage cost.
    lyup = lambda y, u, p: stage_cost(y, u, p)
    
    # Get steady states.
    if model_type == 'grey-box':
        xs = model_pars['xs'][:2]
    else:
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
    tworeac_parameters_nonlin = PickleTool.load(filename=
                                            'tworeac_parameters_nonlin.pickle',
                                            type='read')
    tworeac_train_nonlin = PickleTool.load(filename=
                                            'tworeac_train_nonlin.pickle',
                                            type='read')

    # Get EMPC simulation parameters.
    parameters = tworeac_parameters_nonlin['parameters']
    cost_pars = get_economic_opt_pars(Delta=parameters['Delta'])
    Nsim = cost_pars.shape[0]
    disturbances = np.repeat(parameters['ps'][np.newaxis, :], Nsim)
    
    # Get parameters for the hybrid function and check.
    hybrid_pars = get_hybrid_pars_check_func(parameters=parameters,
                    training_data=tworeac_parameters_nonlin['training_data'],
                    train=tworeac_train_nonlin)

    # Run simulations for different model.
    cl_data_list, avg_stage_costs_list = [], []
    model_odes = [plant_ode, greybox_ode, hybrid_func]
    model_pars = [parameters, parameters, hybrid_pars]
    model_types = ['plant', 'grey-box', 'hybrid']

    # Do different simulations for the different plant models.
    for (model_ode,
         model_par, model_type) in zip(model_odes, model_pars, model_types):
        plant = get_model(parameters=parameters, plant=True)
        controller = get_controller(model_ode, model_par, model_type, cost_pars)
        cl_data, avg_stage_costs = online_simulation(plant, controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=8*60, disturbances=disturbances,
                                         stdout_filename='tworeac_rto.txt')
        cl_data_list += [cl_data]
        avg_stage_costs_list += [avg_stage_costs]
    
    # Save data.
    PickleTool.save(data_object=dict(cl_data_list=cl_data_list,
                                     cost_pars=cost_pars,
                                     disturbances=disturbances,
                                     avg_stage_costs=avg_stage_costs_list),
                    filename='tworeac_rto_nonlin.pickle')

main()