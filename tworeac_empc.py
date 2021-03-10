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
    tworeac_parameters_nonlin = PickleTool.load(filename=
                                            'tworeac_parameters_nonlin.pickle',
                                            type='read')
    tworeac_kooptrain_nonlin = PickleTool.load(filename=
                                            'tworeac_kooptrain_nonlin.pickle',
                                            type='read')

    # Get EMPC simulation parameters.
    parameters = tworeac_parameters_nonlin['parameters']
    cost_pars = get_economic_opt_pars(Delta=parameters['Delta'])
    Nsim = cost_pars.shape[0]
    disturbances = np.repeat(parameters['ps'][np.newaxis, :], Nsim)
    
    # Get parameters for the hybrid function and check.
    #hybrid_pars = get_hybrid_pars_check_func(parameters=parameters,
    #                training_data=tworeac_parameters_nonlin['training_data'],
    #                train=tworeac_train_nonlin)

    # Get parameters for the EMPC nonlin function and check.
    koopman_pars = get_koopman_pars_check_func(parameters=parameters,
                    training_data=tworeac_parameters_nonlin['training_data'],
                    train=tworeac_kooptrain_nonlin)

    # Run simulations for different model.
    cl_data_list, avg_stage_costs_list, openloop_sols = [], [], []
    model_odes = [plant_ode, greybox_ode, koopman_func]
    model_pars = [parameters, parameters, koopman_pars]
    model_types = ['plant', 'grey-box', 'koopman']
    regulator_guess = None
    # Do different simulations for the different plant models.
    for (model_ode,
         model_par, model_type) in zip(model_odes, model_pars, model_types):
        mhe_noise_tuning = get_mhe_noise_tuning(model_type, model_par)
        plant = get_model(parameters=parameters, plant=True)
        controller, hx = get_controller(model_ode, model_par, model_type,
                                    cost_pars, mhe_noise_tuning,
                                    regulator_guess)
        cl_data, avg_stage_costs, openloop_sol = online_simulation(plant,
                                         controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=8*60, disturbances=disturbances,
                                         stdout_filename='tworeac_empc.txt')
        cl_data_list += [cl_data]
        avg_stage_costs_list += [avg_stage_costs]
        if model_type == 'koopman':
            xseq = []
            xkpseq = openloop_sol[1]
            for t in range(controller.Nmpc+1):
                xseq.append(hx(xkpseq[t, :]))
            openloop_sol[1] = np.asarray(xseq)
        openloop_sols += [openloop_sol]
        if model_type == 'plant':
            regulator_guess = dict(u=controller.regulator.useq[0],
                                   x=controller.regulator.xseq[0])

    # Save data.
    PickleTool.save(data_object=dict(cl_data_list=cl_data_list,
                                     cost_pars=cost_pars,
                                     disturbances=disturbances,
                                     avg_stage_costs=avg_stage_costs_list,
                                     openloop_sols=openloop_sols),
                    filename='tworeac_empc_nonlin.pickle')

main()