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
                     interpolate_yseq)
from linNonlinMPC import (NonlinearPlantSimulator, NonlinearEMPCController, 
                         online_simulation)
from tworeac_nonlin_funcs import plant_ode, greybox_ode, measurement
from tworeac_nonlin_funcs import get_model, get_hybrid_pars, hybrid_func
from tworeac_nonlin_funcs import get_economic_opt_pars

def get_controller(model_func, model_pars, model_type,
                   cost_pars, mhe_noise_tuning):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get models.
    ps = model_pars['ps']
    Delta = model_pars['Delta']

    # State-space model (discrete time).
    if model_type == 'hybrid':
        fxu = lambda x, u: model_func(x, u, model_pars)
    else:
        fxu = lambda x, u: model_func(x, u, ps, model_pars)
        fxu = c2dNonlin(fxu, Delta)

    # Measurement function.
    hx = lambda x: measurement(x)

    # Get the state dimension.
    if model_type == 'grey-box':
        Nx = model_pars['Ng']
    else:
        Nx = model_pars['Nx']

    # Get the stage cost.
    lyup = lambda y, u, p: stage_cost(y, u, p)
    
    # Get the sizes/sample time.
    Nu, Ny = model_pars['Nu'], model_pars['Ny']
    Nd = Ny

    # Get the disturbance models.
    Bd = np.zeros((Nx, Nd))
    if model_type == 'plant' or model_type == 'black-box-state-feed':
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
                                   Qwx=Qwx, Qwd=Qwd, Rv=Rv, Nmhe=Nmhe)

def stage_cost(y, u, p):
    """ Custom stage cost for the tworeac system. """    
    # Get inputs, parameters, and states.
    CAf = u[0:1]
    ca, cb = p[0:2]
    CA, CB = y[0:2]
    # Compute and return cost.
    return ca*CAf - cb*CB

def sim_hybrid(hybrid_func, hybrid_pars, uval, training_data):
    """ Hybrid validation simulation to make 
        sure the above programmed function is 
        the same is what tensorflow is training. """
    
    # Get initial state.
    t = hybrid_pars['tsteps_steady']
    Np = hybrid_pars['Npast']
    Ng = hybrid_pars['Ng']
    Ny = hybrid_pars['Ny']
    Nu = hybrid_pars['Nu']
    y = training_data[-1].y
    u = training_data[-1].u
    yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[:, np.newaxis]
    up0seq = u[t-Np:t:, ].reshape(Np*Nu, )[:, np.newaxis]
    z0 = np.concatenate((yp0seq, up0seq))
    xG0 = y[t, :][:, np.newaxis]
    xGz0 = np.concatenate((xG0, z0))

    # Start the validation simulation.
    uval = uval[t:, :]
    Nval = uval.shape[0]
    hx = lambda x: measurement(x)
    fxu = lambda x, u: hybrid_func(x, u, hybrid_pars)
    x = xGz0[:, 0]
    yval, xGval = [], []
    xGval.append(x)
    for t in range(Nval):
        yval.append(hx(x))
        x = fxu(x, uval[t, :].T)
        xGval.append(x)
    yval = np.asarray(yval)
    xGval = np.asarray(xGval)[:-1, :Ng]
    # Return.
    return yval, xGval

def check_hybrid_func(*, parameters, training_data, train):
    """ Get parameters for the hybrid function 
        and test by simulating on validation data. """

    # Get NN weights and parameters for the hybrid function.
    Np = train['Nps'][0]
    fnn_weights = train['trained_weights'][0][-1]
    xuscales = train['xuscales']
    hybrid_pars = get_hybrid_pars(parameters=parameters,
                                  Npast=Np,
                                  fnn_weights=fnn_weights,
                                  xuscales=xuscales)

    # Check the hybrid function.
    uval = training_data[-1].u
    ytfval = train['val_predictions'][0].y
    xGtfval = train['val_predictions'][0].x
    yval, xGval = sim_hybrid(hybrid_func, hybrid_pars, 
                             uval, training_data)

    # Return the parameters.
    return hybrid_pars

def get_mhe_noise_tuning(model_type, model_par):
    # Get MHE tuning.
    if model_type == 'plant' or model_type == 'hybrid':
        Qwx = 1e-6*np.eye(model_par['Nx'])
        Qwd = 1e-6*np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    if model_type == 'grey-box':
        Qwx = 1e-3*np.eye(model_par['Ng'])
        Qwd = np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    return (Qwx, Qwd, Rv)

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
    hybrid_pars = check_hybrid_func(parameters=parameters,
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
        mhe_noise_tuning = get_mhe_noise_tuning(model_type, model_par)
        plant = get_model(parameters=parameters, plant=True)
        controller = get_controller(model_ode, model_par, model_type,
                                    cost_pars, mhe_noise_tuning)
        cl_data, avg_stage_costs = online_simulation(plant, controller,
                                         plant_lyup=controller.lyup,
                                         Nsim=8*60, disturbances=disturbances,
                                         stdout_filename='tworeac_empc.txt')
        cl_data_list += [cl_data]
        avg_stage_costs_list += [avg_stage_costs]
    
    # Save data.
    PickleTool.save(data_object=dict(cl_data_list=cl_data_list,
                                     cost_pars=cost_pars,
                                     disturbances=disturbances,
                                     avg_stage_costs=avg_stage_costs_list),
                    filename='tworeac_empc_nonlin.pickle')

main()