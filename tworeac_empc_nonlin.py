# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] tworeac_parameters_nonlin.py
# [depends] tworeac_parameters_nonlin.pickle
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
from hybridid import (PickleTool, SimData, _resample_fast, c2dNonlin,
                     interpolate_yseq)
from linNonlinMPC import (NonlinearPlantSimulator, NonlinearEMPCController, 
                         online_simulation)
from tworeac_parameters_nonlin import _tworeac_plant_ode as _plant_ode
from tworeac_parameters_nonlin import _tworeac_greybox_ode as _greybox_ode
from tworeac_parameters_nonlin import _tworeac_measurement as _measurement

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
    hx = lambda x: _measurement(x)

    # Get the state dimension.
    if model_type == 'grey-box':
        Nx = model_pars['Ng']
    else:
        Nx = model_pars['Nx']

    # Get the stage cost.
    lxup = lambda x, u, p: stage_cost(x, u, p)
    
    # Get the sizes/sample time.
    Nu, Ny = model_pars['Nu'], model_pars['Ny']
    Nd = Ny

    # Get the disturbance models.
    Bd = np.zeros((Nx, Nd))
    if model_type == 'plant':
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
                                   lxup=lxup, Bd=Bd, Cd=Cd,
                                   Nx=Nx, Nu=Nu, Ny=Ny, Nd=Nd,
                                   xs=xs, us=us, ds=ds, ys=ys,
                                   empc_pars=cost_pars,
                                   ulb=ulb, uub=uub, Nmpc=Nmpc,
                                   Qwx=Qwx, Qwd=Qwd, Rv=Rv, Nmhe=Nmhe)

def get_hybrid_pars(*, parameters, Npast, fnn_weights, xuscales):
    """ Get the hybrid model parameters. """

    hybrid_pars = copy.deepcopy(parameters)
    # Update sizes.
    Ng, Nu, Ny = parameters['Ng'], parameters['Nu'], parameters['Ny']
    hybrid_pars['Nx'] = parameters['Ng'] + Npast*(Nu + Ny)

    # Update steady state.
    ys = _measurement(parameters['xs'])
    yspseq = np.tile(ys, (Npast, ))
    us = parameters['us']
    uspseq = np.tile(us, (Npast, ))
    xs = parameters['xs'][:Ng]
    hybrid_pars['xs'] = np.concatenate((xs, yspseq, uspseq))
    
    # NN pars.
    hybrid_pars['Npast'] = Npast
    hybrid_pars['fnn_weights'] = fnn_weights 

    # Scaling.
    hybrid_pars['xuscales'] = xuscales

    # Return.
    return hybrid_pars

def _fnn(xGz, Npast, fnn_weights):
    """ Compute the NN output. """
    nn_output = xGz[:, np.newaxis]
    for i in range(0, len(fnn_weights)-2, 2):
        (W, b) = fnn_weights[i:i+2]
        nn_output = np.tanh(W.T @ nn_output + b[:, np.newaxis])
    Wf = fnn_weights[-1]
    nn_output = (Wf.T @ nn_output)[:, 0]
    # Return.
    return nn_output

def _hybrid_func(xGz, u, parameters):
    """ The augmented continuous time model. """

    # Extract a few parameters.
    Ng = parameters['Ng']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    ps = parameters['ps']
    Npast = parameters['Npast']
    Delta = parameters['Delta']
    fnn_weights = parameters['fnn_weights']
    xuscales = parameters['xuscales']
    xmean, xstd = xuscales['xscale']
    umean, ustd = xuscales['uscale']
    xGzmean = np.concatenate((xmean,
                               np.tile(xmean, (Npast, )), 
                               np.tile(umean, (Npast, ))))
    xGzstd = np.concatenate((xstd,
                             np.tile(xstd, (Npast, )), 
                             np.tile(ustd, (Npast, ))))
    
    # Extract vectors.
    #xGz = (xGz - xGzmean)/xGzstd 
    #u = (u - umean)/ustd
    xG, ypseq, upseq = xGz[:Ng], xGz[Ng:Ng+Npast*Ny], xGz[-Npast*Nu:]
    
    # Get k1, k2, k3, and k4.
    k1 = _greybox_ode(xG, u, ps, parameters)
    k2 = _greybox_ode(xG + Delta*(k1/2), u, ps, parameters)
    k3 = _greybox_ode(xG + Delta*(k2/2), u, ps, parameters)
    k4 = _greybox_ode(xG + Delta*k3, u, ps, parameters)
    xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
    xGplus = (xGplus-xmean)/xstd

    # Scale for NN input.
    xGz = (xGz - xGzmean)/xGzstd 
    xGplus += _fnn(xGz, Npast, fnn_weights)
    
    # Get the state and the next time step.
    zplus = np.concatenate((ypseq[Ny:], xG, upseq[Nu:], u))
    xGplus = xGplus*xstd + xmean
    xGzplus = np.concatenate((xGplus, zplus))

    # Return the sum.
    return xGzplus

def get_plant(*, parameters):
    """ Return a nonlinear plant simulator object. """
    measurement = lambda x: _measurement(x)
    # Construct and return the plant.
    plant_ode = lambda x, u, p: _plant_ode(x, u, p, parameters)
    xs = parameters['xs'][:, np.newaxis]
    return NonlinearPlantSimulator(fxup = plant_ode,
                                    hx = measurement,
                                    Rv = parameters['Rv'], 
                                    Nx = parameters['Nx'], 
                                    Nu = parameters['Nu'], 
                                    Np = parameters['Np'], 
                                    Ny = parameters['Ny'],
                                    sample_time = parameters['Delta'], 
                                    x0 = xs)

def stage_cost(x, u, p):
    """ Custom stage cost for the tworeac system. """    
    # Get inputs, parameters, and states.
    CAf = u[0:1]
    ca, cb = p[0:2]
    CA, CB = x[0:2]
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
    hx = lambda x: _measurement(x)
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

def get_mhe_noise_tuning(model_type, model_par):
    # Get MHE tuning.
    if model_type == 'plant':
        Qwx = 1e-6*np.eye(model_par['Nx'])
        Qwd = 1e-6*np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    elif model_type == 'grey-box':
        Qwx = 1e-3*np.eye(model_par['Ng'])
        Qwd = np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    else:
        Qwx = 1e-3*np.eye(model_par['Nx'])
        Qwd = np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    return (Qwx, Qwd, Rv)

def get_tworeac_empc_pars(*, Delta):
    """ Get economic MPC parameters for the tworeac example. """
    raw_mat_price = _resample_fast(x = np.array([[105.], [105.], 
                                                 [105.], [105.], 
                                                 [105.]]), 
                                   xDelta=2*60,
                                   newDelta=Delta,
                                   resample_type='zoh')
    product_price = _resample_fast(x = np.array([[170.], [200.], 
                                                 [120.], [160.], 
                                                 [160.]]),
                                   xDelta=2*60,
                                   newDelta=Delta,
                                   resample_type='zoh')
    cost_pars = 100*np.concatenate((raw_mat_price, product_price), axis=1)
    # Return the cost pars.
    return cost_pars

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters_nonlin = PickleTool.load(filename=
                                            'tworeac_parameters_nonlin.pickle',
                                            type='read')
    tworeac_train_nonlin = PickleTool.load(filename=
                                            'tworeac_train_nonlin.pickle',
                                            type='read')

    # Get parameters.
    parameters = tworeac_parameters_nonlin['parameters']
    cost_pars = get_tworeac_empc_pars(Delta=parameters['Delta'])
    Nsim = cost_pars.shape[0]
    disturbances = np.repeat(parameters['ps'][np.newaxis, :], Nsim)

    # Get NN weights and the hybrid ODE.
    Np = tworeac_train_nonlin['Nps'][1]
    fnn_weights = tworeac_train_nonlin['trained_weights'][0][0]
    xuscales = tworeac_train_nonlin['xuscales']
    hybrid_pars = get_hybrid_pars(parameters=parameters,
                                  Npast=Np,
                                  fnn_weights=fnn_weights,
                                  xuscales=xuscales)

    # Check the hybrid function.
    uval = tworeac_parameters_nonlin['training_data'][-1].u
    ytfval = tworeac_train_nonlin['val_predictions'][0].y
    xGtfval = tworeac_train_nonlin['val_predictions'][0].x
    training_data = tworeac_parameters_nonlin['training_data']
    yval, xGval = sim_hybrid(_hybrid_func, hybrid_pars, 
                             uval, training_data)

    # Run simulations for different model.
    cl_data_list, avg_stage_costs_list, openloop_sol_list = [], [], []
    model_odes = [_plant_ode, _hybrid_func]
    model_pars = [parameters, hybrid_pars]
    model_types = ['plant', 'hybrid']
    for (model_ode,
         model_par, model_type) in zip(model_odes, model_pars, model_types):
        mhe_noise_tuning = get_mhe_noise_tuning(model_type, model_par)
        plant = get_plant(parameters=parameters)
        controller = get_controller(model_ode, model_par, model_type,
                                    cost_pars, mhe_noise_tuning)
        cl_data, avg_stage_costs, openloop_sol = online_simulation(plant,
                                         controller,
                                         plant_lxup=controller.lxup,
                                         Nsim=8*60, disturbances=disturbances,
                                         stdout_filename='tworeac_empc.txt')
        cl_data_list += [cl_data]
        avg_stage_costs_list += [avg_stage_costs]
        openloop_sol_list += [openloop_sol]
    
    # Save data.
    PickleTool.save(data_object=dict(cl_data_list=cl_data_list,
                                     cost_pars=cost_pars,
                                     disturbances=disturbances,
                                     avg_stage_costs=avg_stage_costs_list,
                                     openloop_sols=openloop_sol_list),
                    filename='tworeac_empc_nonlin.pickle')

main()