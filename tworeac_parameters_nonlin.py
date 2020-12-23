# [depends] %LIB%/hybridid.py
# [makes] pickle
""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
import scipy.linalg
from hybridid import (PickleTool, NonlinearPlantSimulator, 
                      c2d, sample_prbs_like, SimData,
                      NonlinearMHEEstimator)

def _tworeac_plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 2D system. """
    # Extract the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']

    # Extract the plant states into meaningful names.
    (Ca, Cb, Cc) = x[0:3]
    Ca0 = u[0:1]
    tau = p[0:1]

    # Write the ODEs.
    dCabydt = (Ca0-Ca)/tau - k1*Ca
    dCbbydt = k1*Ca - 3*k2*(Cb**3) + 3*k3*Cc - Cb/tau
    dCcbydt = k2*(Cb**3) - k3*Cc - Cc/tau

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCcbydt])

def _tworeac_greybox_ode(x, u, p, parameters):
    """ Simple ODE describing the grey-box plant. """
    # Extract the parameters.
    k1 = parameters['k1']

    # Extract the plant states into meaningful names.
    (Ca, Cb) = x[0:2]
    Ca0 = u[0:1]
    tau = p[0:1]

    # Write the ODEs.
    dCabydt = (Ca0-Ca)/tau - k1*Ca
    dCbbydt = k1*Ca - Cb/tau

    # Return the derivative.
    return np.array([dCabydt, dCbbydt])

def _tworeac_measurement(x):
    # Return the measurement.
    return x[0:2]

def _get_tworeac_parameters():
    """ Get the parameter values for the 
        three reaction example. """
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 1. # m^3/min.
    parameters['k2'] = 0.01 # m^3/min.
    parameters['k3'] = 0.05 # m^3/min.

    # Store the dimensions.
    parameters['Nx'] = 3
    parameters['Ng'] = 2
    parameters['Nu'] = 1
    parameters['Ny'] = 2
    parameters['Np'] = 1

    # Sample time.
    parameters['sample_time'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0.5, 0.5]) # to be updated.
    parameters['us'] = np.array([1.]) # Ca0s
    parameters['ps'] = np.array([5.]) # tau (min)

    # Get the constraints. 
    ulb = np.array([0.5])
    uub = np.array([2.5])
    parameters['lb'] = dict(u=ulb)
    parameters['ub'] = dict(u=uub)

    # Number of time-steps to keep the plant at steady.
    parameters['tsteps_steady'] = 60

    # Measurement noise.
    parameters['Rv'] = np.diag([1e-3, 1e-3])

    # Return the parameters dict.
    return parameters

def _get_tworeac_rectified_xs(*, parameters):
    """ Get the steady state of the plant. """
    # (xs, us, ps)
    xs = parameters['xs']
    us = parameters['us']
    ps = parameters['ps']
    tworeac_plant_ode = lambda x, u, p: _tworeac_plant_ode(x, u, 
                                                      p, parameters)
    # Construct the casadi class.
    model = mpc.DiscreteSimulator(tworeac_plant_ode, 
                                  parameters['sample_time'],
                                  [parameters['Nx'], parameters['Nu'], 
                                   parameters['Np']], 
                                  ["x", "u", "p"])
    # Steady state of the plant.
    for _ in range(360):
        xs = model.sim(xs, us, ps)
    # Return the disturbances.
    return xs

def _get_tworeac_model(*, parameters, plant=True):
    """ Return a nonlinear plant simulator object."""
    if plant:
        # Construct and return the plant.
        tworeac_plant_ode = lambda x, u, p: _tworeac_plant_ode(x, u, 
                                                               p, parameters)
        xs = parameters['xs'][:, np.newaxis]
        return NonlinearPlantSimulator(fxup = tworeac_plant_ode,
                                        hx = _tworeac_measurement,
                                        Rv = parameters['Rv'], 
                                        Nx = parameters['Nx'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                    sample_time = parameters['sample_time'], 
                                        x0 = xs)
    else:
        # Construct and return the grey-box model.
        tworeac_greybox_ode = lambda x, u, p: _tworeac_greybox_ode(x, u, 
                                                               p, parameters)
        xs = parameters['xs'][:-1, np.newaxis]
        return NonlinearPlantSimulator(fxup = tworeac_greybox_ode,
                                        hx = _tworeac_measurement,
                                        Rv = 0*np.eye(parameters['Ny']), 
                                        Nx = parameters['Ng'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                    sample_time = parameters['sample_time'], 
                                        x0 = xs)

def _gen_train_val_data(*, parameters, num_traj, 
                           Nsim_train, Nsim_trainval, 
                           Nsim_val, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    data_list = []
    ulb = parameters['lb']['u']
    uub = parameters['ub']['u']
    tsteps_steady = parameters['tsteps_steady']
    p = parameters['ps'][:, np.newaxis]

    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = _get_tworeac_model(parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tsteps_steady, 1))
        
        # Get input trajectories for different simulatios.
        if traj == num_traj-1:
            "Get input for train val simulation."
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=24, num_steps=Nsim_val, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+1)
        elif traj == num_traj-2:
            "Get input for validation simulation."
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=6, num_steps=Nsim_trainval, 
                                 lb=ulb, ub=uub,
                                 mean_change=20, sigma_change=2, seed=seed+2)
        else:
            "Get input for training simulation."
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=48, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+3)

        # Complete input profile and run open-loop simulation.
        u = np.concatenate((us_init, u), axis=0)
        for t in range(tsteps_steady + Nsim):
            plant.step(u[t:t+1, :], p)
        data_list.append(SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u).squeeze(),
                y=np.asarray(plant.y[0:-1]).squeeze()))
    # Return the data list.
    return data_list

def _get_greybox_val_preds(*, parameters, training_data):
    """ Use the input profile to compute 
        the prediction of the grey-box model
        on the validation data. """
    model = _get_tworeac_model(parameters=parameters, plant=False)
    tsteps_steady = parameters['tsteps_steady']
    p = parameters['ps'][:, np.newaxis]
    u = training_data[-1].u[:, np.newaxis]
    Nsim = u.shape[0]
    # Run the open-loop simulation.
    for t in range(Nsim):
        model.step(u[t:t+1, :], p)
    data = SimData(t=None,
                   x=None,
                   u=None,
                   y=np.asarray(model.y[tsteps_steady:-1]).squeeze())
    return data

def _get_mhe_estimator(*, parameters):
    """ Filter the training data using a combination 
        of grey-box model and an input disturbance model. """

    #ef get_state_estimates(filter, y, uprev, Nx):
    #    """Use the filter object to perform state estimation."""
    #    return np.split(filter.solve(y, uprev), [Nx])

    def state_space_model(Ng, Nd, ps, parameters):
        """ Augmented state-space model for moving horizon estimation. """
        return lambda x, u : np.concatenate((_tworeac_plant_ode(x[:Ng], 
                                                          u, ps, parameters),
                                             np.zeros((Nd,))), axis=0)
    
    def measurement_model(Ng):
        """ Augmented measurement model for moving horizon estimation. """
        return lambda x : _tworeac_measurement(x[:Ng])

    # Get sizes.
    (Ng, Nu, Ny) = (parameters['Nx'], parameters['Nu'], parameters['Ny'])
    Nd = Ny

    # Get the disturbance model.
    Bd = np.zeros((Ng, Nd))
    Bd[0, 0] = 1.
    Bd[1, 1] = 1.
    #Bd[6, 2] = 1.
    #Bd[7, 3] = 1.
    Cd = np.ones((Ny, Nd))

    # Initial states.
    xs = parameters['xs'][:, np.newaxis]
    ps = parameters['ps'][:, np.newaxis]
    us = parameters['us'][:, np.newaxis]
    ys = _tworeac_measurement(xs)
    ds = np.zeros((Nd, 1))

    # Noise covariances.
    Qwx = np.eye(Ng)
    Qwd = np.eye(Nd)
    Rv = np.eye(Ny)

    # MHE horizon length.
    Nmhe = 10

    # Continuous time functions, fxu and hx.
    fxud = state_space_model(Ng, Nd, ps, parameters)
    hxd = measurement_model(Ng)
    
    # Initial data.
    xprior = np.concatenate((xs, ds), axis=0)
    xprior = np.repeat(xprior.T, Nmhe, axis=0)
    u = np.repeat(us.T, Nmhe, axis=0)
    y = np.repeat(ys.T, Nmhe+1, axis=0)
    
    # Penalty matrices.
    Qwxinv = np.linalg.inv(Qwx)
    Qwdinv = np.linalg.inv(Qwd)
    Qwinv = scipy.linalg.block_diag(Qwxinv, Qwdinv)
    P0inv = Qwinv
    Rvinv = np.linalg.inv(Rv)

    # Get the augmented models.
    fxu = mpc.getCasadiFunc(fxud, [Ng+Nd, Nu], ["x", "u"],
                            rk4=True, Delta=parameters['sample_time'], 
                            M=10)
    hx = mpc.getCasadiFunc(hxd, [Ng+Nd], ["x"])
    
    # Create a filter object and return.
    return NonlinearMHEEstimator(fxu=fxu, hx=hx, 
                                 Nmhe=Nmhe, Nx=Ng+Nd, 
                                 Nu=Nu, Ny=Ny,
                                 xprior=xprior, 
                                 u=u, y=y, 
                                 P0inv=P0inv, 
                                 Qwinv=Qwinv, Rvinv=Rvinv)

def main():
    """ Get the parameters/training/validation data."""
    
    # Get parameters.
    parameters = _get_tworeac_parameters()
    parameters['xs'] = _get_tworeac_rectified_xs(parameters=parameters)
    
    # Generate training data.
    training_data = _gen_train_val_data(parameters=parameters, 
                                        num_traj=3, Nsim_train=1440,
                                        Nsim_trainval=120, Nsim_val=720, 
                                        seed=100)
    greybox_val_data = _get_greybox_val_preds(parameters=
                                            parameters, 
                                            training_data=training_data)
    _get_mhe_estimator(parameters=parameters)
    # Create a dict and save.
    tworeac_parameters = dict(parameters=parameters, 
                              training_data=training_data,
                              greybox_validation_data=greybox_val_data)
    # Save data.
    PickleTool.save(data_object=tworeac_parameters, 
                    filename='tworeac_parameters_nonlin.pickle')

main()