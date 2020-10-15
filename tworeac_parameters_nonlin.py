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
from hybridid import (PickleTool, NonlinearPlantSimulator, 
                      c2d, sample_prbs_like, SimData)

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
    parameters['Rv'] = 0*np.diag([1e-4, 1e-3])

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

def _gen_train_val_data(*, parameters,
                           num_traj, Nsim, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    data_list = []
    ulb = parameters['lb']['u']
    uub = parameters['ub']['u']
    tsteps_steady = parameters['tsteps_steady']
    p = parameters['ps'][:, np.newaxis]
    for _ in range(num_traj):
        plant = _get_tworeac_model(parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tsteps_steady, 1))
        u = sample_prbs_like(num_change=6, num_steps=Nsim, 
                             lb=ulb, ub=uub,
                             mean_change=30, sigma_change=2, seed=seed+1)
        u = np.concatenate((us_init, u), axis=0)
        # Run the open-loop simulation.
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

def main():
    """ Get the parameters/training/validation data."""
    # Get parameters.
    parameters = _get_tworeac_parameters()
    parameters['xs'] = _get_tworeac_rectified_xs(parameters=parameters)
    # Generate training data.
    training_data = _gen_train_val_data(parameters=parameters, 
                                        num_traj=3, Nsim=180, seed=100)
    greybox_val_data = _get_greybox_val_preds(parameters=
                                            parameters, 
                                            training_data=training_data)
    tworeac_parameters = dict(parameters=parameters, 
                              training_data=training_data,
                              greybox_validation_data=greybox_val_data)
    # Save data.
    PickleTool.save(data_object=tworeac_parameters, 
                    filename='tworeac_parameters_nonlin.pickle')

main()