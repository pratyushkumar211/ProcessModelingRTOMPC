# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
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
from hybridid import PickleTool, c2d, sample_prbs_like, SimData
from linNonlinMPC import NonlinearPlantSimulator

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
    Ca0 = u[0]
    tau = p[0]

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
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0.5, 0.5]) # to be updated.
    parameters['us'] = np.array([1.5]) # Ca0s
    parameters['ps'] = np.array([10.]) # tau (min)

    # Get the constraints. 
    ulb = np.array([0.5])
    uub = np.array([2.5])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Number of time-steps to keep the plant at steady.
    parameters['tsteps_steady'] = 10

    # Measurement noise.
    parameters['Rv'] = 0*np.diag([1e-3, 1e-3])

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
                                  parameters['Delta'],
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
                                    sample_time = parameters['Delta'], 
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
                                    sample_time = parameters['Delta'], 
                                        x0 = xs)

def _gen_train_val_data(*, parameters, num_traj, 
                           Nsim_train, Nsim_trainval, 
                           Nsim_val, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    data_list = []
    ulb = parameters['ulb']
    uub = parameters['uub']
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
            u = sample_prbs_like(num_change=3, num_steps=Nsim_trainval, 
                                 lb=ulb, ub=uub,
                                 mean_change=20, sigma_change=2, seed=seed+2)
        else:
            "Get input for training simulation."
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=2, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+3)

        # Complete input profile and run open-loop simulation.
        u = np.concatenate((us_init, u), axis=0)
        for t in range(tsteps_steady + Nsim):
            plant.step(u[t:t+1, :], p)
        data_list.append(SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u).squeeze(axis=-1),
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
    u = training_data[-1].u
    Nsim = u.shape[0]
    # Run the open-loop simulation.
    for t in range(Nsim):
        model.step(u[t:t+1, :], p)
    x = np.asarray(model.x[0:-1]).squeeze()
    x = np.insert(x, [2], np.nan*np.ones((Nsim, 1)), axis=1)
    data = SimData(t=np.asarray(model.t[0:-1]), x=x, u=u,
                   y=np.asarray(model.y[:-1]).squeeze())
    # Return daa.
    return data

def main():
    """ Get the parameters/training/validation data."""
    
    # Get parameters.
    parameters = _get_tworeac_parameters()
    parameters['xs'] = _get_tworeac_rectified_xs(parameters=parameters)
    
    # Generate training data.
    training_data = _gen_train_val_data(parameters=parameters, 
                                        num_traj=8, Nsim_train=60,
                                        Nsim_trainval=60, Nsim_val=720, 
                                        seed=100)
    greybox_val_data = _get_greybox_val_preds(parameters=
                                            parameters, 
                                            training_data=training_data)

    # Create a dict and save.
    tworeac_parameters = dict(parameters=parameters, 
                              training_data=training_data,
                              greybox_validation_data=greybox_val_data)
    # Save data.
    PickleTool.save(data_object=tworeac_parameters, 
                    filename='tworeac_parameters_nonlin.pickle')

main()