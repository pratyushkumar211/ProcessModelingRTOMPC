# [makes] pickle
""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
import collections
from hybridid import (PickleTool, NonlinearPlantSimulator, 
                      sample_prbs_like, ModelSimData,
                      PlantSimData)

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
    dCbbydt = k1*Ca - k2*Cb + k3*Cc - Cb/tau
    dCcbydt = k2*Cb - k3*Cc - Cc/tau

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
    parameters['k1'] = 1 # m^3/min.
    parameters['k2'] = 1e-2 # m^3/min.
    parameters['k3'] = 1e-3 # m^3/min.

    #parameters['tau'] = 2. # m^3 
    #parameters['k3'] = 0.05 # m^3/min.
    #parameters['beta'] = 16.
    #parameters['beta'] = 8*parameters['k1']*parameters['k3']
    #parameters['beta'] = parameters['beta']/(parameters['k2']**2)
    #parameters['F'] = 0.1 # m^3/min.

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
    parameters['ps'] = np.array([20.]) # tau # min.

    # Get the constraints. 
    ulb = np.array([0.5])
    uub = np.array([2.5])
    parameters['lb'] = dict(u=ulb)
    parameters['ub'] = dict(u=uub)

    # Measurement noise.
    parameters['Rv'] = 1e-20*np.array([[1e-4]])

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
                                        Rv = parameters['Rv'], 
                                        Nx = parameters['Ng'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                    sample_time = parameters['sample_time'], 
                                        x0 = xs)

def _generate_training_data(*, parameters, 
                               num_trajectories, Nsim, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    datum = []
    ulb = parameters['lb']['u']
    uub = parameters['ub']['u']
    p = parameters['ps'][:, np.newaxis]
    xs = parameters['xs'][:, np.newaxis]
    for _ in range(num_trajectories):
        plant = _get_tworeac_model(parameters=parameters, plant=True)
        u = sample_prbs_like(num_change=3, num_steps=Nsim, 
                             lb=ulb, ub=uub,
                             mean_change=120, sigma_change=2, seed=seed+1)
        # Run the open-loop simulation.
        for t in range(Nsim):
            plant.step(u[t:t+1, :], p)
        datum.append(PlantSimData(time=np.asarray(plant.t[0:-1]).squeeze(),
                Ca=np.asarray(plant.y[0:-1]).squeeze()[:, 0],
                Cb=np.asarray(plant.y[0:-1]).squeeze()[:, 1],
                Cc=np.asarray(plant.x[0:-1]).squeeze()[:, 2],
                Ca0=np.asarray(plant.u).squeeze()))
    # Return the data list.
    return datum

def _get_greybox_validation_predictions(*, parameters, training_data):
    """ Use the input profile to compute 
        the prediction of the grey-box model
        on the validation data. """
    model = _get_tworeac_model(parameters=parameters, plant=False)
    p = parameters['ps'][:, np.newaxis]
    u = training_data[-1].Ca0[:, np.newaxis]
    Nsim = u.shape[0]
   # Run the open-loop simulation.
    for t in range(Nsim):
        model.step(u[t:t+1, :], p)
    data = ModelSimData(time=np.asarray(model.t[0:-1]).squeeze(),
                Ca=np.asarray(model.x[0:-1]).squeeze()[:, 0],
                Cb=np.asarray(model.x[0:-1]).squeeze()[:, 1],
                Ca0=np.asarray(model.u).squeeze())
    return data

def _check_cont_time_obsv(*, parameters):
    """ Check the observability of the continuous time linear system. """
    # Measurement matrix for the plant.
    C = np.array([[1., 0., 0.], 
                  [0., 1., 0.]])
    tau = parameters['ps'][0]
    (k1, k2, k3) = (parameters['k1'], parameters['k2'], parameters['k3'])
    A = np.array([[-k1-(1/tau), 0., 0.], 
                  [k1, -2*k2 - (1/tau), k3], 
                  [0., k2, -k3 -(1/tau)]])
    Obsv = []
    for i in range(parameters['Nx']):
        Obsv.append(C @ np.linalg.matrix_power(A, i))
    Obsv = np.concatenate(Obsv, axis=0)
    print("Rank of the continuous time Observability matrix is: " + 
            str(np.linalg.matrix_rank(Obsv))) 
    return

def main():
    """Compute parameters for the three reactions."""
    parameters = _get_tworeac_parameters()
    parameters['xs'] = _get_tworeac_rectified_xs(parameters=parameters)
    _check_cont_time_obsv(parameters=parameters)
    training_data = _generate_training_data(parameters=parameters, 
                                        num_trajectories=1, Nsim=360, seed=1)
    greybox_validation_data = _get_greybox_validation_predictions(parameters=
                                            parameters, 
                                            training_data=training_data)
    tworeac_parameters = dict(parameters=parameters, 
                                training_data=training_data,
                                greybox_validation_data=greybox_validation_data)
    # Save data.
    PickleTool.save(data_object=tworeac_parameters, 
                    filename='tworeac_parameters.pickle')

main()