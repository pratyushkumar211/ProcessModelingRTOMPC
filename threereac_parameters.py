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
                      sample_prbs_like, 
                      PlantSimData)

def _threereac_plant_ode(x, u, p, parameters):

    # Extract the parameters
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    V = parameters['V']

    # Extract the plant states into meaningful names.
    (Ca, Cb, Cc, Cd) = x[0:4]
    F = u[0:1]
    Ca0 = p[0:1]

    # Write the ODEs.
    dCabydt = F*(Ca0-Ca)/V - k1*Ca
    dCbbydt = k1*Ca - k2*Cb -2*k3*(Cb**2) - F*Cb/V
    dCcbydt = k2*Cb - F*Cc/V
    dCdbydt = k3*(Cb**2) - F*Cd/V

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCcbydt, dCdbydt])

#def _threereac_grey_box_ode(x, u, p, parameters):

    # Extract the parameters.
#    k1 = parameters['k1']
#    beta = parameters['beta']
#    F = parameters['F']
#    Vr = parameters['Vr']

    # Extract the plant states into meaningful names.
#    (Ca, Cd, Cc) = x[0:3]
#    Ca0 = u[0]

    # Define a constant.
#    sqrtoneplusbetaCa = np.sqrt(1 + beta*Ca)

    # Write the ODEs.
#    dCabydt = F*(Ca0-Ca)/Vr - k1*Ca
#    dCdbydt = 0.5*k1*Ca*(-1+sqrtoneplusbetaCa)/(1+sqrtoneplusbetaCa) - F*Cd/Vr
#    dCcbydt = 2*k1*Ca/(1 + sqrtoneplusbetaCa) - F*Cc/Vr

    # Return the derivative.
#    return np.array([dCabydt, dCdbydt, dCcbydt])

def _threereac_measurement(x):
    """ The measurement function."""
    # Return the measurements.
    return x[-2:-1]

def _get_threereac_parameters():
    """ Get the parameter values for the 
        three reaction example. """
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 1. # m^3/min.
    parameters['k2'] = 1e+2 # m^3/min.
    parameters['k3'] = 1e+5 # m^3/min.
    parameters['V'] = 3. # m^3 

    #parameters['beta'] = 16.
    #parameters['beta'] = 8*parameters['k1']*parameters['k3']
    #parameters['beta'] = parameters['beta']/(parameters['k2']**2)
    #parameters['F'] = 0.1 # m^3/min.

    # Store the dimensions.
    parameters['Nx'] = 4
    parameters['Nu'] = 1
    parameters['Ny'] = 1
    parameters['Np'] = 1

    # Sample time.
    parameters['sample_time'] = 1.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0., 0.5, 0.5]) # to be updated.
    parameters['us'] = np.array([0.5])
    parameters['ps'] = np.array([1.])

    # Get the constraints. 
    ulb = np.array([0.])
    uub = np.array([1.])
    parameters['lb'] = dict(u=ulb)
    parameters['ub'] = dict(u=uub)

    # Measurement noise.
    parameters['Rv'] = 1e-20*np.array([[1e-4]])

    # Return the parameters dict.
    return parameters

def _get_threereac_rectified_xs(*, parameters):
    """ Get the steady state of the plant. """
    # (xs, us, ps)
    xs = parameters['xs']
    us = parameters['us']
    ps = parameters['ps']
    threereac_plant_ode = lambda x, u, p: _threereac_plant_ode(x, u, 
                                                        p, parameters)
    # Construct the casadi class.
    model = mpc.DiscreteSimulator(threereac_plant_ode, 
                                  parameters['sample_time'],
                                  [parameters['Nx'], parameters['Nu'], 
                                   parameters['Np']], 
                                  ["x", "u", "p"])
    # Steady state of the plant.
    for _ in range(360):
        xs = model.sim(xs, us, ps)
    # Return the disturbances.
    return xs

def _get_threereac_plant(*, parameters):
    """ Return a nonlinear plant simulator object."""
    # Construct and return the plant.
    threereac_plant_ode = lambda x, u, p: _threereac_plant_ode(x, u, 
                                                               p, parameters)
    xs = parameters['xs'][:, np.newaxis]
    return NonlinearPlantSimulator(fxup = threereac_plant_ode,
                                    hx = _threereac_measurement,
                                    Rv = parameters['Rv'], 
                                    Nx = parameters['Nx'], 
                                    Nu = parameters['Nu'], 
                                    Np = parameters['Np'], 
                                    Ny = parameters['Ny'],
                                    sample_time = parameters['sample_time'], 
                                    x0 = xs)

def _generate_training_data(*, parameters, num_trajectories, Nsim, seed):
    """ Simulate the plant model 
        and generate training and validation data. """
    # Get the data list.
    datum = []
    ulb = parameters['lb']['u']
    uub = parameters['ub']['u']
    p = parameters['ps'][:, np.newaxis]
    xs = parameters['xs'][:, np.newaxis]
    for _ in range(num_trajectories):
        plant = _get_threereac_plant(parameters=parameters)
        u = sample_prbs_like(num_change=4, num_steps=Nsim, 
                             lb=ulb, ub=uub,
                             mean_change=60, sigma_change=5, seed=seed+1)
        # Run the open-loop simulation.
        for t in range(Nsim):
            plant.step(u[t:t+1, :], p)
        datum.append(PlantSimData(time=np.asarray(plant.t[0:-1]).squeeze(),
                Ca=np.asarray(plant.x[0:-1]).squeeze()[:, 0],
                Cb=np.asarray(plant.x[0:-1]).squeeze()[:, 1],
                Cc=np.asarray(plant.y[0:-1]).squeeze(),
                Cd=np.asarray(plant.x[0:-1]).squeeze()[:, 3],
                F=np.asarray(plant.u).squeeze()))
    # Return the data list.
    return datum

#def _get_grey_box_val_predictions(*, uval, parameters):
#    """ Use the input profile to compute 
#        the prediction of the grey-box model
#        on the validation data. """
#    threereac_grey_box_ode = lambda x, u, p: _threereac_grey_box_ode(x, u, 
#                                                               p, parameters)
#    xs = parameters['xs'][(0, 2, 3), np.newaxis]
#    p = np.zeros((parameters['Np'], 1))
#    model = NonlinearPlantSimulator(fxup = threereac_grey_box_ode,
#                                    hx = _threereac_measurement,
#                                    Rv = parameters['Rv'], 
#                                    Nx = parameters['Nx']-1, 
#                                    Nu = parameters['Nu'], 
#                                    Np = parameters['Np'], 
#                                    Ny = parameters['Ny'],
#                                    sample_time = parameters['sample_time'], 
#                                    x0 = xs)
#    uval = uval[..., np.newaxis]
#    Nsim = uval.shape[0]
    # Run the open-loop simulation.
#    for ut in uval:
#        model.step(ut, p)
#    data = ModelSimData(time=np.asarray(model.t[0:-1]).squeeze(),
#                Ca=np.asarray(model.x[0:-1]).squeeze()[:, 0],
#                Cc=np.asarray(model.y[0:-1]).squeeze(),
#                Cd=np.asarray(model.x[0:-1]).squeeze()[:, 1],
#                Ca0=np.asarray(model.u).squeeze())
#    return data

if __name__ == "__main__":
    """Compute parameters for the three reactions."""
    parameters = _get_threereac_parameters()
    parameters['xs'] = _get_threereac_rectified_xs(parameters=parameters)
    training_data = _generate_training_data(parameters=parameters, 
                                    num_trajectories=1, Nsim=240, seed=4)
    #greybox_val_data = _get_grey_box_val_predictions(uval=input_profiles[-1], 
    #                                                 parameters=parameters)
    threereac_parameters = dict(parameters=parameters, 
                                training_data=training_data)
    # Save data.
    PickleTool.save(data_object=threereac_parameters, 
                    filename='threereac_parameters.pickle')