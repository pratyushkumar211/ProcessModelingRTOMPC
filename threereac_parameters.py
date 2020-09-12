""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
import collections
from hybridid import (PickleTool, NonlinearPlantSimulator)

SystemIdData = collections.namedtuple('SystemIdData', 
                    ['time', 'Ca', 'Cb', 'Cc', 
                     'Cd', 'Ca0'])

GreyBoxSimData = collections.namedtuple('GreyBoxSimData', 
                    ['time', 'Ca', 'Cc', 'Cd', 'Ca0'])

def _threereac_plant_ode(x, u, p, parameters):

    # Extract the parameters
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    F = parameters['F']
    Vr = parameters['Vr']

    # Extract the plant states into meaningful names.
    (Ca, Cb, Cd, Cc) = x[0:4]
    Ca0 = u[0]

    # Write the ODEs.
    dCabydt = F*(Ca0-Ca)/Vr - k1*Ca
    dCbbydt = k1*Ca - k2*Cb -2*k3*(Cb**2) - F*Cb/Vr
    dCdbydt = k3*(Cb**2) - F*Cd/Vr
    dCcbydt = k2*Cb - F*Cc/Vr

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCdbydt, dCcbydt])

def _threereac_qssa_ode(x, u, p, parameters):

    # Extract the parameters.
    k1 = parameters['k1']
    beta = parameters['beta']
    F = parameters['F']
    Vr = parameters['Vr']

    # Extract the plant states into meaningful names.
    (Ca, Cd, Cc) = x[0:3]
    Ca0 = u[0:1]

    # Define a constant.
    sqrtoneplusbetaCa = np.sqrt(1 + beta*Ca)

    # Write the ODEs.
    dCabydt = F*(Ca0-Ca)/Vr - k1*Ca
    dCdbydt = 0.5*k1*Ca*(-1+sqrtoneplusbetaCa)/(1+sqrtoneplusbetaCa) - F*Cd/Vr
    dCcbydt = 2*k1*Ca/(1 + sqrtoneplusbetaCa) - F*Cc/Vr

    # Return the derivative.
    return np.array([dCabydt, dCdbydt, dCcbydt])

def _threereac_measurement(x):
    """ The measurement function."""
    # Return the measurements.
    return x[-1:]

def _get_threereac_parameters():
    """ Get the parameter values for the 
        three reaction example
        The sample time is in minutes."""

    # Sample time and state dimensions.
    sample_time = 1. 
    (Nx, Nu, Np, Ny) = (4, 1, 1, 1)
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 1.
    parameters['k2'] = 1e+2
    parameters['k3'] = 2e+4
    parameters['beta'] = 16.
    parameters['F'] = 0.1 # m^3/min.
    parameters['Vr'] = 1. # m^3 

    # Store the dimensions.
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample time.
    parameters['sample_time'] = sample_time

    # Get the steady states.
    parameters['xs'] = np.array([1., 0., 0.5, 0.5])
    parameters['us'] = np.array([1.])

    # Get the constraints. 
    ulb = np.array([0.5])
    uub = np.array([1.5])
    parameters['lb'] = dict(u=ulb)
    parameters['ub'] = dict(u=uub)

    # Measurement noise.
    parameters['Rv'] = 1e-20*np.array([[1e-2]])

    # Return the parameters dict.
    return parameters

def _get_threereac_rectified_xs(*, parameters):
    """ Get the steady state of the plant. """
    # (xs, us, ps)
    xs = parameters['xs']
    us = parameters['us']
    ps = np.zeros((parameters['Np'],))
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

def _get_cstrs_plant(*, parameters):
    """ Return a Nonlinear Plant Simulator object."""
    # Construct and Return the Plant.
    threereac_plant_ode = lambda x, u, p: _threereac_plant_ode(x, u, 
                                                               p, parameters)
    threereac_plant_measurement = lambda x: _threereac_measurement(x, 
                                                            parameters)
    return NonlinearPlantSimulator(fxup = threereac_plant_ode,
                                    hx = threereac_plant_measurement,
                                    Rv = parameters['Rv'], 
                                    Nx = parameters['Nx'], 
                                    Nu = parameters['Nu'], 
                                    Np = parameters['Np'], 
                                    Ny = parameters['Ny'],
                                    sample_time = parameters['sample_time'], 
                                    x0 = xs)

def _generate_id_data():



    return 

def _compute_grey_box_predictions():


    return

if __name__ == "__main__":
    """ Compute parameters for the three reactions. """
    parameters = _get_threereac_parameters()
    parameters['xs'] = _get_threereac_rectified_xs(parameters=parameters)

    _generate_id_data()

    # Save data.
    PickleTool.save(data_object=threereac_parameters, 
                    filename='threereac_parameters.pickle')