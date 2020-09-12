""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool

def _threereac_plant_ode(x, u, p, parameters):

    # Extract the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    Ca0 = parameters['Ca0']
    A = parameters['A']

    # Extract the plant states into meaningful names.
    (Ca, Cb, Cd, h, Cc) = x[0:5]
    F = u[0:1]
    F0 = p[0:1]

    # Write the ODEs.
    dCabydt = F0*Ca0 - k1*Ca - F*Ca
    dCbbydt = k1*Ca - k2*Cb -2*k3*(Cb**2) - F*Cb
    dCcbydt = k2*Cb - F*Cc
    dCdbydt = k3*(Cb**2) - F*Cd
    dhbydt = (F0-F)/A 

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCdbydt, dhbydt, dCcbydt])

def _threereac_qssa_ode(x, u, p, parameters):

    # Extract the parameters.
    k1 = parameters['k1']
    beta = parameters['beta']
    Ca0 = parameters['Ca0']
    A = parameters['A']

    # Extract the plant states into meaningful names.
    (Ca, Cd, h, Cc) = x[0:4]
    F = u[0:1]
    F0 = p[0:1]

    # Define a constant.
    sqrtoneplusbetaCa = np.sqrt(1 + beta*Ca)

    # Write the ODEs.
    dCabydt = F0*Ca0 - k1*Ca - F*Ca
    dCcbydt = 2*k1*Ca/(1 + sqrtoneplusbetaCa) - F*Cc
    dCdbydt = 0.5*k1*Ca*(-1+sqrtoneplusbetaCa)/(1+sqrtoneplusbetaCa) - F*Cd
    dhbydt = (F0-F)/A 

    # Return the derivative.
    return np.array([dCabydt, dCdbydt, dhbydt, dCcbydt])

def _cstrs_measurement(x, parameters):
    """ The measurement function."""
    # Return the measurements.
    return x[-1:]

def _get_cstrs_parameters():
    """ Get the parameter values for the 
        CSTRs in a series with a Flash example.
        The sample time is in seconds."""

    sample_time = 10. 
    Nx = 5
    Nu = 1
    Np = 1 
    Ny = 1
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 3.5 
    parameters['k2'] = 1.1
    parameters['k3'] = 0.5
    parameters['beta'] = 50. 
    parameters['Ca0'] = 3.
    parameters['A'] = 0.3 

    # Store the dimensions.
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample Time.
    parameters['sample_time'] = sample_time

    # Get the steady states.
    parameters['xs'] = np.array([178.56, 1, 0, 313, 
                                 190.07, 1, 0, 313, 
                                 5.17, 1, 0, 313])
    parameters['us'] = np.array([2., 0., 1., 
                                 0., 30., 0.])
    parameters['ps'] = np.array([0.8, 0.1, 0.8, 0.1, 313])

    # Get the constraints.
    ulb = np.array([-0.5, -500., -0.5, -500., -0.5, -500.])
    uub = np.array([0.5, 500., 0.5, 500., 0.5, 500.])
    parameters['lb'] = dict(u=ulb)
    parameters['ub'] = dict(u=uub)

    # Measurement Noise.
    parameters['Rv'] = 1e-20*np.diag(np.array([1e-4, 1e-6, 1e-6, 1e-4, 
                                               1e-4, 1e-6, 1e-6, 1e-4, 
                                               1e-4, 1e-6, 1e-6, 1e-4]))

    # Return the parameters dict.
    return parameters

def _generate_id_data():



    return 

def _compute_grey_box_predictions():


    return

