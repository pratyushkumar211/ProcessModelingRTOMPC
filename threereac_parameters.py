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
    (Ca, Cb, Cc, Cd, h) = x[0:5]
    F = u[0:1]
    F0 = p[0:1]

    # Write the ODEs.
    dCabydt = F0*Ca0 - k1*Ca - F*Ca
    dCbbydt = k1*Ca - k2*Cb -2*k3*(Cb**2) - F*Cb
    dCcbydt = k2*Cb - F*Cc
    dCdbydt = k3*(Cb**2) - F*Cd
    dhbydt = (F0-F)/A 

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCcbydt, dCdbydt, dhbydt])

def _threereac_qssa_ode(x, u, p, parameters):

    # Extract the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    Ca0 = parameters['Ca0']
    A = parameters['A']

    # Extract the plant states into meaningful names.
    (Ca, Cc, Cd, h) = x[0:5]
    F = u[0:1]
    F0 = p[0:1]

    # Write the ODEs.
    dCabydt = F0*Ca0 - k1*Ca - F*Ca
    dCbbydt = k1*Ca - k2*Cb -2*k3*(Cb**2) - F*Cb
    dCcbydt = k2*Cb - F*Cc
    dCdbydt = k3*(Cb**2) - F*Cd
    dhbydt = (F0-F)/A 

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCcbydt, dCdbydt, dhbydt])

def _cstrs_measurement(x, parameters):
    """ The measurement function."""
    yscale = parameters['yscale']
    C = np.diag(1/yscale.squeeze()) @ parameters['C']
    # Return the measurements.
    return C.dot(x)

def _get_cstrs_parameters():
    """ Get the parameter values for the 
        CSTRs in a series with a Flash example.
        The sample time is in seconds."""

    sample_time = 10. 
    Nx = 12
    Nu = 6
    Np = 5 
    Ny = 12
    
    # Parameters.
    parameters = {}
    parameters['alphaA'] = 3.5 
    parameters['alphaB'] = 1.1
    parameters['alphaC'] = 0.5
    parameters['pho'] = 50. # Kg/m^3 # edited.
    parameters['Cp'] = 3. # KJ/(Kg-K) # edited.
    parameters['Ar'] = 0.3 # m^2 
    parameters['Am'] = 2. # m^2 
    parameters['Ab'] = 4. # m^2 
    parameters['kr'] = 2.5 # m^2
    parameters['km'] = 2.5 # m^2
    parameters['kb'] = 1.5 # m^2
    parameters['delH1'] = -40 # kJ/Kg 
    parameters['delH2'] = -50 # kJ/Kg
    parameters['EbyR'] = 150 # K 
    parameters['k1star'] = 4e-4 # 1/sec #edited.
    parameters['k2star'] = 1.8e-6 # 1/sec #edited.
    parameters['Td'] = 313 # K # edited.

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
    ylb = np.array([-5., 0., 0., -10., 
                    -5., 0., 0., -3., 
                    -1., 0., 0., -10.])
    yub = np.array([5., 1., 1., 10., 
                    5., 1., 1., 3., 
                    1., 1, 1., 10.])
    plb = np.array([-0.1, -0.1, -0.1, -0.1, -8.])
    pub = np.array([0.05, 0.05, 0.05, 0.05, 8.])
    parameters['lb'] = dict(u=ulb, y=ylb, p=plb)
    parameters['ub'] = dict(u=uub, y=yub, p=pub)

    # Get the scaling.
    parameters['uscale'] = 0.5*(parameters['ub']['u'] - parameters['lb']['u'])
    parameters['pscale'] = 0.5*(parameters['ub']['p'] - parameters['lb']['p'])
    parameters['yscale'] = 0.5*(parameters['ub']['y'] - parameters['lb']['y'])

    # Scale the lower and upper bounds for the MPC controller.
    # Scale the bounds.
    parameters['lb']['u'] = parameters['lb']['u']/parameters['uscale']
    parameters['ub']['u'] = parameters['ub']['u']/parameters['uscale']
    parameters['lb']['y'] = parameters['lb']['y']/parameters['yscale']
    parameters['ub']['y'] = parameters['ub']['y']/parameters['yscale']
    parameters['lb']['p'] = parameters['lb']['p']/parameters['pscale']
    parameters['ub']['p'] = parameters['ub']['p']/parameters['pscale']

    # The C matrix for the plant.
    parameters['C'] = np.eye(Nx)

    # The H matrix.
    parameters['H'] = np.zeros((6, Ny))
    parameters['H'][0, 0] = 1.
    parameters['H'][1, 3] = 1.
    parameters['H'][2, 4] = 1.
    parameters['H'][3, 7] = 1.
    parameters['H'][4, 8] = 1.
    parameters['H'][5, 11] = 1.

    # Measurement Noise.
    parameters['Rv'] = 1e-20*np.diag(np.array([1e-4, 1e-6, 1e-6, 1e-4, 
                                               1e-4, 1e-6, 1e-6, 1e-4, 
                                               1e-4, 1e-6, 1e-6, 1e-4]))

    # Return the parameters dict.
    return parameters