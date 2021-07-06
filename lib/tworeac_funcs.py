""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import numpy as np
import mpctools as mpc

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 3D system. """
    # Extract the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    
    # Extract the plant states into meaningful names.
    Ca, Cb, Cc = x[0], x[1], x[2] 
    Caf = u[0]
    tau = p[0]

    # Write the ODEs.
    dCabydt = (Caf-Ca)/tau - k1*Ca
    dCbbydt = k1*Ca - 3*k2*(Cb**3) + 3*k3*Cc - Cb/tau
    dCcbydt = k2*(Cb**3) - k3*Cc - Cc/tau

    # Return the derivative.
    return mpc.vcat([dCabydt, dCbbydt, dCcbydt])

# def greybox_ode(x, u, p, parameters):
#     """ Simple ODE describing the grey-box plant. """

#     # Extract the parameters.
#     k1 = parameters['k1']

#     # Extract the plant states into meaningful names.
#     (Ca, Cb) = x[0:2]
#     Caf = u[0]
#     tau = p[0]

#     # Write the ODEs.
#     dCabydt = (Caf-Ca)/tau - k1*Ca
#     dCbbydt = k1*Ca - Cb/tau

#     # Return the derivative.
#     return np.array([dCabydt, dCbbydt])

def get_plant_pars():
    """ Get the parameter values for the 
        three reaction example. """
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 1. # m^3/min.
    parameters['k2'] = 0.01 # m^3/min.
    parameters['k3'] = 0.05 # m^3/min.

    # Store the dimensions.
    parameters['Nx'] = 3
    parameters['Nu'] = 1
    parameters['Ny'] = 3
    parameters['Np'] = 1

    # Sample time.
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0.5, 0.5]) # to be updated.
    parameters['us'] = np.array([1.5]) # Ca0s
    parameters['ps'] = np.array([30.]) # min.

    # Get the constraints. 
    ulb = np.array([0.5])
    uub = np.array([2.5])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Measurement indices and noise.
    parameters['yindices'] = [0, 1, 2]
    parameters['Rv'] = np.diag([1e-4, 1e-3, 1e-4])

    # Return the parameters dict.
    return parameters

def get_hyb_greybox_pars(*, plant_pars):
    """ Get the parameter values for the 
        three reaction example. """
    
    # Parameters.
    parameters = {}
    #parameters['k1'] = 0.2 # m^3/min.

    # Store the dimensions.
    parameters['Nx'] = 3
    parameters['Nu'] = plant_pars['Nu']
    parameters['Ny'] = plant_pars['Ny']
    parameters['Np'] = plant_pars['Np']

    # Sample time.
    parameters['Delta'] = plant_pars['Delta'] # min.

    # Get the steady states.
    gb_indices = [0, 1, 2]
    parameters['xs'] = plant_pars['xs'][gb_indices] # to be updated.
    parameters['us'] = plant_pars['us'] # Cafs
    parameters['ps'] = plant_pars['ps'] # tau (min)

    # Get the constraints.
    parameters['ulb'] = plant_pars['ulb']
    parameters['uub'] = plant_pars['uub']

    # Return the parameters dict.
    return parameters

def cost_yup(y, u, p):
    """ Custom stage cost for the tworeac system. """
    # Get inputs, parameters, and states.
    CAf = u[0:1]
    ca, cb = p[0:2]
    CA, CB = y[0:2]
    # Compute and return cost.
    return ca*CAf - cb*CB

def getEconDistPars(seed=2):
    """ Get the economic and measured disturbance parameters. """

    # Set random number seed.
    np.random.seed(seed)

    # Number of simulation steps.
    Nsim = 48*60 # 2 days.

    # Economic cost parameters.
    NParChange = 180
    plb = np.array([100., 120.])
    pub = np.array([100., 240.])
    econPars = (pub - plb)*np.random.rand(Nsim//NParChange, 2) + plb
    econPars = np.repeat(econPars, NParChange, axis=0)

    # Measured disturbance parameters.
    ps = get_plant_pars()['ps'][:, np.newaxis]
    distPars = np.tile(ps.T, (Nsim, 1))
    

    # Return. 
    return econPars, distPars