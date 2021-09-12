import numpy as np
import mpctools as mpc

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 3D system. """

    # Extract the parameters.
    k1 = parameters['k1']
    k2f = parameters['k2f']
    k2b = parameters['k2b']
    V = parameters['V']

    # Extract the plant states into meaningful names.
    Ca, Cb, Cc = x[0], x[1], x[2]
    Caf = u[0]
    F = p[0]

    # Rate laws.
    r1 = k1*Ca
    r2 = k2f*(Cb**3) - k2b*Cc
    
    # Write the ODEs.
    dCabydt = F*(Caf - Ca)/V - r1
    dCbbydt = -F*Cb/V + r1 - 3*r2
    dCcbydt = -F*Cc/V + r2

    # Return.
    return mpc.vcat([dCabydt, dCbbydt, dCcbydt])

def get_plant_pars():
    """ Plant model parameters. """
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 2e-1 # m^3/min
    parameters['k2f'] = 5e-1 # m^3/min
    parameters['k2b'] = 1e-1 # m^3/min
    parameters['V'] = 15 # m^3

    # Store the dimensions.
    parameters['Nx'] = 3
    parameters['Nu'] = 1
    parameters['Ny'] = 2
    parameters['Np'] = 1

    # Sample time.
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0.5, 0.5]) # to be rectified.
    parameters['us'] = np.array([2.0]) # mol/m^3
    parameters['ps'] = np.array([0.8]) # m^3/min

    # Get the constraints.
    ulb = np.array([1.0])
    uub = np.array([3.0])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Measurement indices and noise.
    parameters['yindices'] = [0, 1]
    parameters['Rv'] = 0*np.diag([1e-3, 1e-4])

    # Return the parameters dict.
    return parameters

def get_hyb_pars(*, plant_pars, Nx):
    """ GreyBox parameters for the partial hybrid model. """
    
    # Parameters.
    parameters = {}
    parameters['V'] = plant_pars['V'] # (volume of the reactor).

    # Model dimensions.
    parameters['Nx'] = Nx
    parameters['Nu'] = plant_pars['Nu']
    parameters['Ny'] = plant_pars['Ny']
    parameters['Np'] = plant_pars['Np']

    # Sample time.
    parameters['Delta'] = plant_pars['Delta']

    # Get the steady states.
    parameters['xs'] = plant_pars['xs'][:Nx] # to be rectified.
    parameters['us'] = plant_pars['us']
    parameters['ps'] = plant_pars['ps']

    # Get the constraints.
    parameters['ulb'] = plant_pars['ulb']
    parameters['uub'] = plant_pars['uub']

    # Return the parameters dict.
    return parameters

def cost_yup(y, u, p):
    """ Economic stage cost. """

    # Get inputs, cost parameters, and measurements.
    CAf = u[0:1]
    costPar_A, costPar_B = p[0:2]
    _, CB = y[0:2]

    # Return.
    return costPar_A*CAf - costPar_B*CB

# def getEconDistPars(seed=2):
#     """ Get the economic and measured disturbance parameters. """

#     # Set random number seed.
#     np.random.seed(seed)

#     # Number of simulation steps.
#     Nsim = 6*24*60 # 6 days.

#     # Economic cost parameters.
#     NParChange = 4*60

#     # Gather two sets of parameters.
#     # In the more constrained region.
#     plb = np.array([100., 200.])
#     pub = np.array([100., 400.])
#     econPars1 = (pub - plb)*np.random.rand(Nsim//NParChange//2, 2) + plb
#     plb = np.array([100., 100.])
#     pub = np.array([100., 600.])
#     econPars2 = (pub - plb)*np.random.rand(Nsim//NParChange//2, 2) + plb
#     econPars = np.concatenate((econPars1, econPars2), axis=0)
#     np.random.shuffle(econPars)

#     # Now repeat.
#     econPars = np.repeat(econPars, NParChange, axis=0)

#     # Measured disturbance parameters.
#     ps = get_plant_pars()['ps'][:, np.newaxis]
#     distPars = np.tile(ps.T, (Nsim, 1))

#     # Return. 
#     return econPars, distPars