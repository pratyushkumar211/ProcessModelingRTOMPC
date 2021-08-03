import numpy as np
import mpctools as mpc

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 3D system. """

    # Extract the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    V = parameters['V']

    # Extract the plant states into meaningful names.
    Ca, Cb, Cc = x[0], x[1], x[2] 
    Caf = u[0]
    F = p[0]

    # Write the ODEs.
    dCabydt = F*(Caf - Ca)/V - k1*Ca
    dCbbydt = k1*Ca - 3*k2*(Cb**3) + 3*k3*Cc - F*Cb/V
    dCcbydt = k2*(Cb**3) - k3*Cc - F*Cc/V

    # Return.
    return mpc.vcat([dCabydt, dCbbydt, dCcbydt])

def get_plant_pars():
    """ Plant model parameters. """
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 6e-2 # m^3/min.
    parameters['k2'] = 4e-1 # m^3/min.
    parameters['k3'] = 8e-1 # m^3/min.
    parameters['V'] = 10 # m^3

    # Store the dimensions.
    parameters['Nx'] = 3
    parameters['Nu'] = 1
    parameters['Ny'] = 3
    parameters['Np'] = 1

    # Sample time.
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0.5, 0.5]) # to be rectified.
    parameters['us'] = np.array([1.]) # mol/m^3
    parameters['ps'] = np.array([0.5]) # m^3/min

    # Get the constraints. 
    ulb = np.array([0.2])
    uub = np.array([1.6])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Measurement indices and noise.
    parameters['yindices'] = [0, 1, 2]
    parameters['Rv'] = np.diag([1e-4, 1e-4, 1e-4])

    # Return the parameters dict.
    return parameters

def get_hyb_greybox_pars(*, plant_pars):
    """ GreyBox parameters for the hybrid model. """
    
    # Parameters.
    parameters = {}

    # Model dimensions.
    parameters['Nx'] = 3
    parameters['Nu'] = plant_pars['Nu']
    parameters['Ny'] = plant_pars['Ny']
    parameters['Np'] = plant_pars['Np']

    # Sample time.
    parameters['Delta'] = plant_pars['Delta']

    # Get the steady states.
    gb_indices = [0, 1, 2]
    parameters['xs'] = plant_pars['xs'][gb_indices] # to be rectified.
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