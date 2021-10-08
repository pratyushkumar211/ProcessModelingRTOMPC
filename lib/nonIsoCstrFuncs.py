import numpy as np
import mpctools as mpc

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 3D system. """

    # Extract the parameters.
    k1 = parameters['k1']
    k2f = parameters['k2f']
    k2b = parameters['k2b']
    A = parameters['A']

    # Extract the plant states into meaningful names.
    H, Ca, Cb, Cc, Cd, T = x[0], x[1], x[2], x[3], x[4], x[5]
    Qfa, Qfb, Qc = u[0], u[1], u[2]
    Caf, Cbf = p[0], p[1]

    # Outlet flow rate.
    Qout = ko*np.sqrt(H)

    # Rate constants.
    r1 = k1*Ca
    r2 = k2f*(Cb**3) - k2b*Cc
    
    # Height dynamics.
    dHbydt = (Qfa + Qfb - Qout)/A

    # Concentration dynamics.
    dCabydt = Qfa*(Caf - Ca)/(A*H) - Qfb*Ca/(A*H) - r1
    dCbbydt = Qfb*(Cbf - Cb)/(A*H) - Qfa*Cb/(A*H) - 4*r1
    dCcbydt = -Cc*(Qfa + Qfb)/(A*H) + r1 - 3*r2
    dCdbydt = -Cd*(Qfa + Qfb)/(A*H) + 2*r2

    # Temperature dynamics.
    dTbydt = Qfa*(Tfa - T)/(A*H) + Qfb*(Tfb - T)/(A*H)
    dTbydt += (r1*delH1 + r2*delH2)/(pho*Cp) + Qc/(pho*Cp*A*H)

    # Return.
    return mpc.vcat([dHbydt, dCabydt, dCbbydt, dCcbydt, dCdbydt, dTbydt])

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

    # Input constraints.
    ulb = np.array([1.0])
    uub = np.array([3.0])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Measurement indices and noise.
    parameters['yindices'] = [0, 1]
    parameters['Rv'] = 0*np.diag([1e-3, 1e-4])

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