import numpy as np
import mpctools as mpc

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 3D system. """

    # Extract the parameters.
    k1 = parameters['k1']
    k2f = parameters['k2f']
    k2b = parameters['k2b']
    V = parameters['V']

    # Extract states.
    cI, cM, cS, T = x[0], x[1], x[2], x[3]
    Tc, lam0, lam1, lam2 = x[4], x[5], x[6], x[7]

    # Extract controls.
    QfI, QfM, QfS, Qc = u[0], u[1], u[2], u[3]

    # Extract disturbances.
    cIf, cMf, cSf, Tf = p[0], p[1], p[2], p[3]
    
    # Balances for concentrations.
    dcIbydt = (QfI*cIf - Qo*cI)/V - rI
    dcMbydt = (QfM*cMf - Qo*cM)/V - rM
    dcSbydt = (QfS*cSf - Qo*cS)/V - rS

    # Temperature balances.
    dTbydt = Qo*(Tf - T)/V + delHr*kp*Cm*Cp/(pho*Cp) - UA*(T - Tc)/(pho*Cp*V)
    dTcbydt = Qo

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

    # Input constraints.
    ulb = np.array([1.0])
    uub = np.array([3.0])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Measurement indices and noise.
    parameters['yindices'] = [0, 1]
    parameters['Rv'] = np.diag([1e-3, 1e-4])

    # Return the parameters dict.
    return parameters