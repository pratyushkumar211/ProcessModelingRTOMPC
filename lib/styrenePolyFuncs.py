import numpy as np
import mpctools as mpc

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing the styrene polymerization plant. """

    # Extract model parameters.
    # Rate constant parameters.
    kd0 = parameters['kd0']
    kp0 = parameters['kp0']
    kfs0 = parameters['kfs0']
    kf0 = parameters['kf0']
    ktc0 = parameters['ktc0']
    ktd0 = parameters['ktd0']
    # Activation energies corresponding to the parameters.
    Ed = parameters['Ed']
    Ep = parameters['Ep']
    Efs = parameters['Efs']
    Ef = parameters['Ef']
    Etc = parameters['Etc']
    Etd = parameters['Etd']
    R = parameters['R']

    # Density, volume, heat capacities, and heat of reaction. 
    f = parameters['f']
    UA = parameters['UA']
    delHr = parameters['delHr']
    phoCp = parameters['phoCp']
    V = parameters['V']
    phocCpc = parameters['phocCpc']
    Vc = parameters['Vc']

    # Feed concentrations. 
    cIf = parameters['cIf']
    cMf = parameters['cMf']
    cSf = parameters['cSf']

    # Extract states.
    cI, cM, cS, T = x[0], x[1], x[2], x[3]
    Tc, lam0, lam1, lam2 = x[4], x[5], x[6], x[7]

    # Extract controls.
    QI, QM, QS, Qc = u[0], u[1], u[2], u[3]

    # Outflow volumetric flow rate. 
    Qo = QI + QM + QS

    # Extract disturbances.
    Tf, Tcf = p[0], p[1]

    # Get the rate constants using Arhenius law.
    kd = kd0*np.exp(-Ed/(R*T))
    kp = kp0*np.exp(-Ep/(R*T))
    kfs = kfs0*np.exp(-Efs/(R*T))
    kf = kf0*np.exp(-Ef/(R*T))
    ktc = ktc0*np.exp(-Etc/(R*T))
    ktd = ktd0*np.exp(-Etd/(R*T))

    # Mean polymer concentration.
    cP = np.sqrt(2*f*kd*cI/(ktd + ktc))

    # Probability of propagation.
    alpha = kp*cM/(kp*cM + kfs*cS + kf*cM + (ktc + ktd)*cP)

    # Reaction rates. 
    rI = kd*cI
    rM = 2*f*kd*cI + (kp + kf)*cM*cP
    rS = kfs*cP*cS

    # Balances for concentrations.
    dcIbydt = (QI*cIf - Qo*cI)/V - rI
    dcMbydt = (QM*cMf - Qo*cM)/V - rM
    dcSbydt = (QS*cSf - Qo*cS)/V - rS

    # Temperature balances.
    dTbydt = Qo*(Tf - T)/V + delHr*rM/phoCp - UA*(T - Tc)/(phoCp*V)
    dTcbydt = Qc*(Tcf - Tc)/Vc + UA*(T - Tc)/(phocCpc*Vc)

    # Moment balances.
    # Zeroth moment.
    dlam0bydt = (kfs*kf*cS*cM + ktd*cP)*alpha*cP + (ktc*(cP**2)/2) - (Qo*lam0/V)

    # First moment.
    dlam1bydt = (kfs*cS + kf*cM + ktd*cP)*(2*alpha - alpha**2) + ktc*cP
    dlam1bydt = (cP/(1 - alpha))*dlam1bydt - (Qo*lam1/V)

    # Second moment.
    dlam2bydt = (kfs*cS + kf*cM + ktd*cP)*(4*alpha - 3*alpha**2 + alpha**3)
    dlam2bydt += ktc*cP*(2 + alpha)
    dlam2bydt = (cP/(1 - alpha)**2)*dlam2bydt - (Qo*lam2/V)

    # Return.
    return mpc.vcat([dcIbydt, dcMbydt, dcSbydt, dTbydt, 
                     dTcbydt, dlam0bydt, dlam1bydt, dlam2bydt])

def get_plant_pars():
    """ Plant model parameters. """
    
    # Parameter dictionary.
    parameters = {}

    # Rate constant parameters.
    parameters['kd0'] = 5.95e+13 # 1/sec
    parameters['kp0'] = 1.06e+4 # m^3/(mol-sec)
    parameters['kfs0'] = 91.45 # m^3/(mol-sec)
    parameters['kf0'] = 53.020 # m^3/(mol-sec)
    parameters['ktc0'] = 6.25e+5 # m^3/(mol-sec)
    parameters['ktd0'] = 6.25e+5 # m^3/(mol-sec)

    # Activation energies for the rate constants.
    parameters['Ed'] = 1.23e+5 # J/mol
    parameters['Ep'] = 2.95e+4 # J/mol
    parameters['Efs'] = 9.14e+4 # J/mol
    parameters['Ef'] = 5.30e+4 # J/mol
    parameters['Etc'] = 7.01e+3 # J/mol
    parameters['Etd'] = 7.01e+3 # J/mol
    parameters['R'] = 8.314 # J/(K-mol)

    # Feed concentrations. 
    parameters['cIf'] = 5.88e+2 # mol/m^3
    parameters['cMf'] = 8.69e+3 # mol/m^3
    parameters['cSf'] = 4e+5 # mol/m^3

    # Density, volume, heat capacities, and heat of reaction. 
    parameters['f'] = 0.6
    parameters['UA'] = 293.076 # W/K
    parameters['delHr'] = 9e+2 # J/mol
    parameters['phoCp'] = 1e+5 # J/(m^3-K)
    parameters['V'] = 25 # m^3
    parameters['phocCpc'] = 4.04e+6 # J/(m^3-K)
    parameters['Vc'] = 30 # m^3

    # Store the dimensions.
    parameters['Nx'] = 8
    parameters['Nu'] = 4
    parameters['Ny'] = 6
    parameters['Np'] = 2

    # Sample time.
    parameters['Delta'] = 1. # sec

    # Get the steady states.
    parameters['xs'] = np.array([2.82e+02, 4.02e+03, 4.83e+03, 3.37e+02,
                                 3.02e+02, 3.50e+00, 1.79e+02, 4.04e+04]) 
    #parameters['xs'] = np.array([1e-2,3,3,320,
    #                             305,1e+2,1e+2,1e+2]) 
    parameters['us'] = np.array([8e-3, 8e-3, 2e-4, 2e-3]) # m^3/sec
    parameters['ps'] = np.array([330, 295]) # K

    # Input constraints.
    ulb = np.array([1e-4, 1e-4, 1e-5, 5e-4])
    uub = np.array([1e-1, 1e-1, 1e-2, 1e-2])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Measurement indices and noise.
    parameters['yindices'] = [1, 3, 4, 5, 6, 7]
    parameters['Rv'] = 0*np.diag([1e-2, 4, 4, 1e-2, 1e-2])

    # Return the parameters dict.
    return parameters