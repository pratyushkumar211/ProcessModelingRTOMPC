# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [makes] pickle
""" Script to generate the necessary
    parameters and training data for the 
    CSTR and flash example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np

def plant_ode(x, u, p, parameters):
    """ ODEs describing the 10-D system. """

    # Extract the parameters.
    alphaA = parameters['alphaA']
    alphaB = parameters['alphaB']
    alphaC = parameters['alphaC']
    pho = parameters['pho']
    Cp = parameters['Cp']
    Ar = parameters['Ar']
    Ab = parameters['Ab']
    kr = parameters['kr']
    kb = parameters['kb']
    delH1 = parameters['delH1']
    delH2 = parameters['delH2']
    E1byR = parameters['E1byR']
    E2byR = parameters['E2byR']
    E3byR = parameters['E3byR']
    k1star = parameters['k1star']
    k2star = parameters['k2star']
    k3star = parameters['k3star']
    Td = parameters['Td']
    Qb = parameters['Qb']
    Qr = parameters['Qr']

    # Extract the plant states into meaningful names.
    Hr, CAr, CBr, CCr, Tr = x[0:5]
    Hb, CAb, CBb, CCb, Tb = x[5:10]
    F, D = u[0:2]
    CAf, Tf = p[0:2]

    # The flash vapor phase mass fractions.
    den = alphaA*CAb + alphaB*CBb + alphaC*CCb
    CAd = alphaA*CAb/den
    CBd = alphaB*CBb/den
    CCd = alphaB*CCb/den

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fb = kb*np.sqrt(Hb)

    # The rate constants.
    k1 = k1star*np.exp(-E1byR/Tr)
    k2 = k2star*np.exp(-E2byR/Tr)
    k3 = k3star*np.exp(-E3byR/Tr)

    # The rate of reactions.
    r1 = k1*CAr
    r2 = k2*(CBr**3)
    r3 = k3*CCr

    # Write the CSTR odes.
    dHrbydt = (F + D - Fr)/Ar
    dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
    dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1 - 3*r2 + r3
    dCCrbydt = (-F*CCr + D*(CCd - CCr))/(Ar*Hr) + r2 - r3
    dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
    dTrbydt = dTrbydt + (r1*delH1 + r2*delH2)/(pho*Cp)
    dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

    # Write the flash odes.
    dHbbydt = (Fr - Fb - D)/Ab
    dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
    dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
    dCCbbydt = (Fr*(CCr - CCb) + D*(CCb - CCd))/(Ab*Hb)
    dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

    # Return the derivative.
    return np.array([dHrbydt, dCArbydt, dCBrbydt, dCCrbydt, dTrbydt,
                     dHbbydt, dCAbbydt, dCBbbydt, dCCbbydt, dTbbydt])

# def greybox_ode(x, u, p, parameters):
#     """ Simple ODE describing the grey-box plant. """

#     # Extract the parameters.
#     alphaA = parameters['alphaA']
#     alphaB = parameters['alphaB']
#     pho = parameters['pho']
#     Cp = parameters['Cp']
#     Ar = parameters['Ar']
#     Ab = parameters['Ab']
#     kr = parameters['kr']
#     kb = parameters['kb']
#     delH1 = parameters['delH1']
#     E1byR = parameters['E1byR']
#     k1star = parameters['k1star']
#     Td = parameters['Td']
#     Qb = parameters['Qb']
#     Qr = parameters['Qr']

#     # Extract the plant states into meaningful names.
#     (Hr, CAr, CBr, Tr) = x[0:4]
#     (Hb, CAb, CBb, Tb) = x[4:8]
#     (F, D) = u[0:2]
#     (CAf, Tf) = p[0:2]

#     # The flash vapor phase mass fractions.
#     den = alphaA*CAb + alphaB*CBb
#     CAd = alphaA*CAb/den
#     CBd = alphaB*CBb/den

#     # The outlet mass flow rates.
#     Fr = kr*np.sqrt(Hr)
#     Fb = kb*np.sqrt(Hb)

#     # Rate constant and reaction rate.
#     k1 = k1star*np.exp(-E1byR/Tr)
#     r1 = k1*CAr

#     # Write the CSTR odes.
#     dHrbydt = (F + D - Fr)/Ar
#     dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
#     dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
#     dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
#     dTrbydt = dTrbydt + (r1*delH1)/(pho*Cp)
#     dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

#     # Write the flash odes.
#     dHbbydt = (Fr - Fb - D)/Ab
#     dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
#     dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
#     dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)
    
#     # Return the derivative.
#     return np.array([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
#                      dHbbydt, dCAbbydt, dCBbbydt, dTbbydt])

def get_plant_pars():
    """ Get the parameter values for the
        CSTRs with flash example. """

    # Parameters.
    parameters = {}
    parameters['alphaA'] = 8.
    parameters['alphaB'] = 1.
    parameters['alphaC'] = 1.
    parameters['pho'] = 6. # Kg/m^3
    parameters['Cp'] = 3. # KJ/(Kg-K)
    parameters['Ar'] = 3. # m^2
    parameters['Ab'] = 3. # m^2
    parameters['kr'] = 4. # m^2
    parameters['kb'] = 2. # m^2
    parameters['delH1'] = 100. # kJ/mol
    parameters['delH2'] = 120. # kJ/mol
    parameters['delH3'] = 50.
    parameters['E1byR'] = 200. # K
    parameters['E2byR'] = 300. # K
    parameters['E3byR'] = 500. # K
    parameters['k1star'] = 0.3 # 1/min
    parameters['k2star'] = 0.5 # 1/min
    parameters['k3star'] = 0. # 1/min
    parameters['Td'] = 310 # K
    parameters['Qb'] = 200 # kJ/min
    parameters['Qr'] = 200 # kJ/min

    # Store the dimensions. 
    Nx, Nu, Np, Ny = 10, 2, 2, 8
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Np'] = Np
    parameters['Ny'] = Ny

    # Sample time.
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([50., 1., 0., 0., 313.,
                                 50., 1., 0., 0., 313.])
    parameters['us'] = np.array([10., 5.])
    parameters['ps'] = np.array([6., 320.])

    # Get the constraints.
    parameters['ulb'] = np.array([5., 2.])
    parameters['uub'] = np.array([15., 8.])

    # The C matrix for the plant.
    parameters['yindices'] = [0, 1, 2, 4, 5, 6, 7, 9]
    parameters['Rv'] = 0*np.diag([0.8, 1e-3, 1e-3, 1., 
                                  0.8, 1e-3, 1e-3, 1.])
    
    # Return the parameters dict.
    return parameters

def get_greybox_pars(*, plant_pars):
    """ Get the parameter values for the
        CSTRs with flash example. """

    # Parameters.
    parameters = {}
    #parameters['alphaA'] = 8.
    #parameters['alphaB'] = 1.
    #parameters['alphaC'] = 1.
    parameters['pho'] = 6. # Kg/m^3
    parameters['Cp'] = 3. # KJ/(Kg-K)
    parameters['Ar'] = 3. # m^2
    parameters['Ab'] = 3. # m^2
    parameters['kr'] = 4. # m^2
    parameters['kb'] = 2. # m^2
    #parameters['delH1'] = 150. # kJ/mol
    #parameters['E1byR'] = 200 # K
    #parameters['k1star'] = 0.3 # 1/min
    parameters['Td'] = 310 # K
    parameters['Qb'] = 200 # kJ/min
    parameters['Qr'] = 200 # kJ/min

    # Store the dimensions.
    parameters['Nx'] = 8
    parameters['Nu'] = plant_pars['Nu']
    parameters['Ny'] = plant_pars['Ny']
    parameters['Np'] = plant_pars['Np']

    # Sample time.
    parameters['Delta'] = 1. # min

    # Get the steady states.
    #gb_indices = [0, 1, 2, 4, 5, 6, 7, 9]
    #parameters['xs'] = plant_pars['xs'][gb_indices]
    #parameters['us'] = plant_pars['us']
    parameters['ps'] = np.array([5., 320.])

    # The C matrix for the grey-box model.
    #parameters['yindices'] = [0, 1, 2, 3, 4, 5, 6, 7]

    # Get the constraints.
    parameters['ulb'] = plant_pars['ulb']
    parameters['uub'] = plant_pars['uub']
    
    # Return the parameters dict.
    return parameters

def cost_yup(y, u, p, pars):
    """ Custom stage cost for the CSTR/Flash system. """
    CAf = pars['ps'][0]
    Td = pars['Td']
    pho = pars['pho']
    Cp = pars['Cp']
    kb = pars['kb']
    
    # Get inputs, parameters, and states.
    F, D = u[0:2]
    ce, ca, cb = p[0:3]
    Hb, CBb, Tb = y[[4, 6, 7]]
    Fb = kb*np.sqrt(Hb)
    
    # Compute and return cost.
    return ca*F*CAf + ce*D*pho*Cp*(Tb-Td) - cb*Fb*CBb

# def get_energy_price(*, num_days, sample_time):
#     """ Get a two day heat disturbance profile. """
#     energy_price = np.zeros((24, 1))
#     energy_price[0:8, :] = 10*np.ones((8, 1))
#     energy_price[8:16, :] = 70*np.ones((8, 1))
#     energy_price[16:24, :] = 10*np.ones((8, 1))
#     energy_price = 1e-2*np.tile(energy_price, (num_days, 1))
#     return _resample_fast(x=energy_price,
#                           xDelta=60,
#                           newDelta=sample_time,
#                           resample_type='zoh')

# def get_economic_opt_pars(*, num_days, sample_time, plant_pars):
#     """ Get the parameters for Empc and RTO simulations. """

#     # Get the cost parameters.
#     energy_price = get_energy_price(num_days=num_days, sample_time=sample_time)
#     raw_mat_price = _resample_fast(x = np.array([[1000.], [1000.], 
#                                                  [1000.], [1000.], 
#                                                  [1000.], [1000.], 
#                                                  [1000.], [1000.]]), 
#                                    xDelta=6*60,
#                                    newDelta=sample_time,
#                                    resample_type='zoh')
#     product_price = _resample_fast(x = np.array([[8000.], [7000.], 
#                                                  [5000.], [4000.], 
#                                                  [4000.], [4000.], 
#                                                  [4000.], [4000.]]),
#                                    xDelta=6*60,
#                                    newDelta=sample_time,
#                                    resample_type='zoh')
#     cost_pars = np.concatenate((energy_price,
#                                 raw_mat_price, product_price), axis=1)
    
#     # Get the plant disturbances.
#     ps = plant_pars['ps'][np.newaxis, :]
#     disturbances = np.repeat(ps, num_days*24*60, axis=0)

#     # Return as a concatenated vector.
#     return cost_pars, disturbances