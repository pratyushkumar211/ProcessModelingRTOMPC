""" Script to generate the necessary
    parameters and training data for the 
    CSTR and flash example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import numpy as np
import mpctools as mpc

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
    k1star = parameters['k1star']
    k2star = parameters['k2star']
    Td = parameters['Td']
    Qb = parameters['Qb']
    Qr = parameters['Qr']

    # Extract the plant states into meaningful names.
    Hr, CAr, CBr, CCr, Tr = x[0], x[1], x[2], x[3], x[4]
    Hb, CAb, CBb, CCb, Tb = x[5], x[6], x[7], x[8], x[9]
    F, D = u[0], u[1]
    CAf, Tf = p[0], p[1]

    # The flash vapor phase mass fractions.
    den = alphaA*CAb + alphaB*CBb + alphaC*CCb
    CAd = alphaA*CAb/den
    CBd = alphaB*CBb/den
    CCd = alphaC*CCb/den

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fb = kb*np.sqrt(Hb)

    # The rate constants.
    k1 = k1star*np.exp(-E1byR/Tr)
    k2 = k2star*np.exp(-E2byR/Tr)

    # The rate of reactions.
    r1 = k1*CAr
    r2 = k2*(CBr**3)

    # Write the CSTR odes.
    dHrbydt = (F + D - Fr)/Ar
    dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
    dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1 - 3*r2
    dCCrbydt = (-F*CCr + D*(CCd - CCr))/(Ar*Hr) + r2
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
    return mpc.vcat([dHrbydt, dCArbydt, dCBrbydt, dCCrbydt, dTrbydt,
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
    parameters['kb'] = 3. # m^2
    parameters['delH1'] = 80. # kJ/mol
    parameters['delH2'] = 90. # kJ/mol
    parameters['E1byR'] = 200. # K
    parameters['E2byR'] = 300. # K
    parameters['k1star'] = 2. # 1/min
    parameters['k2star'] = 0.2 # 1/min
    parameters['Td'] = 310 # K
    parameters['Qb'] = 200 # kJ/min
    parameters['Qr'] = 2000 # kJ/min

    # Store the dimensions. 
    Nx, Nu, Np, Ny = 10, 2, 2, 10
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
    parameters['yindices'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    parameters['Rv'] = np.diag([0.8, 1e-3, 1e-3, 1e-3, 1., 
                                0.8, 1e-3, 1e-3, 1e-3, 1.])
    
    # Return the parameters dict.
    return parameters

def get_hyb_greybox_pars(*, plant_pars):
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
    parameters['kb'] = 3. # m^2
    parameters['delH1'] = 80. # kJ/mol
    parameters['delH2'] = 90. # kJ/mol
    # parameters['E1byR'] = 200. # K
    # parameters['E2byR'] = 300. # K
    # parameters['k1star'] = 2. # 1/min
    # parameters['k2star'] = 0.2 # 1/min
    parameters['Td'] = 310 # K
    parameters['Qb'] = 200 # kJ/min
    parameters['Qr'] = 2000 # kJ/min

    # Store the dimensions.
    parameters['Nx'] = plant_pars['Nx']
    parameters['Nu'] = plant_pars['Nu']
    parameters['Ny'] = plant_pars['Ny']
    parameters['Np'] = plant_pars['Np']

    # Sample time.
    parameters['Delta'] = 1. # min

    # Get the steady states.
    #gb_indices = [0, 1, 2, 4, 5, 6, 7, 9]
    #parameters['xs'] = plant_pars['xs'][gb_indices]
    #parameters['us'] = plant_pars['us']
    parameters['ps'] = np.array([6., 320.])

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
    Hb, CBb, Tb = y[[5, 7, 9]]
    Fb = kb*np.sqrt(Hb)
    
    # Compute and return cost.
    return ca*F*CAf + ce*D*pho*Cp*(Tb-Td) - cb*Fb*CBb

def getEconDistPars(seed=2):

    """ Function to get economic and disturbance parameters 
        for the closed-loop simulations. """

    # Set the random number seed.
    np.random.seed(seed) 

    # Number of simulation steps. 
    Nsim = 24*60

    # Frequency at which to change the parameters.
    NParChange = 4*60

    # Economic parameters.
    elb = np.array([20, 2000, 12000])
    eub = np.array([20, 4000, 16000]) 
    econPars = (eub-elb)*np.random.rand(Nsim//NParChange, 3) + elb
    econPars = np.repeat(econPars, NParChange, axis=0)

    # Disturbance parameters.
    dlb = np.array([6, 300])
    dub = np.array([6, 320])
    distPars = (dub-dlb)*np.random.rand(Nsim//NParChange, 2) + dlb
    distPars = np.repeat(distPars, NParChange, axis=0)

    # Economic and disturbance parameters.
    return econPars, distPars