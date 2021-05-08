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
    k1star = parameters['k1star']
    k2star = parameters['k2star']
    Td = parameters['Td']
    Qb = parameters['Qb']
    Qr = parameters['Qr']

    # Extract the plant states into meaningful names.
    (Hr, CAr, CBr, CCr, Tr) = x[0:5]
    (Hb, CAb, CBb, CCb, Tb) = x[5:10]
    (F, D) = u[0:2]
    (CAf, Tf) = p[0:2]

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
    return np.array([dHrbydt, dCArbydt, dCBrbydt, dCCrbydt, dTrbydt,
                     dHbbydt, dCAbbydt, dCBbbydt, dCCbbydt, dTbbydt])

def greybox_ode(x, u, p, parameters):
    """ Simple ODE describing the grey-box plant. """

    # Extract the parameters.
    alphaA = parameters['alphaA']
    alphaB = parameters['alphaB']
    pho = parameters['pho']
    Cp = parameters['Cp']
    Ar = parameters['Ar']
    Ab = parameters['Ab']
    kr = parameters['kr']
    kb = parameters['kb']
    delH1 = parameters['delH1']
    E1byR = parameters['E1byR']
    k1star = parameters['k1star']
    Td = parameters['Td']
    Qb = parameters['Qb']
    Qr = parameters['Qr']

    # Extract the plant states into meaningful names.
    (Hr, CAr, CBr, Tr) = x[0:4]
    (Hb, CAb, CBb, Tb) = x[4:8]
    (F, D) = u[0:2]
    (CAf, Tf) = p[0:2]

    # The flash vapor phase mass fractions.
    den = alphaA*CAb + alphaB*CBb
    CAd = alphaA*CAb/den
    CBd = alphaB*CBb/den

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fb = kb*np.sqrt(Hb)

    # Rate constant and reaction rate.
    k1 = k1star*np.exp(-E1byR/Tr)
    r1 = k1*CAr

    # Write the CSTR odes.
    dHrbydt = (F + D - Fr)/Ar
    dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
    dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
    dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
    dTrbydt = dTrbydt + (r1*delH1)/(pho*Cp)
    dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

    # Write the flash odes.
    dHbbydt = (Fr - Fb - D)/Ab
    dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
    dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
    dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)
    
    # Return the derivative.
    return np.array([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
                     dHbbydt, dCAbbydt, dCBbbydt, dTbbydt])

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
    parameters['E1byR'] = 200. # K
    parameters['E2byR'] = 300. # K
    parameters['k1star'] = 0.3 # 1/min
    parameters['k2star'] = 0.5 # 1/min
    parameters['Td'] = 310 # K
    parameters['Qb'] = 200 # kJ/min
    parameters['Qr'] = 200 # kJ/min

    # Store the dimensions. 
    Nx, Nu, Np, Ny = 10, 2, 2, 6
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

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
    parameters['yindices'] = [0, 2, 4, 5, 7, 9]
    parameters['Rv'] = 0*np.diag([0.8, 1e-3, 1., 
                                  0.8, 1e-3, 1.])
    
    # Return the parameters dict.
    return parameters

def get_greybox_pars(*, plant_pars):
    """ Get the parameter values for the
        CSTRs with flash example. """

    # Parameters.
    parameters = {}
    parameters['alphaA'] = 6.
    parameters['alphaB'] = 1.
    parameters['pho'] = 6. # Kg/m^3
    parameters['Cp'] = 6. # KJ/(Kg-K)
    parameters['Ar'] = 2. # m^2
    parameters['Ab'] = 2. # m^2
    parameters['kr'] = 2. # m^2
    parameters['kb'] = 1.5 # m^2
    parameters['delH1'] = 150. # kJ/mol
    parameters['E1byR'] = 200 # K
    parameters['k1star'] = 0.3 # 1/min
    parameters['Td'] = 300 # K
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
    gb_indices = [0, 1, 2, 4, 5, 6, 7, 9]
    parameters['xs'] = plant_pars['xs'][gb_indices]
    parameters['us'] = plant_pars['us']
    parameters['ps'] = np.array([5., 320.])

    # The C matrix for the grey-box model.
    parameters['yindices'] = [0, 2, 3, 4, 6, 7]

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
    Hb, CBb, Tb = y[[3, 4, 5]]
    Fb = kb*np.sqrt(Hb)
    
    # Compute and return cost.
    return ca*F*CAf + ce*D*pho*Cp*(Tb-Td) - cb*Fb*CBb

# def get_hybrid_pars(*, greybox_pars, Npast, fnn_weights, xuyscales):
#     """ Get the hybrid model parameters. """

#     hybrid_pars = copy.deepcopy(greybox_pars)
#     # Update sizes.
#     Nu, Ny = greybox_pars['Nu'], greybox_pars['Ny']
#     hybrid_pars['Nx'] = greybox_pars['Ng'] + Npast*(Nu + Ny)

#     # Update steady state.
#     ys = measurement(greybox_pars['xs'], greybox_pars)
#     yspseq = np.tile(ys, (Npast, ))
#     us = greybox_pars['us']
#     uspseq = np.tile(us, (Npast, ))
#     xs = greybox_pars['xs']
#     hybrid_pars['xs'] = np.concatenate((xs, yspseq, uspseq))
    
#     # NN pars.
#     hybrid_pars['Npast'] = Npast
#     hybrid_pars['fnn_weights'] = fnn_weights

#     # Scaling.
#     hybrid_pars['xuyscales'] = xuyscales

#     # Return.
#     return hybrid_pars

# def fnn(xG, z, u, Npast, xuyscales, fnn_weights):
#     """ Compute the NN output. """
#     nn_output = np.concatenate((xG, z, u))
#     nn_output = nn_output[:, np.newaxis]
#     for i in range(0, len(fnn_weights)-2, 2):
#         (W, b) = fnn_weights[i:i+2]
#         nn_output = W.T @ nn_output + b[:, np.newaxis]
#         nn_output = 1./(1. + np.exp(-nn_output))
#     (Wf, bf) = fnn_weights[-2:]
#     nn_output = (Wf.T @ nn_output + bf[:, np.newaxis])[:, 0]
#     # Return.
#     return nn_output

# def hybrid_func(xGz, u, parameters):
#     """ The augmented continuous time model. """

#     # Extract a few parameters.
#     Ng = parameters['Ng']
#     Ny = parameters['Ny']
#     Nu = parameters['Nu']
#     ps = parameters['ps']
#     Npast = parameters['Npast']
#     Delta = parameters['Delta']
#     fnn_weights = parameters['fnn_weights']
#     xuyscales = parameters['xuyscales']
#     xmean, xstd = xuyscales['xscale']
#     umean, ustd = xuyscales['uscale']
#     ymean, ystd = xuyscales['yscale']
#     xGzmean = np.concatenate((xmean,
#                                np.tile(ymean, (Npast, )), 
#                                np.tile(umean, (Npast, ))))
#     xGzstd = np.concatenate((xstd,
#                              np.tile(ystd, (Npast, )), 
#                              np.tile(ustd, (Npast, ))))

#     # Get some vectors.
#     xGz = (xGz - xGzmean)/xGzstd
#     u = (u-umean)/ustd
#     xG, ypseq, upseq = xGz[:Ng], xGz[Ng:Ng+Npast*Ny], xGz[-Npast*Nu:]
#     z = xGz[Ng:]
#     hxG = measurement(xG, parameters)
    
#     # Get k1.
#     k1 = greybox_ode(xG*xstd + xmean, u*ustd + umean, ps, parameters)/xstd
#     k1 +=  fnn(xG, z, u, Npast, xuyscales, fnn_weights)

#     # Interpolate for k2 and k3.
#     ypseq_interp = interpolate_yseq(np.concatenate((ypseq, hxG)), Npast, Ny)
#     z = np.concatenate((ypseq_interp, upseq))
    
#     # Get k2.
#     k2 = greybox_ode((xG + Delta*(k1/2))*xstd + xmean, u*ustd + umean, 
#                        ps, parameters)/xstd
#     k2 += fnn(xG + Delta*(k1/2), z, u, Npast, xuyscales, fnn_weights)

#     # Get k3.
#     k3 = greybox_ode((xG + Delta*(k2/2))*xstd + xmean, u*ustd + umean, 
#                        ps, parameters)/xstd
#     k3 += fnn(xG + Delta*(k2/2), z, u, Npast, xuyscales, fnn_weights)

#     # Get k4.
#     ypseq_shifted = np.concatenate((ypseq[Ny:], hxG))
#     z = np.concatenate((ypseq_shifted, upseq))
#     k4 = greybox_ode((xG + Delta*k3)*xstd + xmean, u*ustd + umean, 
#                        ps, parameters)/xstd
#     k4 += fnn(xG + Delta*k3, z, u, Npast, xuyscales, fnn_weights)
    
#     # Get the current output/state and the next time step.
#     xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
#     zplus = np.concatenate((ypseq_shifted, upseq[Nu:], u))
#     xGzplus = np.concatenate((xGplus, zplus))
#     xGzplus = xGzplus*xGzstd + xGzmean

#     # Return the sum.
#     return xGzplus

# def sim_hybrid(hybrid_func, uval, hybrid_pars, greybox_processed_data):
#     """ Hybrid validation simulation to make 
#         sure the above programmed function is 
#         the same is what tensorflow is training. """
    
#     # Get initial state.
#     t = hybrid_pars['tsteps_steady']
#     Np = hybrid_pars['Npast']
#     Ny = hybrid_pars['Ny']
#     Nu = hybrid_pars['Nu']
#     Ng = hybrid_pars['Ng']
#     y = greybox_processed_data.y
#     u = greybox_processed_data.u
#     x = greybox_processed_data.x
#     yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[:, np.newaxis]
#     up0seq = u[t-Np:t:, ].reshape(Np*Nu, )[:, np.newaxis]
#     z0 = np.concatenate((yp0seq, up0seq))
#     xG0 = x[t, :][:, np.newaxis]
#     xGz0 = np.concatenate((xG0, z0))

#     # Start the validation simulation.
#     uval = uval[t:, :]
#     Nval = uval.shape[0]
#     hx = lambda x: measurement(x, hybrid_pars)
#     fxu = lambda x, u: hybrid_func(x, u, hybrid_pars)
#     x = xGz0[:, 0]
#     yval, xGval = [], []
#     xGval.append(x)
#     for t in range(Nval):
#         yval.append(hx(x))
#         x = fxu(x, uval[t, :].T)
#         xGval.append(x)
#     yval = np.asarray(yval)
#     xGval = np.asarray(xGval)[:-1, :Ng]
#     # Return.
#     return yval, xGval

# def get_hybrid_pars_check_func(*, greybox_pars, train, 
#                                   greybox_processed_data):
#     """ Get the hybrid parameters and check the hybrid function
#         to be used for optimization. """

#     # Get NN weights and the hybrid ODE.
#     Np = train['Nps'][0] # To change.
#     fnn_weights = train['trained_weights'][0][-1] # To change.
#     xuyscales = train['xuyscales']
#     hybrid_pars = get_hybrid_pars(greybox_pars=greybox_pars,
#                                   Npast=Np,
#                                   fnn_weights=fnn_weights,
#                                   xuyscales=xuyscales)

#     # Check the hybrid function.
#     uval = greybox_processed_data[-1].u
#     ytfval = train['val_predictions'][0].y
#     xGtfval = train['val_predictions'][0].x
#     greybox_processed_data = greybox_processed_data[-1]
#     yval, xGval = sim_hybrid(hybrid_func, uval, 
#                              hybrid_pars, greybox_processed_data)

#     # Just return the hybrid parameters.
#     return hybrid_pars

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