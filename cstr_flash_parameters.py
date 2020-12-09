# [depends] %LIB%/hybridid.py
# [makes] pickle
""" Script to generate the necessary
    parameters and training data for the 
    CSTR and flash example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
from hybridid import (PickleTool, NonlinearPlantSimulator,
                      c2d, sample_prbs_like, SimData)

def _plant_ode(x, u, p, parameters):
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
    EbyR = parameters['EbyR']
    k1star = parameters['k1star']
    k2star = parameters['k2star']
    Td = parameters['Td']

    # Extract the plant states into meaningful names.
    (Hr, CAr, CBr, CCr, Tr) = x[0:5]
    (Hb, CAb, CBb, CCb, Tb) = x[5:10]
    (F, Qr, D, Qb) = u[0:4]
    (CAf, Tf) = p[0:2]

    # The flash vapor phase mass fractions.
    denominator = alphaA*CAb + alphaB*CBb + alphaC*CCb
    CAd = alphaA*CAb/denominator
    CBd = alphaB*CBb/denominator
    CCd = alphaB*CCb/denominator

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fb = kb*np.sqrt(Hb)

    # The rate constants.
    k1 = k1star*np.exp(-EbyR/Tr)
    k2 = k2star*np.exp(-EbyR/Tr)

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
    dTrbydt = dTrbydt + Qr/(pho*Ar*Cp*Hr)

    # Write the flash odes.
    dHbbydt = (Fr - Fb - D)/Ab
    dCAbbydt = (Fr*(CAr - CAb) - D*(CAd - CAb))/(Ab*Hb)
    dCBbbydt = (Fr*(CBr - CBb) - D*(CBd - CBb))/(Ab*Hb)
    dCCbbydt = (Fr*(CCr - CCb) - D*(CCd - CCb))/(Ab*Hb)
    dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

    # Return the derivative.
    return np.array([dHrbydt, dCArbydt, dCBrbydt, dCCrbydt, dTrbydt,
                     dHbbydt, dCAbbydt, dCBbbydt, dCCbbydt, dTbbydt])

def _greybox_ode(x, u, p, parameters):
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
    EbyR = parameters['EbyR']
    k1star = parameters['k1star']
    Td = parameters['Td']

#    # Extract the plant states into meaningful names.
#    (Hr, CAr, CBr, Tr) = x[0:5]
#    (Hb, CAb, CBb, Tb) = x[5:10]
#    (F, Qr, D, Qb) = u[0:4]
#    (CAf, Tf) = p[0:2]

    # The flash vapor phase mass fractions.
#    denominator = alphaA*CAb + alphaB*CBb + alphaC*CCb
#    CAd = alphaA*CAb/denominator
#    CBd = alphaB*CBb/denominator
#    CCd = alphaB*CCb/denominator

    # The outlet mass flow rates.
#    Fr = kr*np.sqrt(Hr)
#    Fb = kb*np.sqrt(Hb)
#    Fp = purgeFrac*D

    # The rate constants.
#    k1 = k1star*np.exp(-EbyR/Tr)
#    k2 = k2star*np.exp(-EbyR/Tr)

    # Write the CSTR odes.
#    dHrbydt = (F + D - Fr)/Ar
#    dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - k1*CAr
#    dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + k1*CAr - 3*k2*(CBr**3)
#    dCBrbydt = dCBrbydt + 3*k3*CCr
#    dCCrbydt = (-F*CCr + D*(CCd - CCr))/(Ar*Hr) + k2*(CBr**3) - k3*CCr
#    dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr) 
#    dTrbydt = dTrbydt + (k1*CAr*delH1 + k2*(CBr**3)*delH2)/(pho*Cp)
#    dTrbydt = dTrbydt + Qr/(pho*Ar*Cp*Hr)

    # Write the flash odes.
#    dHbbydt = (Fr - Fb - D - Fp)/Ab
#    dCAbbydt = (Fr*(CAr - CAb) - (D + Fp)*(CAd - CAb))/(Ab*Hb)
#    dCBbbydt = (Fr*(CBr - CBb) - (D + Fp)*(CBd - CBb))/(Ab*Hb)
#    dCCbbydt = (Fr*(CCr - CCb) - (D + Fp)*(Ccd - CCb))/(Ab*Hb)
#    dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

    # Return the derivative.
#   return np.array([dCabydt, dCbbydt])

def _measurement(x, parameters):
    yindices = parameters['yindices']
    # Return the measurement.
    return x[yindices]

def _get_greybox_parameters():
    """ Get the parameter values for the 
        CSTRs with flash example. """

    # Parameters.
    alphaA = parameters['alphaA']
    alphaB = parameters['alphaB']
    pho = parameters['pho']
    Cp = parameters['Cp']
    Ar = parameters['Ar']
    Ab = parameters['Ab']
    kr = parameters['kr']
    kb = parameters['kb']
    delH1 = parameters['delH1']
    EbyR = parameters['EbyR']
    k1star = parameters['k1star']
    Td = parameters['Td']

    # Store the dimensions.
    Nx, Nu, Np, Ny = 8, 4, 2, 8
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample time.
    parameters['Delta'] = 1. # min

    # Get the steady states.
    parameters['xs'] = np.array([50., 1., 0., 0., 313.,
                                 50., 1., 0., 0., 313.])
    parameters['us'] = np.array([30., -10000., 10., 0.])
    parameters['ps'] = np.array([5., 300])

    # Get the constraints.
    parameters['ulb'] = np.array([30., -8000., 5., 0.])
    parameters['uub'] = np.array([50., 0., 15., 8000.])

    # The C matrix for the plant.
    parameters['yindices'] = [0, 4, 5, 9]
    parameters['tsteps_steady'] = 60

    # Return the parameters dict.
    return parameters

def _get_plant_parameters():
    """ Get the parameter values for the 
        CSTRs with flash example. """

    # Parameters.
    parameters = {}
    parameters['alphaA'] = 6.
    parameters['alphaB'] = 0.6
    parameters['alphaC'] = 0.5 
    parameters['pho'] = 10. # Kg/m^3
    parameters['Cp'] = 3. # KJ/(Kg-K)
    parameters['Ar'] = 2. # m^2 
    parameters['Ab'] = 1. # m^2 
    parameters['kr'] = 5. # m^2
    parameters['kb'] = 5. # m^2
    parameters['delH1'] = 30 # kJ/mol
    parameters['delH2'] = 10 # kJ/mol
    parameters['EbyR'] = 10 # K
    parameters['k1star'] = 30. # 1/min
    parameters['k2star'] = 10. # 1/min
    parameters['Td'] = 300 # K

    # Store the dimensions.
    Nx, Nu, Np, Ny = 10, 4, 2, 8
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample time.
    parameters['Delta'] = 1. # min

    # Get the steady states.
    parameters['xs'] = np.array([50., 1., 0., 0., 313.,
                                 50., 1., 0., 0., 313.])
    parameters['us'] = np.array([5., -1000., 1., -1000.])
    parameters['ps'] = np.array([10., 300])

    # Get the constraints.
    parameters['ulb'] = np.array([2., -2000., 0.5, -2000.])
    parameters['uub'] = np.array([10., 0., 1.5, 0.])

    # The C matrix for the plant.
    parameters['yindices'] = [0, 4, 5, 9]
    parameters['tsteps_steady'] = 60
    parameters['Rv'] = 0*np.diag(np.array([1e-4, 1e-6, 1e-6, 1e-4, 
                                           1e-4, 1e-6, 1e-6, 1e-4]))

    # Return the parameters dict.
    return parameters

def _get_rectified_xs(*, parameters):
    """ Get the steady state of the plant. """
    # (xs, us, ps)
    xs = parameters['xs']
    us = parameters['us']
    ps = parameters['ps']
    plant_ode = lambda x, u, p: _plant_ode(x, u, p, parameters)
    # Construct the casadi class.
    model = mpc.DiscreteSimulator(plant_ode,
                                  parameters['Delta'],
                                  [parameters['Nx'], parameters['Nu'], 
                                   parameters['Np']], 
                                  ["x", "u", "p"])
    # Steady state of the plant.
    for _ in range(360):
        xs = model.sim(xs, us, ps)
    # Return the disturbances.
    return xs

def _get_model(*, parameters, plant=True):
    """ Return a nonlinear plant simulator object."""
    if plant:
        # Construct and return the plant.
        plant_ode = lambda x, u, p: _plant_ode(x, u, p, parameters)
        measurement = lambda x: _measurement(x, parameters)
        xs = parameters['xs'][:, np.newaxis]
        return NonlinearPlantSimulator(fxup = plant_ode,
                                        hx = measurement,
                                        Rv = parameters['Rv'], 
                                        Nx = parameters['Nx'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                        sample_time = parameters['Delta'], 
                                        x0 = xs)
    else:
        # Construct and return the grey-box model.
        tworeac_greybox_ode = lambda x, u, p: _tworeac_greybox_ode(x, u, 
                                                               p, parameters)
        xs = parameters['xs'][:-1, np.newaxis]
        return NonlinearPlantSimulator(fxup = tworeac_greybox_ode,
                                        hx = _tworeac_measurement,
                                        Rv = 0*np.eye(parameters['Ny']), 
                                        Nx = parameters['Ng'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                    sample_time = parameters['sample_time'], 
                                        x0 = xs)

def _gen_train_val_data(*, parameters, num_traj,
                           Nsim_train, Nsim_trainval, 
                           Nsim_val, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    data_list = []
    ulb = parameters['ulb']
    uub = parameters['uub']
    tsteps_steady = parameters['tsteps_steady']
    p = parameters['ps'][:, np.newaxis]

    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = _get_model(parameters=parameters, plant=True)
        np.random.seed(seed)
        us_init = np.tile(np.random.uniform(ulb, uub), (tsteps_steady, 1))

        # Get input trajectories for different simulatios.
        if traj == num_traj-1:
            "Get input for train val simulation."
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=24, num_steps=Nsim_val,
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+1)
        elif traj == num_traj-2:
            "Get input for validation simulation."
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=9, num_steps=Nsim_trainval,
                                 lb=ulb, ub=uub,
                                 mean_change=20, sigma_change=2, seed=seed+2)
        else:
            "Get input for training simulation."
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=54, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+3)

        # Complete input profile and run open-loop simulation.
        u = np.concatenate((us_init, u), axis=0)
        for t in range(tsteps_steady + Nsim):
            plant.step(u[t:t+1, :].T, p)
        data_list.append(SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u).squeeze(),
                y=np.asarray(plant.y[0:-1]).squeeze()))
    # Return the data list.
    return data_list

#def _get_greybox_val_preds(*, parameters, training_data):
#    """ Use the input profile to compute 
#        the prediction of the grey-box model
#        on the validation data. """
#    model = _get_tworeac_model(parameters=parameters, plant=False)
#    tsteps_steady = parameters['tsteps_steady']
#    p = parameters['ps'][:, np.newaxis]
#    u = training_data[-1].u[:, np.newaxis]
#    Nsim = u.shape[0]
    # Run the open-loop simulation.
#    for t in range(Nsim):
#        model.step(u[t:t+1, :], p)
#    data = SimData(t=None,
#                   x=None,
#                   u=None,
#                   y=np.asarray(model.y[tsteps_steady:-1]).squeeze())
#    return data

def main():
    """ Get the parameters/training/validation data."""
    # Get parameters.
    parameters = _get_plant_parameters()
    parameters['xs'] = _get_rectified_xs(parameters=parameters)
    breakpoint()
    # Generate training data.
    training_data = _gen_train_val_data(parameters=parameters, num_traj=3,
                                        Nsim_train=27*60, Nsim_trainval=3*60,
                                        Nsim_val=12*60, seed=10)
    #greybox_val_data = _get_greybox_val_preds(parameters=
    #                                        parameters, 
    #                                        training_data=training_data)
    cstr_flash_parameters = dict(parameters=parameters, 
                              training_data=training_data)
    # Save data.
    PickleTool.save(data_object=cstr_flash_parameters, 
                    filename='cstr_flash_parameters.pickle')

main()