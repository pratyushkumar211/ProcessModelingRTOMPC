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
import scipy.linalg
from hybridid import (PickleTool, NonlinearPlantSimulator,
                      c2d, sample_prbs_like, SimData,
                      NonlinearMHEEstimator)
from hybridid import _cstr_flash_plant_ode as _plant_ode
from hybridid import _cstr_flash_greybox_ode as _greybox_ode
from hybridid import _cstr_flash_measurement as _measurement

def _get_greybox_parameters():
    """ Get the parameter values for the 
        CSTRs with flash example. """

    # Parameters.
    parameters = {}
    parameters['alphaA'] = 6.
    parameters['alphaB'] = 1.
    parameters['pho'] = 5. # Kg/m^3
    parameters['Cp'] = 6. # KJ/(Kg-K)
    parameters['Ar'] = 3. # m^2
    parameters['Ab'] = 3. # m^2
    parameters['kr'] = 2. # m^2
    parameters['kb'] = 2. # m^2
    parameters['delH1'] = 70. # kJ/mol
    parameters['E1byR'] = 200 # K
    parameters['k1star'] = 0.5 # 1/min
    parameters['Td'] = 300 # K

    # Store the dimensions.
    Ng, Nu, Np, Ny = 8, 4, 2, 6
    parameters['Ng'] = Ng
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample time.
    parameters['Delta'] = 1. # min

    # Get the steady states.
    parameters['xs'] = np.array([50., 1., 0., 313.,
                                 50., 1., 0., 313.])
    parameters['us'] = np.array([10., 200., 4., 200.])
    parameters['ps'] = np.array([4., 300])

    # The C matrix for the plant.
    parameters['tsteps_steady'] = 120
    parameters['yindices'] = [0, 1, 3, 4, 5, 7]

    # Get the constraints.
    parameters['ulb'] = np.array([5., 0., 2., 0.])
    parameters['uub'] = np.array([15., 400., 6., 400.])

    # Return the parameters dict.
    return parameters

def _get_plant_parameters():
    """ Get the parameter values for the
        CSTRs with flash example. """

    # Parameters.
    parameters = {}
    parameters['alphaA'] = 6.
    parameters['alphaB'] = 1.
    parameters['alphaC'] = 1.
    parameters['pho'] = 5. # Kg/m^3
    parameters['Cp'] = 3. # KJ/(Kg-K)
    parameters['Ar'] = 3. # m^2
    parameters['Ab'] = 3. # m^2
    parameters['kr'] = 2. # m^2
    parameters['kb'] = 2. # m^2
    parameters['delH1'] = 100. # kJ/mol
    parameters['delH2'] = 100. # kJ/mol
    parameters['E1byR'] = 200 # K
    parameters['E2byR'] = 50 # K
    parameters['k1star'] = 0.5 # 1/min
    parameters['k2star'] = 0.8 # 1/min
    parameters['Td'] = 300 # K

    # Store the dimensions.
    Nx, Nu, Np, Ny = 10, 4, 2, 6
    parameters['Nx'] = Nx
    parameters['Nu'] = Nu
    parameters['Ny'] = Ny
    parameters['Np'] = Np

    # Sample time.
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([50., 1., 0., 0., 313.,
                                 50., 1., 0., 0., 313.])
    parameters['us'] = np.array([10., 200., 6., 200.])
    parameters['ps'] = np.array([6., 310.])

    # Get the constraints.
    parameters['ulb'] = np.array([5., 0., 2., 0.])
    parameters['uub'] = np.array([15., 400., 8., 400.])

    # The C matrix for the plant.
    parameters['yindices'] = [0, 1, 4, 5, 6, 9]
    parameters['tsteps_steady'] = 120
    parameters['Rv'] = 0*np.diag([0.8, 1e-3, 1., 0.8, 1e-3, 1.])
    
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
    measurement = lambda x: _measurement(x, parameters)
    if plant:
        # Construct and return the plant.
        plant_ode = lambda x, u, p: _plant_ode(x, u, p, parameters)
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
        greybox_ode = lambda x, u, p: _greybox_ode(x, u, p, parameters)
        xs = parameters['xs'][:, np.newaxis]
        return NonlinearPlantSimulator(fxup = greybox_ode,
                                        hx = measurement,
                                        Rv = 0*np.eye(parameters['Ny']), 
                                        Nx = parameters['Ng'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                        sample_time = parameters['Delta'], 
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
    np.random.seed(seed)
    
    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = _get_model(parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tsteps_steady, 1))

        # Get input trajectories for different simulatios.
        if traj == num_traj-1:
            "Get input for train val simulation."
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=24, num_steps=Nsim_val,
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed)
        elif traj == num_traj-2:
            "Get input for validation simulation."
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=6, num_steps=Nsim_trainval,
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed)
        else:
            "Get input for training simulation."
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=8, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed)
        seed += 1
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

def _get_greybox_val_preds(*, parameters, training_data):
    """ Use the input profile to compute
        the prediction of the grey-box model
        on the validation data. """
    model = _get_model(parameters=parameters, plant=False)
    tsteps_steady = parameters['tsteps_steady']
    Ng = parameters['Ng']
    p = parameters['ps'][:, np.newaxis]
    u = training_data[-1].u
    Nsim = u.shape[0]
   # Run the open-loop simulation.
    for t in range(Nsim):
        model.step(u[t:t+1, :], p)
    # Insert Nones.
    x = np.asarray(model.x[0:-1]).squeeze()
    x = np.insert(x, [3, 7], np.nan*np.ones((Nsim, 2)), axis=1)
    data = SimData(t=np.asarray(model.t[0:-1]).squeeze(), x=x,
                   u=np.asarray(model.u).squeeze(),
                   y=np.asarray(model.y[0:-1]).squeeze())
    return data

def _get_mhe_estimator(*, parameters):
    """ Filter the training data using a combination 
        of grey-box model and an input disturbance model. """

    def state_space_model(Ng, Bd, Nd, ps, parameters):
        """ Augmented state-space model for moving horizon estimation. """
        return lambda x, u : np.concatenate((_greybox_ode(x[:Ng], 
                                             u, ps, parameters) + Bd @ x[Ng:],
                                             np.zeros((Nd,))), axis=0)
    
    def measurement_model(Ng, parameters):
        """ Augmented measurement model for moving horizon estimation. """
        return lambda x : _measurement(x[:Ng], parameters)

    # Get sizes.
    (Ng, Nu, Ny) = (parameters['Ng'], parameters['Nu'], parameters['Ny'])
    Nd = Ny

    # Get the disturbance model.
    #Bd = np.eye(Ng)
    Bd = np.zeros((Ng, Nd))
    Bd[1, 0] = 1.
    Bd[2, 1] = 1.
    Bd[3, 2] = 1.
    Bd[5, 3] = 1.
    Bd[6, 4] = 1.
    Bd[7, 5] = 1.

    # Initial states.
    xs = parameters['xs'][:, np.newaxis]
    ps = parameters['ps'][:, np.newaxis]
    us = parameters['us'][:, np.newaxis]
    ys = _measurement(xs, parameters)
    ds = np.zeros((Nd, 1))

    # Noise covariances.
    Qwx = np.eye(Ng)
    Qwd = 4*np.eye(Nd)
    Rv = np.eye(Ny)

    # MHE horizon length.
    Nmhe = 15

    # Continuous time functions, fxu and hx.
    fxud = state_space_model(Ng, Bd, Nd, ps, parameters)
    hxd = measurement_model(Ng, parameters)
    
    # Initial data.
    xprior = np.concatenate((xs, ds), axis=0)
    xprior = np.repeat(xprior.T, Nmhe, axis=0)
    u = np.repeat(us.T, Nmhe, axis=0)
    y = np.repeat(ys.T, Nmhe+1, axis=0)
    
    # Penalty matrices.
    Qwxinv = np.linalg.inv(Qwx)
    Qwdinv = np.linalg.inv(Qwd)
    Qwinv = scipy.linalg.block_diag(Qwxinv, Qwdinv)
    P0inv = Qwinv
    Rvinv = np.linalg.inv(Rv)

    # Get the augmented models.
    fxu = mpc.getCasadiFunc(fxud, [Ng+Nd, Nu], ["x", "u"],
                            rk4=True, Delta=parameters['Delta'], 
                            M=10)
    hx = mpc.getCasadiFunc(hxd, [Ng+Nd], ["x"])
    
    # Create a filter object and return.
    return NonlinearMHEEstimator(fxu=fxu, hx=hx,
                                 Nmhe=Nmhe, Nx=Ng+Nd, 
                                 Nu=Nu, Ny=Ny, xprior=xprior,
                                 u=u, y=y, P0inv=P0inv,
                                 Qwinv=Qwinv, Rvinv=Rvinv), Bd

def _get_gb_mhe_processed_training_data(*, parameters, training_data):
    """ Process all the training data and add grey-box state estimates. """
    def get_state_estimates(mhe_estimator, y, uprev, Ng):
        """Use the filter object to perform state estimation. """
        return np.split(mhe_estimator.solve(y, uprev), [Ng])
    # Data list.
    Ng = parameters['Ng']
    processed_data = []
    for data in training_data:
        mhe_estimator, Bd = _get_mhe_estimator(parameters=parameters)
        Nsim = len(data.t)
        (u, y) = (data.u, data.y)
        xhats = [mhe_estimator.xhat[-1][:Ng]]
        dhats = [mhe_estimator.xhat[-1][Ng:]]
        for t in range(Nsim-1):
            uprevt = u[t:t+1, :].T
            yt = y[t+1:t+2, :].T
            xhat, dhat = get_state_estimates(mhe_estimator, yt, uprevt, Ng)
            xhats.append(xhat)
            dhats.append(dhat)
        xhats = np.asarray(xhats)
        dhats = np.asarray(dhats)
        processed_data.append(SimData(t=data.t, u=data.u, y=data.y,
                                      x=xhats))
    # Return the processed data list.
    return processed_data

def main():
    """ Get the parameters/training/validation data."""
    # Get parameters.
    plant_pars = _get_plant_parameters()
    plant_pars['xs'] = _get_rectified_xs(parameters=plant_pars)
    greybox_pars = _get_greybox_parameters()

    # Generate training data.
    training_data = _gen_train_val_data(parameters=plant_pars, num_traj=3,
                                        Nsim_train=4*60, Nsim_trainval=3*60,
                                        Nsim_val=12*60, seed=10)
    
    greybox_processed_data = _get_gb_mhe_processed_training_data(parameters=
                                                                greybox_pars,
                                                    training_data=training_data)
    
    greybox_val_data = _get_greybox_val_preds(parameters=greybox_pars,
                                              training_data=training_data)
    cstr_flash_parameters = dict(plant_pars=plant_pars,
                                 greybox_pars=greybox_pars,
                                 training_data=training_data,
                                 greybox_processed_data=greybox_processed_data,
                                 greybox_val_data=greybox_val_data)
    # Save data.
    PickleTool.save(data_object=cstr_flash_parameters,
                    filename='cstr_flash_parameters.pickle')

main()