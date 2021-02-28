# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
import copy
from linNonlinMPC import NonlinearPlantSimulator
from hybridid import get_scaling, _resample_fast, interpolate_yseq

def plant_ode(x, u, p, parameters):
    """ Simple ODE describing a 2D system. """
    # Extract the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    
    # Extract the plant states into meaningful names.
    (Ca, Cb, Cc) = x[0:3]
    Ca0 = u[0:1]
    tau = p[0:1]

    # Write the ODEs.
    dCabydt = (Ca0-Ca)/tau - k1*Ca
    dCbbydt = k1*Ca - 3*k2*(Cb**3) + 3*k3*Cc - Cb/tau
    dCcbydt = k2*(Cb**3) - k3*Cc - Cc/tau

    # Return the derivative.
    return np.array([dCabydt, dCbbydt, dCcbydt])

def greybox_ode(x, u, p, parameters):
    """ Simple ODE describing the grey-box plant. """
    # Extract the parameters.
    k1 = parameters['k1']

    # Extract the plant states into meaningful names.
    (Ca, Cb) = x[0:2]
    Ca0 = u[0]
    tau = p[0]

    # Write the ODEs.
    dCabydt = (Ca0-Ca)/tau - k1*Ca
    dCbbydt = k1*Ca - Cb/tau

    # Return the derivative.
    return np.array([dCabydt, dCbbydt])

def measurement(x, parameters):
    Ny = parameters['Ny']
    # Return the measurement.
    return x[0:Ny]

def get_parameters():
    """ Get the parameter values for the 
        three reaction example. """
    
    # Parameters.
    parameters = {}
    parameters['k1'] = 1. # m^3/min.
    parameters['k2'] = 0.01 # m^3/min.
    parameters['k3'] = 0.05 # m^3/min.

    # Store the dimensions.
    parameters['Nx'] = 3
    parameters['Ng'] = 2
    parameters['Nu'] = 1
    parameters['Ny'] = 2
    parameters['Np'] = 1

    # Sample time.
    parameters['Delta'] = 1. # min.

    # Get the steady states.
    parameters['xs'] = np.array([1., 0.5, 0.5]) # to be updated.
    parameters['us'] = np.array([1.5]) # Ca0s
    parameters['ps'] = np.array([10.]) # tau (min)

    # Get the constraints. 
    ulb = np.array([0.5])
    uub = np.array([2.5])
    parameters['ulb'] = ulb
    parameters['uub'] = uub

    # Number of time-steps to keep the plant at steady.
    parameters['tsteps_steady'] = 10

    # Measurement noise.
    parameters['Rv'] = 0*np.diag([1e-3, 1e-3])

    # Return the parameters dict.
    return parameters

def get_rectified_xs(*, parameters):
    """ Get the steady state of the plant. """
    # (xs, us, ps)
    xs = parameters['xs']
    us = parameters['us']
    ps = parameters['ps']
    tworeac_plant_ode = lambda x, u, p: plant_ode(x, u, p, parameters)

    # Construct the casadi class.
    model = mpc.DiscreteSimulator(tworeac_plant_ode, 
                                  parameters['Delta'],
                                  [parameters['Nx'], parameters['Nu'], 
                                   parameters['Np']], 
                                  ["x", "u", "p"])
    # Steady state of the plant.
    for _ in range(360):
        xs = model.sim(xs, us, ps)
    # Return the disturbances.
    return xs

def get_model(*, parameters, plant=True):
    """ Return a nonlinear plant simulator object."""
    tworeac_measurement = lambda x: measurement(x, parameters)
    if plant:
        # Construct and return the plant.
        tworeac_plant_ode = lambda x, u, p: plant_ode(x, u, p, parameters)
        xs = parameters['xs'][:, np.newaxis]
        return NonlinearPlantSimulator(fxup = tworeac_plant_ode,
                                        hx = tworeac_measurement,
                                        Rv = parameters['Rv'], 
                                        Nx = parameters['Nx'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                    sample_time = parameters['Delta'], 
                                        x0 = xs)
    else:
        # Construct and return the grey-box model.
        tworeac_greybox_ode = lambda x, u, p: greybox_ode(x, u, p, parameters)
        Ng = parameters['Ng']
        xs = parameters['xs'][:Ng, np.newaxis]
        return NonlinearPlantSimulator(fxup = tworeac_greybox_ode,
                                        hx = tworeac_measurement,
                                        Rv = 0*parameters['Rv'],
                                        Nx = parameters['Ng'], 
                                        Nu = parameters['Nu'], 
                                        Np = parameters['Np'], 
                                        Ny = parameters['Ny'],
                                    sample_time = parameters['Delta'], 
                                        x0 = xs)

def get_train_val_data(*, Np, xuyscales, parameters, data_list):
    """ Get the data for training in appropriate format. """

    # Get some parameters.
    ts = parameters['tsteps_steady']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']

    # Lists to store data.
    inputs, xz0, yz0, yz, outputs = [], [], [], [], []

    # Loop through the data list.
    for data in data_list:
        
        # Scale data.
        x = (data.x - xmean)/xstd
        u = (data.u - umean)/ustd
        y = (data.y - ymean)/ystd
                
        # Get input trajectory.
        u_traj = u[ts:][np.newaxis, :]
        
        # Get initial states.
        x0 = x[ts, :][np.newaxis, :]
        y0 = y[ts, :][np.newaxis, :]
        yp0seq = y[ts-Np:ts, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[ts-Np:ts, :].reshape(Np*Nu, )[np.newaxis, :]
        xz0_traj = np.concatenate((x0, yp0seq, up0seq), axis=-1)
        yz0_traj = np.concatenate((y0, yp0seq, up0seq), axis=-1)

        # Get output trajectory.
        y_traj = y[ts:, :][np.newaxis, ...]

        # Get yz_traj.
        Nt = u.shape[0]
        z_traj = []
        for t in range(ts, Nt):
            ypseq = y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
            upseq = u[t-Np:t, :].reshape(Np*Nu, )[np.newaxis, :]
            z_traj.append(np.concatenate((ypseq, upseq), axis=-1))
        z_traj = np.concatenate(z_traj, axis=0)[np.newaxis, ...]
        yz_traj = np.concatenate((y_traj, z_traj), axis=-1)

        # Collect the trajectories in list.
        inputs.append(u_traj)
        xz0.append(xz0_traj)
        yz0.append(yz0_traj)
        yz.append(yz_traj)
        outputs.append(y_traj)
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      xz0=np.concatenate(xz0[:-2], axis=0),
                      yz0=np.concatenate(yz0[:-2], axis=0),
                      yz=np.concatenate(yz[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], xz0=xz0[-2],
                         yz0=yz0[-2], yz=yz[-2], outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], xz0=xz0[-1],
                    yz0=yz0[-1], yz=yz[-1], outputs=outputs[-1])
    # Return.
    return (train_data, trainval_data, val_data)

def get_hybrid_pars(*, parameters, Npast, fnn_weights, xuscales):
    """ Get the hybrid model parameters. """

    hybrid_pars = copy.deepcopy(parameters)
    # Update sizes.
    Ng, Nu, Ny = parameters['Ng'], parameters['Nu'], parameters['Ny']
    hybrid_pars['Nx'] = parameters['Ng'] + Npast*(Nu + Ny)

    # Update steady state.
    ys = measurement(parameters['xs'])
    yspseq = np.tile(ys, (Npast, ))
    us = parameters['us']
    uspseq = np.tile(us, (Npast, ))
    xs = parameters['xs'][:Ng]
    hybrid_pars['xs'] = np.concatenate((xs, yspseq, uspseq))
    
    # NN pars.
    hybrid_pars['Npast'] = Npast
    hybrid_pars['fnn_weights'] = fnn_weights 

    # Scaling.
    hybrid_pars['xuscales'] = xuscales

    # Return.
    return hybrid_pars

def fnn(xG, z, Npast, fnn_weights):
    """ Compute the NN output. """
    nn_output = np.concatenate((xG, z))[:, np.newaxis]
    for i in range(0, len(fnn_weights)-2, 2):
        (W, b) = fnn_weights[i:i+2]
        nn_output = W.T @ nn_output + b[:, np.newaxis]
        nn_output = 1./(1. + np.exp(-nn_output))
    Wf = fnn_weights[-1]
    nn_output = (Wf.T @ nn_output)[:, 0]
    # Return.
    return nn_output

def hybrid_func(xGz, u, parameters):
    """ The augmented continuous time model. """

    # Extract a few parameters.
    Ng = parameters['Ng']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    ps = parameters['ps']
    Npast = parameters['Npast']
    Delta = parameters['Delta']
    fnn_weights = parameters['fnn_weights']
    xuscales = parameters['xuscales']
    xmean, xstd = xuscales['xscale']
    umean, ustd = xuscales['uscale']
    xGzmean = np.concatenate((xmean,
                              np.tile(xmean, (Npast, )), 
                              np.tile(umean, (Npast, ))))
    xGzstd = np.concatenate((xstd,
                             np.tile(xstd, (Npast, )), 
                             np.tile(ustd, (Npast, ))))
    
    # Get some vectors.
    xGz = (xGz - xGzmean)/xGzstd
    u = (u-umean)/ustd
    xG, ypseq, upseq = xGz[:Ng], xGz[Ng:Ng+Npast*Ny], xGz[-Npast*Nu:]
    z = xGz[Ng:]
    hxG = measurement(xG)
    
    # Get k1.
    k1 = greybox_ode(xG*xstd + xmean, u*ustd + umean, ps, parameters)/xstd
    k1 += fnn(xG, z, Npast, fnn_weights)

    # Interpolate for k2 and k3.
    ypseq_interp = interpolate_yseq(np.concatenate((ypseq, hxG)), Npast, Ny)
    z = np.concatenate((ypseq_interp, upseq))
    
    # Get k2.
    k2 = greybox_ode((xG + Delta*(k1/2))*xstd + xmean, u*ustd + umean, 
                      ps, parameters)/xstd
    k2 += fnn(xG + Delta*(k1/2), z, Npast, fnn_weights)

    # Get k3.
    k3 = greybox_ode((xG + Delta*(k2/2))*xstd + xmean, u*ustd + umean, 
                      ps, parameters)/xstd
    k3 += fnn(xG + Delta*(k2/2), z, Npast, fnn_weights)

    # Get k4.
    ypseq_shifted = np.concatenate((ypseq[Ny:], hxG))
    z = np.concatenate((ypseq_shifted, upseq))
    k4 = greybox_ode((xG + Delta*k3)*xstd + xmean, u*ustd + umean, 
                      ps, parameters)/xstd
    k4 += fnn(xG + Delta*k3, z, Npast, fnn_weights)
    
    # Get the current output/state and the next time step.
    xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
    zplus = np.concatenate((ypseq_shifted, upseq[Nu:], u))
    xGzplus = np.concatenate((xGplus, zplus))
    xGzplus = xGzplus*xGzstd + xGzmean

    # Return the sum.
    return xGzplus

def get_economic_opt_pars(*, Delta):
    """ Get economic MPC parameters for the tworeac example. """
    raw_mat_price = _resample_fast(x = np.array([[105.], [105.], 
                                                 [105.], [105.], 
                                                 [105.], [105.]]), 
                                   xDelta=2*60,
                                   newDelta=Delta,
                                   resample_type='zoh')
    product_price = _resample_fast(x = np.array([[170.], [210.], 
                                                 [120.], [160.], 
                                                 [160.], [160.]]),
                                   xDelta=2*60,
                                   newDelta=Delta,
                                   resample_type='zoh')
    cost_pars = 100*np.concatenate((raw_mat_price, product_price), axis=1)
    # Return the cost pars.
    return cost_pars

def sim_hybrid(hybrid_func, hybrid_pars, uval, training_data):
    """ Hybrid validation simulation to make 
        sure the above programmed function is 
        the same is what tensorflow is training. """
    
    # Get initial state.
    t = hybrid_pars['tsteps_steady']
    Np = hybrid_pars['Npast']
    Ng = hybrid_pars['Ng']
    Ny = hybrid_pars['Ny']
    Nu = hybrid_pars['Nu']
    y = training_data[-1].y
    u = training_data[-1].u
    yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[:, np.newaxis]
    up0seq = u[t-Np:t:, ].reshape(Np*Nu, )[:, np.newaxis]
    z0 = np.concatenate((yp0seq, up0seq))
    xG0 = y[t, :][:, np.newaxis]
    xGz0 = np.concatenate((xG0, z0))

    # Start the validation simulation.
    uval = uval[t:, :]
    Nval = uval.shape[0]
    hx = lambda x: measurement(x)
    fxu = lambda x, u: hybrid_func(x, u, hybrid_pars)
    x = xGz0[:, 0]
    yval, xGval = [], []
    xGval.append(x)
    for t in range(Nval):
        yval.append(hx(x))
        x = fxu(x, uval[t, :].T)
        xGval.append(x)
    yval = np.asarray(yval)
    xGval = np.asarray(xGval)[:-1, :Ng]
    # Return.
    return yval, xGval

def get_hybrid_pars_check_func(*, parameters, training_data, train):
    """ Get parameters for the hybrid function 
        and test by simulating on validation data. """

    # Get NN weights and parameters for the hybrid function.
    Np = train['Nps'][0]
    fnn_weights = train['trained_weights'][0][-1]
    xuscales = train['xuscales']
    hybrid_pars = get_hybrid_pars(parameters=parameters,
                                  Npast=Np,
                                  fnn_weights=fnn_weights,
                                  xuscales=xuscales)

    # Check the hybrid function.
    uval = training_data[-1].u
    ytfval = train['val_predictions'][0].y
    xGtfval = train['val_predictions'][0].x
    yval, xGval = sim_hybrid(hybrid_func, hybrid_pars,
                             uval, training_data)

    # Return the parameters.
    return hybrid_pars