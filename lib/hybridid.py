# [depends] linNonlinMPC.py

import sys
import numpy as np
import mpctools as mpc
import scipy.linalg
import matplotlib.pyplot as plt
import casadi
import collections
import pickle
import plottools
import time
from linNonlinMPC import NonlinearPlantSimulator

SimData = collections.namedtuple('SimData',
                                ['t', 'x', 'u', 'y'])

class PickleTool:
    """Class which contains a few static methods for saving and
    loading pkl data files conveniently."""
    @staticmethod
    def load(filename, type='write'):
        """Wrapper to load data."""
        if type == 'read':
            with open(filename, "rb") as stream:
                return pickle.load(stream)
        if type == 'write':
            with open(filename, "wb") as stream:
                return pickle.load(stream)
    
    @staticmethod
    def save(data_object, filename):
        """Wrapper to pickle a data object."""
        with open(filename, "wb") as stream:
            pickle.dump(data_object, stream)

def _sample_repeats(num_change, num_simulation_steps,
                    mean_change, sigma_change):
    """ Sample the number of times a repeat in each
        of the sampled vector is required."""
    repeat = sigma_change*np.random.randn(num_change-1) + mean_change
    repeat = np.floor(repeat)
    repeat = np.where(repeat<=0., 0., repeat)
    repeat = np.append(repeat, num_simulation_steps-np.int(np.sum(repeat)))
    return repeat.astype(int)

def sample_prbs_like(*, num_change, num_steps, 
                        lb, ub, mean_change, sigma_change, seed=1):
    """Sample a PRBS like sequence.
    num_change: Number of changes in the signal.
    num_simulation_steps: Number of steps in the signal.
    mean_change: mean_value after which a 
                 change in the signal is desired.
    sigma_change: standard deviation of changes in the signal.
    """
    signal_dimension = lb.shape[0]
    lb = lb.squeeze() # Squeeze the vectors.
    ub = ub.squeeze() # Squeeze the vectors.
    np.random.seed(seed)
    values = (ub-lb)*np.random.rand(num_change, signal_dimension) + lb
    repeat = _sample_repeats(num_change, num_steps,
                             mean_change, sigma_change)
    return np.repeat(values, repeat, axis=0)

def resample_fast(*, x, xDelta, newDelta, resample_type):
    """ Resample with either first of zero-order hold. """
    Delta_ratio = int(xDelta/newDelta)
    if resample_type == 'zoh':
        return np.repeat(x, Delta_ratio, axis=0)
    else:
        x = np.concatenate((x, x[-1, np.newaxis, :]), axis=0)
        return np.concatenate([np.linspace(x[t, :], x[t+1, :], Delta_ratio)
                               for t in range(x.shape[0]-1)], axis=0)

# def interpolate_yseq(yseq, Npast, Ny):
#     """ y is of dimension: (None, (Npast+1)*p)
#         Return y of dimension: (None, Npast*p). """
#     yseq_interp = []
#     for t in range(Npast):
#         yseq_interp.append(0.5*(yseq[t*Ny:(t+1)*Ny] + yseq[(t+1)*Ny:(t+2)*Ny]))
#     # Return.
#     return np.concatenate(yseq_interp)

def get_scaling(*, data):
    """ Scale the input/output. """
    # Xmean.
    xmean = np.mean(data.x, axis=0)
    xstd = np.std(data.x, axis=0)
    # Umean.
    umean = np.mean(data.u, axis=0)
    ustd = np.std(data.u, axis=0)
    # Ymean.
    ymean = np.mean(data.y, axis=0)
    ystd = np.std(data.y, axis=0)
    # Return.
    return dict(xscale = (xmean, xstd), 
                uscale = (umean, ustd), 
                yscale = (ymean, ystd))

def quick_sim(fxu, hx, x0, u):
    """ Do a quick open-loop simulation. """
    Nsim = u.shape[0]
    y, x = [], []
    x.append(x0)
    xt = x0
    for t in range(Nsim):
        y.append(hx(xt))
        xt = fxu(xt, u[t, :])
        x.append(xt)
    y = np.asarray(y)
    x = np.asarray(x[:-1])
    # Return.
    return x, y

def get_train_val_data(*, tthrow, Np, xuyscales, data_list):
    """ Get the data for training/validation in appropriate format after 
        scaling. """

    # Get scaling pars.
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']
    Ny, Nu = len(ymean), len(umean)

    # Lists to store data.
    inputs, yz0, yz, x0, outputs = [], [], [], [], []

    # Loop through the data list.
    for data in data_list:
        
        # Scale data.
        x = (data.x - xmean)/xstd
        u = (data.u - umean)/ustd
        y = (data.y - ymean)/ystd
        
        # Get input/output trajectory.
        u_traj = u[tthrow:, :][np.newaxis, ...]
        y_traj = y[tthrow:, :][np.newaxis, ...]

        # Get initial states.
        yp0seq = y[tthrow-Np:tthrow, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[tthrow-Np:tthrow, :].reshape(Np*Nu, )[np.newaxis, :]
        y0 = y[tthrow, np.newaxis, :]
        yz0_traj = np.concatenate((y0, yp0seq, up0seq), axis=-1)
        x0_traj = x[tthrow, np.newaxis, :]

        # Get z_traj.
        Nt = u.shape[0]
        z_traj = []
        for t in range(tthrow, Nt):
            ypseq = y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
            upseq = u[t-Np:t, :].reshape(Np*Nu, )[np.newaxis, :]
            z_traj += [np.concatenate((ypseq, upseq), axis=-1)]
        z_traj = np.concatenate(z_traj, axis=0)[np.newaxis, ...]
        yz_traj = np.concatenate((y_traj, z_traj), axis=-1)

        # Collect the trajectories in list.
        inputs += [u_traj]
        yz0 += [yz0_traj]
        yz += [yz_traj]
        x0 += [x0_traj]
        outputs += [y_traj]
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      yz0=np.concatenate(yz0[:-2], axis=0),
                      yz=np.concatenate(yz[:-2], axis=0),
                      x0=np.concatenate(x0[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], yz0=yz0[-2],
                          yz=yz[-2], x0=x0[-2], outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], yz0=yz0[-1],
                    yz=yz[-1], x0=x0[-1], outputs=outputs[-1])
    # Return.
    return (train_data, trainval_data, val_data)

def measurement(x, parameters):
    yindices = parameters['yindices']
    # Return the measurement.
    return x[yindices]

def get_rectified_xs(*, ode, parameters):
    """ Get the steady state of the plant. """

    # ODE Func.
    ode_func = lambda x, u, p: ode(x, u, p, parameters)

    # Some parameters.
    xs, us, ps = parameters['xs'], parameters['us'], parameters['ps']
    Nx, Nu, Np = parameters['Nx'], parameters['Nu'], parameters['Np']
    Delta = parameters['Delta']

    # Construct the casadi class.
    model = mpc.DiscreteSimulator(ode_func, Delta,
                                  [Nx, Nu, Np], ["x", "u", "p"])

    # Steady state of the plant.
    for _ in range(360):
        xs = model.sim(xs, us, ps)
    # Return the disturbances.
    return xs

def get_model(*, ode, parameters, plant=True):
    """ Return a nonlinear plant simulator object."""
    
    # Lambda functions for ODEs.
    ode_func = lambda x, u, p: ode(x, u, p, parameters)
    meas_func = lambda x: measurement(x, parameters)

    # Get sizes. 
    Nx, Nu = parameters['Nx'], parameters['Nu']
    Np, Ny = parameters['Np'], parameters['Ny']

    # Steady state/measurement noise/sample time.
    xs = parameters['xs'][:, np.newaxis]
    if plant:
        Rv = parameters['Rv']
    else:
        Rv = 0*np.eye(Ny)
    Delta = parameters['Delta']

    # Return a simulator object.
    return NonlinearPlantSimulator(fxup=ode_func, hx=meas_func,
                                   Rv = Rv, Nx = Nx, Nu = Nu, Np = Np, Ny = Ny,
                                   sample_time = Delta, x0 = xs)