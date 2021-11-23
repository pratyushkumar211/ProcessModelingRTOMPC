# [depends] linNonlinMPC.py
import sys
import random
import numpy as np
import mpctools as mpc
import scipy.linalg
import matplotlib.pyplot as plt
import casadi
import collections
import pickle
import plottools
import time
from linNonlinMPC import getXsYsSscost, c2dNonlin

# Custom class to store datasets.
SimData = collections.namedtuple('SimData',
                                ['t', 'x', 'u', 'y', 'p'])

class PickleTool:
    """ Class which contains a few static methods for saving and
        loading pkl data files conveniently. """

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
                        lb, ub, mean_change, sigma_change, 
                        num_constraint=0):
    """ Sample a PRBS like sequence.
    num_change: Number of changes in the signal.
    num_steps: Number of steps in the signal.
    mean_change: mean_value after which a 
                 change in the signal is desired.
    sigma_change: standard deviation of changes in the signal.
    """

    # Signal dimension. 
    signal_dimension = lb.shape[0]

    # Squeeze the bounds so that the signals are 1D arrays. 
    lb = lb.squeeze() # Squeeze the vectors.
    ub = ub.squeeze() # Squeeze the vectors.

    # Sample values. 
    values = (ub-lb)*np.random.rand(num_change, signal_dimension) + lb

    # Sample some values at constraints.
    if num_constraint > 0:

        # Make sure even number of values at the constraints 
        # are required, so that you can equally sample at both the upper and 
        # lower limits of the constraints.
        assert num_constraint % 2 == 0
        assert num_constraint < num_change

        # Sample.
        constraint_indices = random.sample(range(0, num_change), num_constraint)
        values[constraint_indices[:num_constraint//2]] = ub
        values[constraint_indices[num_constraint//2:]] = lb        
    
    # Sample how many time each value should be repeated. 
    repeat = _sample_repeats(num_change, num_steps,
                             mean_change, sigma_change)

    # Get the signal.
    signal = np.repeat(values, repeat, axis=0)

    # Return.
    return signal

def get_noisy_drift_signal(*, t0val, tfval, noise_Rv, num_steps, seed=2):
    """ Get a noisy drift signal. """

    # Signal.
    numVar = len(t0val)
    signal = np.linspace(t0val, tfval, num_steps)
    ynoise_std = np.diag(np.sqrt(noise_Rv))
    signal += ynoise_std*np.random.rand(num_steps, numVar)

    # Get a noisy drift signal.
    return signal

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
    """ Quick open-loop simulation. """

    # Number of simulation timesteps.
    Nsim = u.shape[0]
    y, x = [], []
    x += [x0]
    xt = x0
    
    # Run the simulation.
    for t in range(Nsim):
        y += [hx(xt)]
        xt = fxu(xt, u[t, :])
        x += [xt]

    # Get arrays for measurements and states.
    y = np.asarray(y)
    x = np.asarray(x[:-1])

    # Return.
    return x, y

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

def get_train_val_data(*, Ntstart, Np, xuyscales, data_list):
    """ Get the data for training and validation in 
        appropriate format for training.
        All data are already scaled.
    """

    # Get scaling parameters.
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']
    Nx, Nu, Ny = len(xmean), len(umean), len(ymean)

    # Lists to store data.
    xseq, useq, yseq = [], [], []
    y0, z0, yz0 = [], [], []

    # Loop through the data list.
    for data in data_list:
        
        # Scale data.
        x = (data.x - xmean)/xstd
        u = (data.u - umean)/ustd
        y = (data.y - ymean)/ystd

        # Get the input and output trajectory.
        x_traj = x[Ntstart:, :][np.newaxis, ...]
        u_traj = u[Ntstart:, :][np.newaxis, ...]
        y_traj = y[Ntstart:, :][np.newaxis, ...]

        # Get initial states.
        yp0seq = y[Ntstart-Np:Ntstart, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[Ntstart-Np:Ntstart, :].reshape(Np*Nu, )[np.newaxis, :]
        y0_traj = y[Ntstart, np.newaxis, :]
        z0_traj = np.concatenate((yp0seq, up0seq), axis=-1)
        yz0_traj = np.concatenate((y0_traj, z0_traj), axis=-1)

        # Collect the trajectories in list.
        xseq += [x_traj]
        useq += [u_traj]
        yseq += [y_traj]
        y0 += [y0_traj]
        z0 += [z0_traj]
        yz0 += [yz0_traj]
    
    # Get training, trainval, and validation data in compact dicts.
    train_data = dict(xseq=np.concatenate(xseq[:-2], axis=0),
                      useq=np.concatenate(useq[:-2], axis=0),
                      yseq=np.concatenate(yseq[:-2], axis=0),
                      y0=np.concatenate(y0[:-2], axis=0),
                      z0=np.concatenate(z0[:-2], axis=0),   
                      yz0=np.concatenate(yz0[:-2], axis=0))
    trainval_data = dict(xseq=xseq[-2], useq=useq[-2], yseq=yseq[-2], 
                         y0=y0[-2], z0=z0[-2], yz0=yz0[-2])
    val_data = dict(xseq=xseq[-1], useq=useq[-1], yseq=yseq[-1], 
                    y0=y0[-1], z0=z0[-1], yz0=yz0[-1])
    
    # Return.
    return (train_data, trainval_data, val_data)

def get_rectified_xs(*, ode, parameters, Nsim):
    """ Get a rectified steady state of the plant
        upto numerical precision. """

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
    for _ in range(Nsim):
        xs = model.sim(xs, us, ps)

    # Return.
    return xs