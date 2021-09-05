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
                        num_constraint=0, seed=1):
    """ Sample a PRBS like sequence.
    num_change: Number of changes in the signal.
    num_steps: Number of steps in the signal.
    mean_change: mean_value after which a 
                 change in the signal is desired.
    sigma_change: standard deviation of changes in the signal.
    """
    signal_dimension = lb.shape[0]
    lb = lb.squeeze() # Squeeze the vectors.
    ub = ub.squeeze() # Squeeze the vectors.
    np.random.seed(seed) # Seed for numpy.
    random.seed(seed) # Seed for the random package.
    values = (ub-lb)*np.random.rand(num_change, signal_dimension) + lb
    # Sample some values at constraints.
    if num_constraint > 0:
        assert num_constraint % 2 == 0
        assert num_constraint < num_change
        constraint_indices = random.sample(range(0, num_change), num_constraint)
        values[constraint_indices[:num_constraint//2]] = ub
        values[constraint_indices[num_constraint//2:]] = lb        
    repeat = _sample_repeats(num_change, num_steps,
                             mean_change, sigma_change)
    return np.repeat(values, repeat, axis=0)

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

def get_train_val_data(*, tthrow, Np, xuyscales, data_list,
                          unmeasGbx0_list):
    """ Get the data for training and validation in 
        appropriate format for training. 
        All data are already scaled.
    """
    
    # Get scaling parameters.
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']
    Nx, Ny, Nu = len(xmean), len(ymean), len(umean)

    # Lists to store data.
    inuputs, outputs = [], []
    xseq = []
    y0, z0 = [], []

    # Loop through the data list.
    for data in data_list:
        
        # Scale data.
        x = (data.x - xmean)/xstd
        u = (data.u - umean)/ustd
        y = (data.y - ymean)/ystd

        # Get the input and output trajectory.
        u_traj = u[tthrow:, :][np.newaxis, ...]
        y_traj = y[tthrow:, :][np.newaxis, ...]
        x_traj = x[tthrow:, :][np.newaxis, ...]

        # Get initial states.
        yp0seq = y[tthrow-Np:tthrow, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[tthrow-Np:tthrow, :].reshape(Np*Nu, )[np.newaxis, :]
        y0_traj = y[tthrow, np.newaxis, :]
        z0_traj = np.concatenate((yp0seq, up0seq), axis=-1)

        # Get z_traj.
        # Nt = u.shape[0]
        # z_traj = []
        # for t in range(tthrow, Nt):
        #    ypseq = y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
        #    upseq = u[t-Np:t, :].reshape(Np*Nu, )[np.newaxis, :]
        #    z_traj += [np.concatenate((ypseq, upseq), axis=-1)]
        # z_traj = np.concatenate(z_traj, axis=0)[np.newaxis, ...]
        # yz_traj = np.concatenate((y_traj, z_traj), axis=-1)

        # Collect the trajectories in list.
        inputs += [u_traj]
        xseq += [x_traj]
        y0 += [y0_traj]
        z0 += [z0_traj]
        outputs += [y_traj]
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      xseq=np.concatenate(xseq[:-2], axis=0),
                      y0=np.concatenate(y0[:-2], axis=0),
                      z0=np.concatenate(z0[:-2], axis=0),   
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], xseq=xseq[-2], y0=y0[-2],
                         z0=z0[-2], outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], xseq=xseq[-1], y0=y0[-1],
                    z0=z0[-1], outputs=outputs[-1])
    
    # Return.
    return (train_data, trainval_data, val_data)

# def get_ss_train_val_data(*, xuyscales, training_data, 
#                              datasize_fracs, Nt=2):
#     """ Get steady state-data for training and validation in 
#         appropriate format after scaling. """

#     # Get scaling pars.
#     umean, ustd = xuyscales['uscale']
#     ymean, ystd = xuyscales['yscale']
#     Ny, Nu = len(ymean), len(umean)

#     # Lists to store data.
#     inputs, x0, outputs = [], [], []
    
#     # Scale data.
#     u = (training_data.u - umean)/ustd
#     y = (training_data.y - ymean)/ystd
    
#     # Get the input and output trajectory.
#     x0 = y
#     u = np.repeat(u[:, np.newaxis, :], Nt, axis=1)
#     y = np.repeat(y[:, np.newaxis, :], Nt, axis=1)

#     # Now split the data.
#     train_frac, trainval_frac, val_frac = datasize_fracs
#     Ndata = u.shape[0]
#     Ntrain = int(Ndata*train_frac)
#     Ntrainval = int(Ndata*trainval_frac)
#     Nval = int(Ndata*val_frac)

#     # Get the three types of data.
#     u = np.split(u, [Ntrain, Ntrain + Ntrainval, ], axis=0)
#     y = np.split(y, [Ntrain, Ntrain + Ntrainval, ], axis=0)
#     x0 = np.split(x0, [Ntrain, Ntrain + Ntrainval, ], axis=0)

#     # Get dictionaries of data.
#     train_data = dict(inputs=u[0], x0=x0[0], outputs=y[0])
#     trainval_data = dict(inputs=u[1], x0=x0[1], outputs=y[1])
#     val_data = dict(inputs=u[2], x0=x0[2], outputs=y[2])
    
#     # Return.
#     return train_data, trainval_data, val_data

def get_rectified_xs(*, ode, parameters):
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
    for _ in range(360):
        xs = model.sim(xs, us, ps)

    # Return.
    return xs

# def genPlantSsdata(*, fxu, hx, parameters, Ndata,
#                       xguess=None, seed=10):
#     """ Function to generate steady state data for the plant model. 
#         fxu: plant model in continous time.
#     """

#     # Set numpy seed.
#     np.random.seed(seed)

#     # Convert plant model to discrete time. 
#     Delta = parameters['Delta']
#     fxu_dt = c2dNonlin(fxu, Delta)

#     # Get a list of random inputs.
#     Ny, Nu = parameters['Ny'], parameters['Nu']
#     ulb, uub = parameters['ulb'], parameters['uub']
#     us_list = list((uub-ulb)*np.random.rand(Ndata, Nu) + ulb)

#     # Get a list to store the steady state costs.
#     xs_list, ys_list = [], []
    
#     # Loop over all the generated us.
#     for us in us_list:

#         # Solve the steady state equation.
#         xs, ys, _ = getXsYsSscost(fxu=fxu_dt, hx=hx, lyu=None, 
#                                   us=us, parameters=parameters, 
#                                   xguess=xguess)
#         xs_list += [xs]
#         ys_list += [ys]

#     # Get arrays to return the generated data.
#     u = np.array(us_list)
#     x = np.array(xs_list)
#     y = np.array(ys_list)

#     # Add measurement noise to y.
#     Rv = parameters['Rv']
#     ynoise_std = np.sqrt(np.diag(Rv))
#     y += ynoise_std*np.random.randn(Ndata, Ny)

#     # Create a dictionary.
#     ss_data = SimData(t=None, u=u, x=x, y=y)

#     # Return.
#     return ss_data