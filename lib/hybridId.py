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

# Custom class to store datasets.
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

def get_train_val_data(*, tthrow, Np, xuyscales, data_list):
    """ Get the data for training and validation in 
        appropriate format after scaling. """

    # Get scaling pars.
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']
    Ny, Nu = len(ymean), len(umean)

    # Lists to store data.
    inputs, yz0, yz, outputs = [], [], [], [], []

    # Loop through the data list.
    for data in data_list:
        
        # Scale data.
        u = (data.u - umean)/ustd
        y = (data.y - ymean)/ystd
        
        # Get the input and output trajectory.
        u_traj = u[tthrow:, :][np.newaxis, ...]
        y_traj = y[tthrow:, :][np.newaxis, ...]

        # Get initial states.
        yp0seq = y[tthrow-Np:tthrow, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[tthrow-Np:tthrow, :].reshape(Np*Nu, )[np.newaxis, :]
        y0 = y[tthrow, np.newaxis, :]
        yz0_traj = np.concatenate((y0, yp0seq, up0seq), axis=-1)

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
        outputs += [y_traj]
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      yz0=np.concatenate(yz0[:-2], axis=0),
                      yz=np.concatenate(yz[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], yz0=yz0[-2],
                          yz=yz[-2], outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], yz0=yz0[-1],
                    yz=yz[-1], outputs=outputs[-1])
    
    # Return.
    return (train_data, trainval_data, val_data)

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

def generatePlantSSdata(*, fxu, hx, cost_yu, parameters, Ndata, 
                          xguess=None, seed=10):
    """ Function to generate data to train the ICNN. """

    # Set numpy seed.
    np.random.seed(seed)

    # Get a list of random inputs.
    Nu = parameters['Nu']
    ulb, uub = parameters['ulb'], parameters['uub']
    us_list = list((uub-ulb)*np.random.rand(Ndata, Nu) + ulb)

    # Get a list to store the steady state costs.
    ss_costs = []

    # Loop over all the generated us.
    for us in us_list:

        # Solve the steady state equation.
        _, ss_cost = get_xs_sscost(fxu=fxu, hx=hx, lyu=cost_yu, 
                                   us=us, parameters=parameters, 
                                   xguess=xguess)
        ss_costs += [ss_cost]

    # Get arrays to return the generated data.
    u = np.array(us_list)
    lyu = np.array(ss_costs)

    # Return.
    return u, lyu