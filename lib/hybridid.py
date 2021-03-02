# Pratyush Kumar, pratyushkumar@ucsb.edu

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

FIGURE_SIZE_A4 = (9, 10)
PRESENTATION_FIGSIZE = (6, 6)
PAPER_FIGSIZE = (6, 6)

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

def c2d(A, B, sample_time):
    """ Custom c2d function for linear systems."""
    
    # First construct the incumbent matrix
    # to take the exponential.
    (Nx, Nu) = B.shape
    M1 = np.concatenate((A, B), axis=1)
    M2 = np.zeros((Nu, Nx+Nu))
    M = np.concatenate((M1, M2), axis=0)
    Mexp = scipy.linalg.expm(M*sample_time)

    # Return the extracted matrices.
    Ad = Mexp[:Nx, :Nx]
    Bd = Mexp[:Nx, -Nu:]
    return (Ad, Bd)

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

def c2dNonlin(fxu, Delta):
    """ Quick function to 
        convert a ode to discrete
        time using the RK4 method.
        
        fxu is a function such that 
        dx/dt = f(x, u)
        assume zero-order hold on the input.
    """
    # Get k1, k2, k3, k4.
    k1 = fxu
    k2 = lambda x, u: fxu(x + Delta*(k1(x, u)/2), u)
    k3 = lambda x, u: fxu(x + Delta*(k2(x, u)/2), u)
    k4 = lambda x, u: fxu(x + Delta*k3(x, u), u)
    # Final discrete time function.
    xplus = lambda x, u: x + (Delta/6)*(k1(x, u) + 
                                        2*k2(x, u) + 2*k3(x, u) + k4(x, u))
    return xplus

def get_plotting_arrays(simdata, plot_range):
    """ Get data and return for plotting. """
    start, end = plot_range
    u = simdata.u[start:end, :]
    x = simdata.x[start:end, :]
    y = simdata.y[start:end, :]
    t = simdata.t[start:end]/60 # Convert to hours.
    # Return t, x, y, u.
    return (t, x, y, u)

def get_plotting_array_list(*, simdata_list, plot_range):
    """ Get all data as lists. """
    ulist, xlist, ylist = [], [], []
    for simdata in simdata_list:
        t, x, y, u = get_plotting_arrays(simdata, plot_range)
        ulist += [u]
        xlist += [x]
        ylist += [y]
    # Return lists.
    return (t, ulist, ylist, xlist)

def _resample_fast(*, x, xDelta, newDelta, resample_type):
    """ Resample with either first of zero-order hold. """
    Delta_ratio = int(xDelta/newDelta)
    if resample_type == 'zoh':
        return np.repeat(x, Delta_ratio, axis=0)
    else:
        x = np.concatenate((x, x[-1, np.newaxis, :]), axis=0)
        return np.concatenate([np.linspace(x[t, :], x[t+1, :], Delta_ratio)
                               for t in range(x.shape[0]-1)], axis=0)

def interpolate_yseq(yseq, Npast, Ny):
    """ y is of dimension: (None, (Npast+1)*p)
        Return y of dimension: (None, Npast*p). """
    yseq_interp = []
    for t in range(Npast):
        yseq_interp.append(0.5*(yseq[t*Ny:(t+1)*Ny] + yseq[(t+1)*Ny:(t+2)*Ny]))
    # Return.
    return np.concatenate(yseq_interp)

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

def plot_profit_curve(*, us, costs, colors, legends, 
                         figure_size=PAPER_FIGSIZE,
                         ylabel_xcoordinate=-0.12, 
                         left_label_frac=0.15):
    """ Plot the profit curves. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size, 
                                    gridspec_kw=dict(left=left_label_frac))
    xlabel = r'$C_{Af} \ (\textnormal{mol/m}^3)$'
    ylabel = r'Cost ($\$ $)'
    for (cost, color) in zip(costs, colors):
        # Plot the corresponding data.
        axes.plot(us, cost, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=False)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    axes.set_xlim([np.min(us), np.max(us)])
    #figlabel = r'$\ell(y, u), \ \textnormal{subject to} \ f(x, u)=0, y=h(x)$'
    #figure.suptitle(figlabel,
    #                x=0.55, y=0.94)
    # Return the figure object.
    return [figure]

def plot_avg_profits(*, t, avg_stage_costs,
                    legend_colors, legend_names, 
                    figure_size=PAPER_FIGSIZE, 
                    ylabel_xcoordinate=-0.15):
    """ Plot the profit. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                                  sharex=True,
                                  figsize=figure_size,
                                  gridspec_kw=dict(left=0.15))
    xlabel = 'Time (hr)'
    ylabel = '$\Lambda_k$'
    for (cost, color) in zip(avg_stage_costs, legend_colors):
        # Plot the corresponding data.
        profit = -cost
        axes.plot(t, profit, color)
    axes.legend(legend_names)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=True)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    axes.set_xlim([np.min(t), np.max(t)])
    # Return.
    return [figure]

def plot_val_metrics(*, num_samples, val_metrics, colors, legends, 
                     figure_size=PAPER_FIGSIZE,
                     ylabel_xcoordinate=-0.11, 
                     left_label_frac=0.15):
    """ Plot validation metric on open loop data. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                  sharex=True, 
                                  figsize=figure_size, 
                                  gridspec_kw=dict(left=left_label_frac))
    xlabel = 'Hours of training samples'
    ylabel = 'MSE'
    num_samples = num_samples/60
    for (val_metric, color) in zip(val_metrics, colors):
        # Plot the corresponding data.
        axes.semilogy(num_samples, val_metric, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    axes.set_xlim([np.min(num_samples), np.max(num_samples)])
    figure.suptitle('Mean squared error (MSE) - Validation data', 
                    x=0.52, y=0.92)
   # Return the figure object.
    return [figure]

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

def fnn_koopman(yz, fnn_weights):
    """ Compute the NN output. """
    nn_output = yz
    for i in range(0, len(fnn_weights)-2, 2):
        (W, b) = fnn_weights[i:i+2]
        nn_output = W.T @ nn_output + b[:, np.newaxis]
        nn_output = np.tanh(nn_output)
    (Wf, bf) = fnn_weights[-2:]
    xkp = (Wf.T @ nn_output + bf[:, np.newaxis])
    # Return.
    return xkp

def koopman_func(xkp, u, parameters):
    """ The Koopman Operator model function. """

    # Fnn weights.
    A = parameters['A']
    B = parameters['B']
    
    # Get scaling.
    xuyscales = parameters['xuyscales']
    umean, ustd = xuyscales['uscale']

    # Scale inputs/state.
    u = (u - umean)/ustd

    # The Deep Koopman linear model.
    xkpplus = A @ xkp + B @ u

    # Return the sum.
    return xkpplus

def get_koopman_pars_check_func(*, parameters, training_data, train):
    """ Get the Koopman operator model parameters. """

    # Get weights.
    trained_weights = train['trained_weights'][-1]
    fnn_weights = trained_weights[:-3]
    A = trained_weights[-3].T
    B = trained_weights[-2].T
    H = trained_weights[-1].T

    # Get sizes.
    Np = train['Np']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    Nx = train['fnn_dims'][-1]

    # Get scaling.
    xuyscales = train['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
                             np.tile(umean, (Np, ))))[:, np.newaxis]
    yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
                             np.tile(ustd, (Np, ))))[:, np.newaxis]

    # Get steady state in the lifted space.
    yindices = parameters['yindices']
    us = parameters['us']
    ys = parameters['xs'][yindices]
    yzs = np.concatenate((np.tile(ys, (Np+1, )), 
                          np.tile(us, (Np, ))))[:, np.newaxis]
    breakpoint()
    yzs = (yzs - yzmean)/yzstd
    xkps = fnn_koopman(yzs, fnn_weights)[:, 0]

    # Constraints for MPC.
    ulb, uub = parameters['ulb'], parameters['uub']

    # Make the parameter dictionary.
    koopman_pars  = dict(Np=Np, Nx=Nx, Nu=Nu, Ny=Ny,
                         A=A, B=B, H=H, xs=xkps, us=us,
                         ulb=ulb, uub=uub, xuyscales=xuyscales)

    # Get the control input profile for the simulation.
    training_data = training_data[-1]
    y = training_data.y
    u = training_data.u
    ts = parameters['tsteps_steady']
    uval = u[ts:, :]

    # Get initial state for the simulation.
    yp0seq = y[ts-Np:ts, :].reshape(Np*Ny, )[:, np.newaxis]
    up0seq = u[ts-Np:ts:, ].reshape(Np*Nu, )[:, np.newaxis]
    y0 = y[ts, :, np.newaxis]
    yz0 = np.concatenate((y0, yp0seq, up0seq))
    yz0 = (yz0 - yzmean)/yzstd
    xkp0 = fnn_koopman(yz0, fnn_weights)[:, 0]

    # Get the functions.
    koopman_fxu = lambda x, u: koopman_func(x, u, koopman_pars)
    koopman_hx = lambda x: (H @ x)[:Ny]*ystd + ymean

    # Run the simulation.
    yzval, yval = quick_sim(koopman_fxu, koopman_hx, xkp0, uval)

    # To compare with predictions made by the tensorflow model.
    ytfval = train['val_predictions'][-1].y

    breakpoint()
    # Just return the hybrid parameters.
    return koopman_pars

def get_train_val_data(*, Np, xuyscales, parameters, data_list):
    """ Get the data for training in appropriate format. """

    # Get some parameters.
    ts = parameters['tsteps_steady']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']

    # Lists to store data.
    inputs, yz0, yz, outputs = [], [], [], []

    # Loop through the data list.
    for data in data_list:
        
        # Scale data.
        u = (data.u - umean)/ustd
        y = (data.y - ymean)/ystd
                
        # Get input trajectory.
        u_traj = u[ts:][np.newaxis, :]
        
        # Get initial states.
        y0 = y[ts, :][np.newaxis, :]
        yp0seq = y[ts-Np:ts, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[ts-Np:ts, :].reshape(Np*Nu, )[np.newaxis, :]
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
        yz0.append(yz0_traj)
        yz.append(yz_traj)
        outputs.append(y_traj)
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      yz0=np.concatenate(yz0[:-2], axis=0),
                      yz=np.concatenate(yz[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], yz0=yz0[-2], yz=yz[-2], 
                          outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], yz0=yz0[-1], 
                    yz=yz[-1], outputs=outputs[-1])
    # Return.
    return (train_data, trainval_data, val_data)