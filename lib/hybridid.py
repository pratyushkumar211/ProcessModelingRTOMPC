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
PAPER_FIGSIZE = (5, 6)

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

def _get_energy_price(*, num_days, sample_time):
    """ Get a two day heat disturbance profile. """
    energy_price = np.zeros((24, 1))
    energy_price[0:8, :] = np.ones((8, 1))
    energy_price[8:16, :] = 70*np.ones((8, 1))
    energy_price[16:24, :] = np.ones((8, 1))
    energy_price = 1e-2*np.tile(energy_price, (num_days, 1))
    return _resample_fast(x=energy_price,
                          xDelta=60,
                          newDelta=sample_time,
                          resample_type='zoh')

def get_cstr_flash_empc_pars(*, num_days, sample_time, plant_pars):
    """ Get the parameters for Empc and RTO simulations. """

    # Get the cost parameters.
    energy_price = _get_energy_price(num_days=num_days, sample_time=sample_time)
    raw_mat_price = _resample_fast(x = np.array([[1000.], [1000.], 
                                                 [1000.], [950.], 
                                                 [950.], [950.], 
                                                 [950.], [950.]]), 
                                   xDelta=6*60,
                                   newDelta=sample_time,
                                   resample_type='zoh')
    product_price = _resample_fast(x = np.array([[7000.], [8000.], 
                                                 [7000.], [6000.], 
                                                 [6000.], [6000.], 
                                                 [6000.], [6000.]]),
                                   xDelta=6*60,
                                   newDelta=sample_time,
                                   resample_type='zoh')
    cost_pars = np.concatenate((energy_price,
                                raw_mat_price, product_price), axis=1)
    
    # Get the plant disturbances.
    ps = plant_pars['ps'][np.newaxis, :]
    disturbances = np.repeat(ps, 24*60, axis=0)

    # Return as a concatenated vector.
    return cost_pars, disturbances

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

def get_tworeac_train_val_data(*, Np, parameters, data_list):
    """ Get the data for training in appropriate format. """
    tsteps_steady = parameters['tsteps_steady']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    xuyscales = get_scaling(data=data_list[0])
    xuscales = dict(xscale=xuyscales['yscale'], uscale=xuyscales['uscale'])
    inputs, xGz0, outputs = [], [], []
    # Loop through the data list.
    for data in data_list:

        # Scale data.
        u = (data.u-xuscales['uscale'][0])/xuscales['uscale'][1]
        y = (data.y-xuscales['xscale'][0])/xuscales['xscale'][1]

        # Starting time point.
        t = tsteps_steady
        
        # Get input trajectory.
        u_traj = u[t:][np.newaxis, :]
        
        # Get initial state.
        xG0 = y[t, :][np.newaxis, :]
        yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[t-Np:t, :].reshape(Np*Nu, )[np.newaxis, :]
        xGz0_traj = np.concatenate((xG0, yp0seq, up0seq), axis=-1)
        
        # Get output trajectory.
        y_traj = y[t:, :][np.newaxis, ...]

        # Collect the trajectories in list.
        inputs.append(u_traj)
        xGz0.append(xGz0_traj)
        outputs.append(y_traj)
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      xGz0=np.concatenate(xGz0[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], xGz0=xGz0[-2],
                         outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], xGz0=xGz0[-1],
                    outputs=outputs[-1])
    # Return.
    return (train_data, trainval_data, val_data, xuscales)

def get_cstr_flash_train_val_data(*, Np, parameters,
                                     greybox_processed_data):
    """ Get the data for training in appropriate format. """
    tsteps_steady = parameters['tsteps_steady']
    (Ng, Ny, Nu) = (parameters['Ng'], parameters['Ny'], parameters['Nu'])
    xuyscales = get_scaling(data=greybox_processed_data[0])
    inputs, xGz0, yz0, outputs, xG = [], [], [], [], []
    for data in greybox_processed_data:
        
        # Scale data.
        u = (data.u-xuyscales['uscale'][0])/xuyscales['uscale'][1]
        y = (data.y-xuyscales['yscale'][0])/xuyscales['yscale'][1]
        x = (data.x-xuyscales['xscale'][0])/xuyscales['xscale'][1]

        t = tsteps_steady
        # Get input trajectory.
        u_traj = u[t:, :][np.newaxis, ...]

        # Get initial state.
        yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[t-Np:t:, ].reshape(Np*Nu, )[np.newaxis, :]
        z0 = np.concatenate((yp0seq, up0seq), axis=-1)
        xG0 = x[t, :][np.newaxis, :]
        y0 = y[t, :][np.newaxis, :]
        xGz0_traj = np.concatenate((xG0, z0), axis=-1)
        yz0_traj = np.concatenate((y0, z0), axis=1)

        # Get grey-box state trajectory.
        xG_traj = x[t:, :][np.newaxis, :]

        # Get output trajectory.
        y_traj = y[t:, :][np.newaxis, ...]

        # Collect the trajectories in list.
        inputs.append(u_traj)
        xGz0.append(xGz0_traj)
        yz0.append(yz0_traj)
        outputs.append(y_traj)
        xG.append(xG_traj)

    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      xGz0=np.concatenate(xGz0[:-2], axis=0),
                      yz0=np.concatenate(yz0[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0), 
                      xG=np.concatenate(xG[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], xGz0=xGz0[-2],
                         yz0=yz0[-2], outputs=outputs[-2], xG=xG[-2])
    val_data = dict(inputs=inputs[-1], xGz0=xGz0[-1],
                    yz0=yz0[-1], outputs=outputs[-1], xG=xG[-1])
    # Return.
    return (train_data, trainval_data, val_data, xuyscales)

def _cstr_flash_plant_ode(x, u, p, parameters):
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

    # Extract the plant states into meaningful names.
    (Hr, CAr, CBr, CCr, Tr) = x[0:5]
    (Hb, CAb, CBb, CCb, Tb) = x[5:10]
    (F, Qr, D, Qb) = u[0:4]
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

def _cstr_flash_greybox_ode(x, u, p, parameters):
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

    # Extract the plant states into meaningful names.
    (Hr, CAr, CBr, Tr) = x[0:4]
    (Hb, CAb, CBb, Tb) = x[4:8]
    (F, Qr, D, Qb) = u[0:4]
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

def _cstr_flash_measurement(x, parameters):
    yindices = parameters['yindices']
    # Return the measurement.
    return x[yindices]

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