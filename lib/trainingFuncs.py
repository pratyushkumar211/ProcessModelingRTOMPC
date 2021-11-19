# [depends] linNonlinMPC.py
import sys
import random
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import collections
import pickle
import plottools
import time

# Custom class to store datasets.
SimData = collections.namedtuple('SimData',
                                ['t', 'x', 'u', 'y', 'p'])

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