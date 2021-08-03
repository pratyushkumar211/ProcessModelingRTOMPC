# [depends] %LIB%/hybridid.py tworeac_parameters_nonlin.pickle
# [depends] tworeac_train_nonlin.pickle
# [makes] pickle
""" Script to use the grey-box models to evaluate 
    different disturbance models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
sys.path.append('../')
import numpy as np
import casadi
import matplotlib.pyplot as plt
import mpctools as mpc
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE, c2d,
                      plot_profit_curve, KalmanFilter,
                      get_augmented_matrices_for_filter)
from tworeac_parameters_nonlin import _get_tworeac_model

def _get_linear_greybox_model(*, parameters):
    """ Check the observability of the original linear system
        and compute the matrix required to predict the correct 
        grey-box state evolution. """
    # Measurement matrix for the plant.
    sample_time = parameters['sample_time']
    tau = parameters['ps'].squeeze()
    Ng = parameters['Ng']
    C = np.eye(Ng)
    # Get the continuous time A/B matrices.
    k1 = parameters['k1']
    A = np.array([[-k1-(1/tau), 0.], 
                  [k1, -(1/tau)]])
    B = np.array([[1/tau], [0.]]) 
    (Ad, Bd) = c2d(A, B, sample_time)
    return (Ad, Bd, C)

def _get_yps_ygs_dists_cost_estimates(*, us, dist_model, parameters):
    """ Do a closed-loop simulation and get disturbance estimates. """
    # Construct and return the plant.
    (Bd, Cd) = dist_model
    ps = parameters['ps']
    (Nx, Ng, Nu, Ny) = (parameters['Nx'], parameters['Ng'], 
                        parameters['Nu'], parameters['Ny'])
    (Nsim, Nd) = (120, Ny)

    # Get the grey-box/dist model and Kalman filter.
    (Bd, Cd) = dist_model
    (A, B, C) = _get_linear_greybox_model(parameters=parameters)
    (Qwx, Qwd, Rv) = (1e-6*np.eye(Ng), 1e-3*np.eye(Nd), 1e-8*np.eye(Ny))
    (Aaug, Baug, 
     Caug, Qwaug) = get_augmented_matrices_for_filter(A, B, C, Bd, 
                                                      Cd, Qwx, Qwd)
    plant = _get_tworeac_model(parameters=parameters)
    xprior = np.concatenate((plant.x[0][:Ng, :], np.zeros((Nd, 1))), axis=0)
    kf_filter = KalmanFilter(A=Aaug, B=Baug, C=Caug, 
                             Qw=Qwaug, Rv=Rv, xprior=xprior)
    kf_filter.solve(plant.y[0], us)
    for _ in range(Nsim):
        yps = plant.step(us, ps)
        kf_filter.solve(yps, us)
    (ygs, dists) = np.split(kf_filter.xhat[-1], [Ng,], axis=0)
    # Return.
    return (yps, ygs, dists)

def _plot_yps_ygs_dists(*, yps, ygs, dists, us, 
                         figure_size=PAPER_FIGSIZE,
                         ylabel_xcoordinate=-0.33, 
                         left_label_frac=0.15, wspace=0.5):
    """ Plot the profit curves. """
    (nrows, ncols) = (2, 2)
    (figure, axes) = plt.subplots(nrows=nrows, ncols=ncols, 
                                  sharex=True, 
                                  figsize=figure_size, 
                                  gridspec_kw=dict(left=left_label_frac, 
                                                   wspace=wspace))
    xlabel = r'$C_{Af} \ (\textnormal{mol/m}^3)$'
    left_ylabels = [r'$C_A$', r'$C_B$']
    right_ylabels = [r'$\hat{d}_{1s}$', r'$\hat{d}_{2s}$']
    for (row, col) in itertools.product(range(nrows), range(ncols)):
        if col == 0:
            plant_handle = axes[row, col].plot(us, yps[0][:, row], 'b--')
            axes[row, col].plot(us, yps[1][:, row], 'b--')
            outputdm_handle = axes[row, col].plot(us, ygs[0][:, row], 'g')
            inputdm_handle = axes[row, col].plot(us, ygs[1][:, row], 'tomato')
            axes[row, col].set_ylabel(left_ylabels[row], rotation=False)
        else:
            axes[row, col].plot(us, dists[0][:, row], 'g')
            axes[row, col].plot(us, dists[1][:, row], 'tomato')
            axes[row, col].set_ylabel(right_ylabels[row], rotation=False)
        if row == 1:
            axes[row, col].set_xlabel(xlabel)
        axes[row, col].set_xlim([np.min(us), np.max(us)])
        axes[row, col].get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5)
    handles = plant_handle + outputdm_handle + inputdm_handle
    figure.legend(handles=handles, 
                  labels=['Plant', 'Output DM', 'Input DM'], 
                  ncol=3, loc=(0.16, 0.92)) 
    # Return the figure object.
    return [figure]

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_nonlin.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']  
    (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
    Ny = parameters['Ny']  
    dist_models = [(0*np.eye(Ny), np.eye(Ny)), (np.eye(Ny), 0*np.eye(Ny))]
    uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
    (yps, ygs, dists) = ([], [], [])
    for dist_model in dist_models:
        (model_yp, model_yg, model_dist) = ([], [], [])
        for us in uss:
            (yp, yg, dist) = _get_yps_ygs_dists_cost_estimates(us=us, 
                                              dist_model=dist_model, 
                                              parameters=parameters)
            model_yp.append(yp)
            model_yg.append(yg)
            model_dist.append(dist)
        yps.append(np.asarray(model_yp).squeeze())
        ygs.append(np.asarray(model_yg).squeeze())
        dists.append(np.asarray(model_dist).squeeze())
    us = uss.squeeze()
    # Figures.
    figures = _plot_yps_ygs_dists(yps=yps, ygs=ygs, dists=dists, us=us)
    with PdfPages('tworeac_distmodels_nonlin.pdf', 
                  'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()