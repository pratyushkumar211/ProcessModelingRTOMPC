# [depends] %LIB%/hybridid.py tworeac_parameters_lin.pickle
# [makes] pickle
""" Script to use the trained hybrid model for 
    steady-state optimization.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE, plot_profit_curve)

def _plant_ys(*, us, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu = 0 (continuous time SS) 
    """
    # Get the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    Ny = parameters['Ny']
    tau = parameters['ps'].squeeze()
    # Get the matrices and xs. 
    A = np.array([[-k1-(1/tau), 0., 0.], 
                  [k1, -k2 - (1/tau), k3], 
                  [0., k2, -k3 -(1/tau)]])
    B = np.array([[1/tau], [0.], [0.]]) 
    xs = -np.linalg.inv(A) @ (B @ us)
    # Return.
    return xs[0:Ny]

def _greybox_ys(*, us, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu = 0 (continuous time SS) 
    """
    # Get the parameters.
    k1 = parameters['k1']
    Ny = parameters['Ny']
    tau = parameters['ps'].squeeze()
    # Get the matrices and xs. 
    A = np.array([[-k1-(1/tau), 0.], 
                  [k1, -(1/tau)]])
    B = np.array([[1/tau], [0.]]) 
    xs = -np.linalg.inv(A) @ (B @ us)
    # Return.
    return xs[0:Ny]

def _hybrid_ys(*, us, Np, fnn_weight, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu = 0 (continuous time SS) 
    """
    # Get the parameters.
    k1 = parameters['k1']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    tau = parameters['ps'].squeeze()
    # Get the matrices and xs. 
    A = np.array([[-k1-(1/tau), 0.], 
                  [k1, -(1/tau)]])
    B = np.array([[1/tau], [0.]])
    (Apast, Bpast) = np.split(fnn_weight.T, [3*Ny, ], axis=1) 
    # Correct the original A and B.
    for i in range(Np-1):
        A += Apast[:, i*Ny:(i+1)*Ny]
        B += Bpast[:, i*Nu:(i+1)*Nu]
    A += Apast[:, (Np-1)*Ny:]
    xs = -np.linalg.inv(A) @ (B @ us)
    # Return.
    return xs[0:Ny]

def cost(ys, us):
    """ Compute the steady-state cost. """
    (cost, profit) = (100, 150)
    # Return.
    return cost*us - profit*ys[:, -1:]

def compute_cost_curves(*, Np, fnn_weight, parameters):
    """ Compute the profit curves for the three models. """
    (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
    uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
    (ys_plant, ys_greybox, ys_hybrid) = ([], [], [])
    for us in uss:
        ys_plant.append(_plant_ys(us=us, parameters=parameters))
        ys_greybox.append(_greybox_ys(us=us, 
                                      parameters=parameters))
        ys_hybrid.append(_hybrid_ys(us=us, Np=Np, fnn_weight=fnn_weight,
                                    parameters=parameters))
    ys_plant = np.asarray(ys_plant).squeeze()
    ys_greybox = np.asarray(ys_greybox).squeeze()
    ys_hybrid = np.asarray(ys_hybrid).squeeze()
    us = uss.squeeze(axis=-1)
    (cost_plant, 
     cost_greybox, cost_hybrid) = (cost(ys_plant, us), 
                                   cost(ys_greybox, us), 
                                   cost(ys_hybrid, us))
    # Return the compiled model.
    return (cost_plant, cost_greybox, cost_hybrid, us)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_lin.pickle',
                                         type='read')
    tworeac_train = PickleTool.load(filename='tworeac_train_lin.pickle',
                                    type='read')
    # Create the hybrid model.
    (Np, fnn_weight) = (tworeac_train['Np'], tworeac_train['fnn_weight'])
    (cost_plant, 
     cost_greybox, 
     cost_hybrid, us) = compute_cost_curves(Np=Np, fnn_weight=fnn_weight,
                                parameters=tworeac_parameters['parameters'])
    figures = plot_profit_curve(us=us,
                                costs=[cost_plant, cost_greybox, cost_hybrid])
    # Save data.
    with PdfPages('tworeac_ssopt_lin.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()