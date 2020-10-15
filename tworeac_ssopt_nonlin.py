# [depends] %LIB%/hybridid.py tworeac_parameters_nonlin.pickle
# [makes] pickle
""" Script to use the trained hybrid model for 
    steady-state optimization.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import casadi
import matplotlib.pyplot as plt
import mpctools as mpc
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE, plot_profit_curve)
from tworeac_parameters_nonlin import (_tworeac_plant_ode, 
                                       _tworeac_greybox_ode)

def _plant_ys(*, us, parameters):
    """ Return the plant xs. 
        Solve: f(x, u) = 0 (continuous time)
    """
    # Construct and return the plant.
    ps = parameters['ps']
    Nx, Ny = parameters['Nx'], parameters['Ny']
    plant = lambda x: _tworeac_plant_ode(x, us, ps, parameters)
    xs = casadi.SX.sym('xs', Nx)
    plant = mpc.getCasadiFunc(plant, [Nx], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', plant)
    xguess = 0.5*np.ones((Nx, 1))
    xs = np.asarray(rootfinder(xguess))
    # Return.
    return xs[0:Ny]

def _greybox_ys(*, us, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu = 0 (continuous time SS) 
    """
    # Construct and return the plant.
    ps = parameters['ps']
    (Nx, Ny) = (parameters['Ng'], parameters['Ny'])
    greybox = lambda x: _tworeac_greybox_ode(x, us, ps, parameters)
    xs = casadi.SX.sym('xs', Nx)
    greybox = mpc.getCasadiFunc(greybox, [Nx], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', greybox)
    xguess = 0.5*np.ones((Nx, 1))
    xs = np.asarray(rootfinder(xguess))
    # Return.
    return xs[0:Ny]

def cost(ys, us):
    """ Compute the steady-state cost. """
    (cost, profit) = (105, 160)
    # Return.
    return cost*us - profit*ys[:, -1:]

def compute_cost_curves(*, parameters):
    """ Compute the profit curves for the three models. """
    (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
    uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
    (ys_plant, ys_greybox) = ([], [])
    for us in uss:
        ys_plant.append(_plant_ys(us=us, parameters=parameters))
        ys_greybox.append(_greybox_ys(us=us, 
                                      parameters=parameters))
    ys_plant = np.asarray(ys_plant).squeeze()
    ys_greybox = np.asarray(ys_greybox).squeeze()
    us = uss.squeeze(axis=-1)
    (cost_plant, cost_greybox) = (cost(ys_plant, us), cost(ys_greybox, us))
    # Return the compiled model.
    return (cost_plant, cost_greybox, us)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_nonlin.pickle',
                                         type='read')
    # Create the hybrid model.
    #Np = 3
    #fnn_dims = [8, 2]
    (cost_plant, cost_greybox, us) = compute_cost_curves(parameters=
                             tworeac_parameters['parameters'])
    figures = plot_profit_curve(us=us, costs=[cost_plant, cost_greybox])
    # Save data.
    with PdfPages('tworeac_ssopt_nonlin.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()