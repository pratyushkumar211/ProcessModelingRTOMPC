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
    greybox = mpc.getCasadiFunc(greybox, [Nx], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', greybox)
    xguess = 0.5*np.ones((Nx, 1))
    xs = np.asarray(rootfinder(xguess))
    # Return.
    return xs[0:Ny]

def _hybrid_ode(xs, us, ps, Np, fnn_weights, parameters):
    """ Feed forward evaluation. """
    xs_greybox = _tworeac_greybox_ode(xs, us, ps, parameters)
    xs = xs[:, np.newaxis]
    nn_input = np.concatenate((np.tile(xs, (Np, 1)), 
                               np.tile(us, (Np-1, 1))), axis=0)
    for i in range(0, len(fnn_weights)-1, 2):
        (W, b) = fnn_weights[i:i+2]
        nn_input = np.tanh(W.T @ nn_input + b[:, np.newaxis])
    nn_output = fnn_weights[-1].T @ nn_input
    return xs_greybox + nn_output[:, 0]

def _hybrid_ys(*, us, Np, fnn_weights, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu + fnn() = 0 (continuous time SS) 
    """
    # Construct and return the plant.
    ps = parameters['ps']
    (Nx, Ny) = (parameters['Ng'], parameters['Ny'])
    hybrid = lambda xs: _hybrid_ode(xs, us, ps, Np, fnn_weights, parameters)
    hybrid = mpc.getCasadiFunc(hybrid, [Nx], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', hybrid)
    xguess = 0.5*np.ones((Nx, 1))
    xs = np.asarray(rootfinder(xguess))
    # Return.
    return xs[0:Ny]

def cost(ys, us):
    """ Compute the steady-state cost. """
    (cost, profit) = (105, 160)
    # Return.
    return cost*us - profit*ys[:, -1:]

def compute_cost_curves(*, Np, fnn_weights, parameters):
    """ Compute the profit curves for the three models. """
    (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
    uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
    (ys_plant, ys_greybox, ys_hybrid) = ([], [], [])
    for us in uss:
        ys_plant.append(_plant_ys(us=us, parameters=parameters))
        ys_greybox.append(_greybox_ys(us=us, 
                                      parameters=parameters))
        ys_hybrid.append(_hybrid_ys(us=us, Np=Np, 
                                    fnn_weights=fnn_weights, 
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

def solve_plant_hybrid_nlps(*, Np, fnn_weights, parameters):
    """ Solve steady state NLPs for plant and 
        hybrid models. """
    ps = parameters['ps']
    (ulb, uub) = (parameters['lb']['u'].squeeze(), 
                parameters['ub']['u'].squeeze())
    (Nx, Ng, Ny, Nu) = (parameters['Nx'], parameters['Ng'], 
                        parameters['Ny'], parameters['Nu'])

    # Get casadi funcs.
    plant = lambda x, u: _tworeac_plant_ode(x, u, ps, parameters)
    plant = mpc.getCasadiFunc(plant, [Nx, Nu], ['xs', 'us'])
    hybrid = lambda x, u: _hybrid_ode(x, u, ps, Np, fnn_weights, parameters)
    hybrid = mpc.getCasadiFunc(hybrid, [Ng, Nu], ['xs', 'us'])

    # Variables.
    xs = casadi.SX.sym('xs', Nx)
    us = casadi.SX.sym('us', Nu)

    # First construct the NLP for the plant.
    plant_nlp = dict(x=casadi.vertcat(xs, us), 
                     f=cost(xs.T[0:Ny], us), 
                     g=casadi.vertcat(plant(xs, us), us))
    plant_nlp = casadi.nlpsol('plant_nlp', 'ipopt', plant_nlp)
    xguess = 0.8*np.ones((Nx+Nu, 1))
    lbg = np.concatenate((np.zeros((Nx, 1)), np.array([[ulb]])), axis=0)
    ubg = np.concatenate((np.zeros((Nx, 1)), np.array([[uub]])), axis=0)
    plant_nlp_soln = plant_nlp(x0=xguess, lbg=lbg, ubg=ubg)
    (plant_cost, plant_us) = (plant_nlp_soln['f'].full().squeeze(axis=-1)[0], 
                               plant_nlp_soln['g'].full().squeeze()[-1])

    # First construct the NLP for the hybrid model.
    xs = casadi.SX.sym('xs', Ng)
    hybrid_nlp = dict(x=casadi.vertcat(xs, us), 
                      f=cost(xs.T[0:Ny], us), 
                      g=casadi.vertcat(hybrid(xs, us), us))
    hybrid_nlp = casadi.nlpsol('hybrid_nlp', 'ipopt', hybrid_nlp)
    xguess = 0.8*np.ones((Ng+Nu, 1))
    lbg = np.concatenate((np.zeros((Ng, 1)), np.array([[ulb]])), axis=0)
    ubg = np.concatenate((np.zeros((Ng, 1)), np.array([[uub]])), axis=0)
    hybrid_nlp_soln = hybrid_nlp(x0=xguess, lbg=lbg, ubg=ubg)
    (hybrid_cost, 
     hybrid_us) = (hybrid_nlp_soln['f'].full().squeeze(axis=-1)[0], 
                    hybrid_nlp_soln['g'].full().squeeze()[-1])

    print("Plant cost: " + str(plant_cost) + "Plant u: " + str(plant_us))
    print("Hybrid cost: " + str(hybrid_cost) + "Plant u: " + str(hybrid_us))

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_nonlin.pickle',
                                         type='read')    
    tworeac_train = PickleTool.load(filename='tworeac_train_nonlin.pickle',
                                    type='read')
    (Np, fnn_weights) = (tworeac_train['Np'], tworeac_train['fnn_weights'])

    # Create cost curves.
    (cost_plant, 
      cost_greybox, 
      cost_hybrid, us) = compute_cost_curves(Np=Np, fnn_weights=fnn_weights,
                                    parameters=tworeac_parameters['parameters'])
    figures = plot_profit_curve(us=us, 
                                costs=[cost_plant, cost_greybox, cost_hybrid])
    solve_plant_hybrid_nlps(Np=Np, fnn_weights=fnn_weights,
                            parameters=tworeac_parameters['parameters'])
    # Save data.
    with PdfPages('tworeac_ssopt_nonlin.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()