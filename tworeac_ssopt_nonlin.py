# [depends] %LIB%/hybridid.py tworeac_parameters_nonlin.pickle
# [depends] tworeac_train_nonlin.pickle
# [makes] pickle
""" Script to use the trained hybrid model for 
    steady-state optimization.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
sys.path.append('../')
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
    Ng = parameters['Ng']
    greybox = lambda x: _tworeac_greybox_ode(x, us, ps, parameters)
    greybox = mpc.getCasadiFunc(greybox, [Ng], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', greybox)
    xguess = 0.5*np.ones((Ng, 1))
    xs = np.asarray(rootfinder(xguess))
    # Return.
    return xs[0:Ng]

def _blackbox_ys(*, us, Np, fnn_weights, parameters):
    """ Return the plant xs. 
        Solve: -xs + fnn(xs, 2*xs, 2*us, us) = 0 (continuous time SS) 
    """
    def _fnn(xs, us, Np, fnn_weights):
        xs = xs[:, np.newaxis]
        nn_input = np.concatenate((np.tile(xs, (Np+1, 1)), 
                                   np.tile(us, (Np+1, 1))), axis=0)
        for i in range(0, len(fnn_weights)-1, 2):
            (W, b) = fnn_weights[i:i+2]
            nn_input = np.tanh(W.T @ nn_input + b[:, np.newaxis])
        nn_output = fnn_weights[-1].T @ nn_input
        return nn_output[:, 0]
    # Construct and return the plant.
    Ng = parameters['Ng']
    blackbox = lambda xs: -xs + _fnn(xs, us, Np, fnn_weights)
    blackbox = mpc.getCasadiFunc(blackbox, [Ng], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', blackbox)
    xguess = 0.5*np.ones((Ng, 1))
    xGsys = np.asarray(rootfinder(xguess))
    # Return.
    return xGsys[-Ng:]

def _residual_ys(*, us, Np, fnn_weights, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu + fnn() = 0 (continuous time SS) 
    """
    def _fg(xs, us, ps, parameters):
        """ Grey-box part. """ 
        Delta = parameters['sample_time']
        k1 = _tworeac_greybox_ode(xs, us, ps, parameters)
        k2 = _tworeac_greybox_ode(xs + Delta*(k1/2), us, ps, parameters)
        k3 = _tworeac_greybox_ode(xs + Delta*(k2/2), us, ps, parameters)
        k4 = _tworeac_greybox_ode(xs + Delta*k3, us, ps, parameters)
        return xs + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
    def _fnn(xs, us, Np, fnn_weights):
        """ Neural network part. """
        xs = xs[:, np.newaxis]
        nn_input = np.concatenate((np.tile(xs, (Np, 1)), 
                                   np.tile(us, (Np, 1))), axis=0)
        for i in range(0, len(fnn_weights)-1, 2):
            (W, b) = fnn_weights[i:i+2]
            nn_input = np.tanh(W.T @ nn_input + b[:, np.newaxis])
        nn_output = fnn_weights[-1].T @ nn_input
        return nn_output[:, 0]
    def _residual_func(xGsys, us, ps, Ng, parameters, Np, fnn_weights):
        (xGs, ys) = np.split(xGsys, [Ng, ])
        row1 = -xGs + _fg(xGs, us, ps, parameters)
        row2 = -ys + xGs + _fnn(ys, us, Np, fnn_weights)
        return np.concatenate((row1, row2), axis=0)
    # Construct and return the plant.
    ps = parameters['ps']
    Ng = parameters['Ng']
    residual = lambda xGsys: _residual_func(xGsys, us, ps, Ng, parameters,
                                            Np, fnn_weights)
    residual = mpc.getCasadiFunc(residual, [2*Ng], ['xGsys'])
    rootfinder = casadi.rootfinder('fxu', 'newton', residual)
    xguess = 0.5*np.ones((2*Ng, 1))
    xGsys = np.asarray(rootfinder(xguess))
    # Return.
    return xGsys[-Ng:]

def _hybrid_ys(*, us, Np, fnn_weights, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu + fnn() = 0 (continuous time SS) 
    """
    def _hybrid_ode(xs, us, ps, Np, fnn_weights, parameters):
        """ Feed forward evaluation. """
        xs_greybox = _tworeac_greybox_ode(xs, us, ps, parameters)
        xs = xs[:, np.newaxis]
        nn_input = np.concatenate((np.tile(xs, (Np+1, 1)), 
                                   np.tile(us, (Np, 1))), axis=0)
        for i in range(0, len(fnn_weights)-1, 2):
            (W, b) = fnn_weights[i:i+2]
            nn_input = np.tanh(W.T @ nn_input + b[:, np.newaxis])
        nn_output = fnn_weights[-1].T @ nn_input
        return xs_greybox + nn_output[:, 0]
    # Construct and return the plant.
    ps = parameters['ps']
    Ng = parameters['Ng']
    hybrid = lambda xs: _hybrid_ode(xs, us, ps, Np, fnn_weights, parameters)
    hybrid = mpc.getCasadiFunc(hybrid, [Ng], ['xs'])
    rootfinder = casadi.rootfinder('fxu', 'newton', hybrid)
    xguess = 0.5*np.ones((Ng, 1))
    xs = np.asarray(rootfinder(xguess))
    # Return.
    return xs[0:Ng]

def cost(ys, us):
    """ Compute the steady-state cost. """
    (cost, profit) = (105, 160)
    # Return.
    return cost*us - profit*ys[:, -1:]

def compute_cost_curves(*, Nps, model_types, fnn_weights, parameters):
    """ Compute the profit curves for the three models. """
    (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
    uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
    costs = []
    model_types = ['plant', 'grey-box'] + model_types
    Nps = [None, None] + Nps
    fnn_weights = [None, None] + fnn_weights
    for (Np, model_type, fnn_weight) in zip(Nps, model_types, fnn_weights):
        ys = []
        for us in uss:
            if model_type == 'plant':
                ys.append(_plant_ys(us=us, parameters=parameters))
            elif model_type == 'grey-box':
                ys.append(_greybox_ys(us=us, parameters=parameters))
            elif model_type == 'black-box':
                ys.append(_blackbox_ys(us=us, Np=Np, 
                                     fnn_weights=fnn_weight, 
                                     parameters=parameters))
            elif model_type == 'residual':
                ys.append(_residual_ys(us=us, Np=Np, 
                                     fnn_weights=fnn_weight, 
                                     parameters=parameters))
            elif model_type == 'hybrid':
                ys.append(_hybrid_ys(us=us, Np=Np, 
                                     fnn_weights=fnn_weight, 
                                     parameters=parameters))
        ys = np.asarray(ys).squeeze()
        costs.append(cost(ys, uss[:, 0]))
    us = uss.squeeze(axis=-1)
    # Return the compiled model.
    return (costs, us)

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

    print("Plant cost: " + str(plant_cost) + ", Plant u: " + str(plant_us))
    print("Hybrid cost: " + str(hybrid_cost) + ", Plant u: " + str(hybrid_us))
    return None

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_nonlin.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']    
    tworeac_train = PickleTool.load(filename='tworeac_train_nonlin.pickle',
                                    type='read')
    (Nps, model_types, 
     trained_weights) = (tworeac_train['Nps'], 
                         tworeac_train['model_types'],
                         tworeac_train['trained_weights'])
    
    # Create cost curves.
    #xps = _plant_ys(us=np.array([[1.]]), 
    #                parameters=parameters)
    #xGs = _greybox_ys(us=np.array([[1.]]),
    #                  parameters=parameters)
    #xbs = _blackbox_ys(us=np.array([[1.]]), Np=Nps[0], 
    #                   fnn_weights=trained_weights[0][0], 
    #                   parameters=parameters)
    #xrs = _residual_ys(us=np.array([[1.]]), Np=Nps[1], 
    #                   fnn_weights=trained_weights[1][0], 
    #                   parameters=parameters)
    #xhs = _hybrid_ys(us=np.array([[1.]]), Np=Nps[2], 
    #                   fnn_weights=trained_weights[2][0], 
    #                   parameters=parameters)
    fnn_weights = []
    for model_trained_weights in trained_weights:
        fnn_weights.append(model_trained_weights[-1])
    (costs, us) = compute_cost_curves(Nps=Nps, model_types=model_types,
                                      fnn_weights=fnn_weights,
                                      parameters=parameters)
    #solve_plant_hybrid_nlps(Np=Np, fnn_weights=fnn_weights,
    #                        parameters=tworeac_parameters['parameters'])

    PickleTool.save(data_object=dict(us=us, costs=costs),
                    filename='tworeac_ssopt_nonlin.pickle')

main()