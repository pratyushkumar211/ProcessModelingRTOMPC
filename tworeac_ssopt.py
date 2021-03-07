# [depends] %LIB%/hybridid.py tworeac_parameters.pickle
# [depends] tworeac_bbtrain.pickle
# [makes] pickle
""" Script to use the trained hybrid model for 
    steady-state optimization.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import casadi
import copy
import matplotlib.pyplot as plt
import mpctools as mpc
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import PickleTool, measurement
from economicopt import get_bb_parameters, bb_fxu, bb_hx
from economicopt import get_ss_optimum, get_xuguess, c2dNonlin
from tworeac_funcs import cost_yup, plant_ode, greybox_ode

# def _plant_ys(*, us, parameters):
#     """ Return the plant xs. 
#         Solve: f(x, u) = 0 (continuous time)
#     """
#     # Construct and return the plant.
#     ps = parameters['ps']
#     Nx, Ny = parameters['Nx'], parameters['Ny']
#     plant = lambda x: _tworeac_plant_ode(x, us, ps, parameters)
#     plant = mpc.getCasadiFunc(plant, [Nx], ['xs'])
#     rootfinder = casadi.rootfinder('fxu', 'newton', plant)
#     xguess = 0.5*np.ones((Nx, 1))
#     xs = np.asarray(rootfinder(xguess))
#     # Return.
#     return xs[0:Ny]

# def _greybox_ys(*, us, parameters):
#     """ Return the plant xs. 
#         Solve: Ax + Bu = 0 (continuous time SS) 
#     """
#     Construct and return the plant.
#     ps = parameters['ps']
#     Ng = parameters['Ng']
#     greybox = lambda x: _tworeac_greybox_ode(x, us, ps, parameters)
#     greybox = mpc.getCasadiFunc(greybox, [Ng], ['xs'])
#     rootfinder = casadi.rootfinder('fxu', 'newton', greybox)
#     xguess = 0.5*np.ones((Ng, 1))
#     xs = np.asarray(rootfinder(xguess))
#     Return.
#     return xs[0:Ng]

# def _blackbox_ys(*, us, Np, fnn_weights, parameters):
#     """ Return the plant xs. 
#         Solve: -xs + fnn(xs, 2*xs, 2*us, us) = 0 (continuous time SS) 
#     """
#     Construct and return the plant.
#     Ng = parameters['Ng']
#     blackbox = lambda xs: _blackbox_func(xs, us, Np, fnn_weights)
#     blackbox = mpc.getCasadiFunc(blackbox, [Ng], ['xs'])
#     rootfinder = casadi.rootfinder('fxu', 'newton', blackbox)
#     xguess = 0.5*np.ones((Ng, 1))
#     xGsys = np.asarray(rootfinder(xguess))
#     Return.
#     return xGsys[-Ng:]

# def _residual_func(xGsys, us, ps, Ng, parameters, Np, fnn_weights):
#     """ Residual function. """
#     def _fg(xs, us, ps, parameters):
#         """ Grey-box part. """ 
#         Delta = parameters['sample_time']
#         k1 = _tworeac_greybox_ode(xs, us, ps, parameters)
#         k2 = _tworeac_greybox_ode(xs + Delta*(k1/2), us, ps, parameters)
#         k3 = _tworeac_greybox_ode(xs + Delta*(k2/2), us, ps, parameters)
#         k4 = _tworeac_greybox_ode(xs + Delta*k3, us, ps, parameters)
#         return xs + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
#     def _fnn(xs, us, Np, fnn_weights):
#         """ Neural network part. """
#         xs = xs[:, np.newaxis]
#         nn_input = np.concatenate((np.tile(xs, (Np, 1)), 
#                                    np.tile(us, (Np, 1))), axis=0)
#         for i in range(0, len(fnn_weights)-1, 2):
#             (W, b) = fnn_weights[i:i+2]
#             nn_input = np.tanh(W.T @ nn_input + b[:, np.newaxis])
#         nn_output = fnn_weights[-1].T @ nn_input
#         return nn_output[:, 0]
#     (xGs, ys) = np.split(xGsys, [Ng, ])
#     row1 = -xGs + _fg(xGs, us, ps, parameters)
#     row2 = -ys + xGs + _fnn(ys, us, Np, fnn_weights)
#     return np.concatenate((row1, row2), axis=0)

# def _residual_ys(*, us, Np, fnn_weights, parameters):
#     """ Return the plant xs. 
#         Solve: Ax + Bu + fnn() = 0 (continuous time SS) 
#     """
#     Construct and return the plant.
#     ps = parameters['ps']
#     Ng = parameters['Ng']
#     residual = lambda xGsys: _residual_func(xGsys, us, ps, Ng, parameters,
#                                             Np, fnn_weights)
#     residual = mpc.getCasadiFunc(residual, [2*Ng], ['xGsys'])
#     rootfinder = casadi.rootfinder('fxu', 'newton', residual)
#     xguess = 0.5*np.ones((2*Ng, 1))
#     xGsys = np.asarray(rootfinder(xguess))
#     Return.
#     return xGsys[-Ng:]

# def _hybrid_ode(xs, us, ps, Np, fnn_weights, parameters):
#     """ Feed forward evaluation. """
#     xs_greybox = _tworeac_greybox_ode(xs, us, ps, parameters)
#     xs = xs[:, np.newaxis]
#     nn_input = np.concatenate((np.tile(xs, (Np+1, 1)), 
#                                np.tile(us, (Np, 1))), axis=0)
#     for i in range(0, len(fnn_weights)-1, 2):
#         (W, b) = fnn_weights[i:i+2]
#         nn_input = np.tanh(W.T @ nn_input + b[:, np.newaxis])
#     nn_output = fnn_weights[-1].T @ nn_input
#     return xs_greybox + nn_output[:, 0]

# def _hybrid_ys(*, us, Np, fnn_weights, parameters):
#     """ Return the plant xs. 
#         Solve: Ax + Bu + fnn() = 0 (continuous time SS) 
#     """
#     Construct and return the plant.
#     ps = parameters['ps']
#     Ng = parameters['Ng']
#     hybrid = lambda xs: _hybrid_ode(xs, us, ps, Np, fnn_weights, parameters)
#     hybrid = mpc.getCasadiFunc(hybrid, [Ng], ['xs'])
#     rootfinder = casadi.rootfinder('fxu', 'newton', hybrid)
#     xguess = 0.5*np.ones((Ng, 1))
#     xs = np.asarray(rootfinder(xguess))
#     Return.
#     return xs[0:Ng]

# def cost(ys, us):
#     """ Compute the steady-state cost. """
#     (cost, profit) = (105, 160)
#     Return.
#     return cost*us - profit*ys[:, -1:]

# def compute_cost_curves(*, Nps, model_types, fnn_weights, parameters):
#     """ Compute the profit curves for the three models. """
#     (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
#     uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
#     costs = []
#     model_types = ['plant', 'grey-box'] + model_types
#     Nps = [None, None] + Nps
#     fnn_weights = [None, None] + fnn_weights
#     for (Np, model_type, fnn_weight) in zip(Nps, model_types, fnn_weights):
#         ys = []
#         for us in uss:
#             if model_type == 'plant':
#                 ys.append(_plant_ys(us=us, parameters=parameters))
#             elif model_type == 'grey-box':
#                 ys.append(_greybox_ys(us=us, parameters=parameters))
#             elif model_type == 'black-box':
#                 ys.append(_blackbox_ys(us=us, Np=Np, 
#                                      fnn_weights=fnn_weight, 
#                                      parameters=parameters))
#             elif model_type == 'residual':
#                 ys.append(_residual_ys(us=us, Np=Np, 
#                                      fnn_weights=fnn_weight, 
#                                      parameters=parameters))
#             elif model_type == 'hybrid':
#                 ys.append(_hybrid_ys(us=us, Np=Np, 
#                                      fnn_weights=fnn_weight, 
#                                      parameters=parameters))
#         ys = np.asarray(ys).squeeze()
#         costs.append(cost(ys, uss[:, 0]))
#     us = uss.squeeze(axis=-1)
#     Return the compiled model.
#     return (costs, us)

# def plant_opt(parameters):
#     """ Compute plant optimal. """
#     Get parameters and casadi funcs.
#     ps = parameters['ps']
#     (ulb, uub) = (parameters['lb']['u'].squeeze(), 
#                   parameters['ub']['u'].squeeze())
#     (Nx, Ny, Nu) = (parameters['Nx'], 
#                     parameters['Ny'],
#                     parameters['Nu'])
#     plant = lambda x, u: _tworeac_plant_ode(x, u, ps, parameters)
#     plant = mpc.getCasadiFunc(plant, [Nx, Nu], ['xs', 'us'])

#     Construct NLP and solve.
#     xs = casadi.SX.sym('xs', Nx)
#     us = casadi.SX.sym('us', Nu)
#     plant_nlp = dict(x=casadi.vertcat(xs, us), 
#                      f=cost(xs.T[0:Ny], us), 
#                      g=casadi.vertcat(plant(xs, us), us))
#     plant_nlp = casadi.nlpsol('plant_nlp', 'ipopt', plant_nlp)
#     xguess = 0.8*np.ones((Nx+Nu, 1))
#     lbg = np.concatenate((np.zeros((Nx, 1)), np.array([[ulb]])), axis=0)
#     ubg = np.concatenate((np.zeros((Nx, 1)), np.array([[uub]])), axis=0)
#     plant_nlp_soln = plant_nlp(x0=xguess, lbg=lbg, ubg=ubg)
#     (opt_cost, opt_us) = (plant_nlp_soln['f'].full().squeeze(axis=-1)[0], 
#                   plant_nlp_soln['g'].full().squeeze()[-1])
#     Return.
#     return (opt_cost, opt_us)

# def compute_suboptimality_gaps(*, Nps, model_types, 
#                                   trained_weights, parameters):
#     """ Solve steady state NLPs for plant and 
#         hybrid models. """
#     def _construct_and_solve_nlp(model_type, Ng, Nu, Np, 
#                                  fnn_weights, parameters):
#         """ Construct and solve NLP depending on model type. """
#         if model_type == 'residual':
#             model = lambda xGsys, us: _residual_func(xGsys, us, ps, Ng, 
#                                                     parameters, Np, fnn_weights)
#             model = mpc.getCasadiFunc(model, [2*Ng, Nu], ['xGsys', 'us'])
#             xGsys = casadi.SX.sym('xGsys', 2*Ng)
#             us = casadi.SX.sym('us', Nu)
#             First construct the NLP for the hybrid model.
#             model_nlp = dict(x=casadi.vertcat(xGsys, us), 
#                              f=cost(xGsys.T[Ng:], us), 
#                              g=casadi.vertcat(model(xGsys, us), us))
#             model_nlp = casadi.nlpsol('model_nlp', 'ipopt', model_nlp)
#             xguess = 0.8*np.ones((2*Ng+Nu, 1))
#             lbg = np.concatenate((np.zeros((2*Ng, 1)), np.array([[ulb]])), 
#                                   axis=0)
#             ubg = np.concatenate((np.zeros((2*Ng, 1)), np.array([[uub]])), 
#                                   axis=0)
#         else:
#             if model_type == 'black-box':
#                 model = lambda xs, us: _blackbox_func(xs, us, Np, fnn_weights)
#             else:
#                 model = lambda xs, us: _hybrid_ode(xs, us, ps, Np, 
#                                                    fnn_weights, parameters)
#             model = mpc.getCasadiFunc(model, [Ng, Nu], ['xs', 'us'])
#             xs = casadi.SX.sym('xs', Ng)
#             us = casadi.SX.sym('us', Nu)
#             First construct the NLP for the hybrid model.
#             model_nlp = dict(x=casadi.vertcat(xs, us), 
#                              f=cost(xs.T[0:Ng], us), 
#                              g=casadi.vertcat(model(xs, us), us))
#             model_nlp = casadi.nlpsol('model_nlp', 'ipopt', model_nlp)
#             xguess = 0.8*np.ones((Ng+Nu, 1))
#             lbg = np.concatenate((np.zeros((Ng, 1)), np.array([[ulb]])), axis=0)
#             ubg = np.concatenate((np.zeros((Ng, 1)), np.array([[uub]])), axis=0)
#         model_nlp_soln = model_nlp(x0=xguess, lbg=lbg, ubg=ubg)
#         us_opt = model_nlp_soln['g'].full()[-Nu:, :]
#         return us_opt
#     Get plant optimal costs.
#     (plant_opt_cost, _) = plant_opt(parameters)

#     Get some parameters.    
#     ps = parameters['ps']
#     (ulb, uub) = (parameters['lb']['u'].squeeze(), 
#                 parameters['ub']['u'].squeeze())
#     (Ng, Nu) = (parameters['Ng'], parameters['Nu'])

#     Create lists to store suboptimality gaps, and do calcs.
#     sub_gaps = []
#     for (Np, model_type, 
#          model_trained_weights) in zip(Nps, model_types, trained_weights):
#         model_sub_gaps = []
#         for fnn_weights in model_trained_weights:
#             Get casadi funcs and variables.
#             model_us = _construct_and_solve_nlp(model_type, Ng, Nu, Np, 
#                                                 fnn_weights, parameters)
#             plant_ys = _plant_ys(us=model_us, parameters=parameters)
#             plant_cost = cost(plant_ys.T, model_us).squeeze()
#             gap = 100*np.abs((plant_cost-plant_opt_cost)/plant_opt_cost)
#             model_sub_gaps.append(gap)
#         model_sub_gaps = np.asarray(model_sub_gaps).squeeze()
#         sub_gaps.append(model_sub_gaps)
#     return sub_gaps

# def compute_cost_mses(costss):
#     """ Get the absolute difference between the 
#         plant cost and model costs. """
#     num_models = len(costss[0][2:])
#     cost_mses = []
#     for i in range(num_models):
#         model_cost_mse = []
#         for costs in costss:
#             plant = costs[0]
#             Delta = np.abs(plant - costs[i+2]).squeeze()
#             model_cost_mse.append(Delta)
#         cost_mses.append(model_cost_mse)
#     return cost_mses

def get_bbpars_fxu_hx(*, train, parameters):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    Np = train['Np']
    hN_weights = train['trained_weights'][-1]
    xuyscales = train['xuyscales']
    bb_pars = get_bb_parameters(Np=Np, xuyscales=xuyscales, 
                                hN_weights=hN_weights, 
                                parameters=parameters)
    
    # Get function handles.
    fxu = lambda x, u: bb_fxu(x, u, bb_pars)
    hx = lambda x: bb_hx(x, bb_pars)

    # Return.
    return bb_pars, fxu, hx

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    tworeac_bbtrain = PickleTool.load(filename='tworeac_bbtrain.pickle',
                                      type='read')

    # Get cost function handle.
    p = [105, 160]
    lyu = lambda y, u: cost_yup(y, u, p)

    # Get the black-box model parameters and function handles.
    bb_pars, blackb_fxu, blackb_hx = get_bbpars_fxu_hx(train=tworeac_bbtrain, 
                                                       parameters=parameters) 

    # Get the plant function handle.
    Delta = parameters['Delta']
    ps = parameters['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, parameters)
    plant_fxu = c2dNonlin(plant_fxu, Delta)
    plant_hx = lambda x: measurement(x, parameters)

    # Get the grey-box function handle.
    gb_fxu = lambda x, u: greybox_ode(x, u, ps, parameters)
    gb_fxu = c2dNonlin(gb_fxu, Delta)
    gb_pars = copy.deepcopy(parameters)
    gb_pars['Nx'] = len(parameters['gb_indices'])

    model_types = ['plant', 'grey-box', 'black-box']
    fxu_list = [plant_fxu, gb_fxu, blackb_fxu]
    hx_list = [plant_hx, plant_hx, blackb_hx]
    par_list = [parameters, gb_pars, bb_pars]
    Nps = [None, None, bb_pars['Np']]
    for (model_type, fxu, hx, model_pars, Np) in zip(model_types, fxu_list, 
                                                     hx_list, par_list, Nps):

        # Get guess. 
        xuguess = get_xuguess(model_type=model_type, 
                              plant_pars=parameters, Np=Np)

        # Get the steady state optimum.
        xs, us, ys = get_ss_optimum(fxu=fxu, hx=hx, lyu=lyu, 
                                    parameters=model_pars, guess=xuguess)

        # Print. 
        print("Model type: " + model_type)
        print('us: ' + str(us))
        print('ys: ' + str(ys))

    # Create cost curves.
    # costss = []
    # num_samples = list(num_samples)
    # for (i, num_sample) in enumerate(num_samples):
    #     fnn_weights = []
    #     for model_trained_weights in trained_weights:
    #         fnn_weights.append(model_trained_weights[i])
    #     (costs, us) = compute_cost_curves(Nps=Nps, model_types=model_types,
    #                                       fnn_weights=fnn_weights,
    #                                       parameters=parameters)
    #     costss.append(costs)
    
    # cost_mses = compute_cost_mses(costss=costss)
    # sub_gaps = compute_suboptimality_gaps(Nps=Nps, model_types=model_types,
    #                                       trained_weights=trained_weights,
    #                                       parameters=parameters)

    # PickleTool.save(data_object=dict(us=us, costss=costss, 
    #                                  cost_mses=cost_mses,
    #                                  sub_gaps=sub_gaps),
    #                 filename='tworeac_ssopt_nonlin.pickle')

main()