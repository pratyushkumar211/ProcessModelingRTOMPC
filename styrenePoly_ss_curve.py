# [depends] %LIB%/reacFuncs.py
# [makes] pickle
import sys
sys.path.append('lib/')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from linNonlinMPC import (c2dNonlin, getSSOptimum,
                          getXsYsSscost, getXsUsSSCalcGuess)
from styrenePolyFuncs import plant_ode

# def get_xguess(*, model_type, plant_pars, model_pars):
#     """ Get x guess based on model type. """
#     if model_type == 'Plant':
#         xs = plant_pars['xs']
#     elif model_type == 'Black-Box-NN':
#         None
#     elif model_type == 'Hybrid-FullGb':
#         None
#     elif model_type == 'Hybrid-PartialGb':
#         None
#     else:
#         None
#     # Return.
#     return xs

def getSSCurveData(*, fxu, hx, model_pars, Nus, usind):
    """ Function to get the plant steady-state profile. """

    # Fix the steady state values of all the control inputs except 
    # the input index provided.
    ulb = model_pars['us'].copy()
    uub = model_pars['us'].copy()
    ulb[usind] = model_pars['ulb'][usind]
    uub[usind] = model_pars['uub'][usind]
    us_list = list(np.linspace(ulb, uub, Nus))

    # Create list to store the steady state xs.
    xs_list = []

    # Loop over all the steady state inputs.
    for us in us_list:

        # Get guess.
        xguess = model_pars['xs']

        # Get the xs and sscost.
        xs, _, _ = getXsYsSscost(fxu=fxu, hx=hx, lxu=None, 
                                 us=us, parameters=model_pars, 
                                 xguess=xguess)

        xs_list += [xs]

    # Get steady-states as arrays.
    xs = np.asarray(xs_list)
    us = np.asarray(us_list)

    # Create a dictionary and save.
    ssCurveData = dict(us=us, xs=xs)

    # Return.
    return ssCurveData

def main():
    """ Main function to be executed. """

    # Load data.
    with open("styrenePoly_parameters.pickle", "rb") as stream:
        styrenePoly_parameters = pickle.load(stream)

    # Get plant and hybrid model parameters.
    plant_pars = styrenePoly_parameters['plant_pars']

    # Plant function handles.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_f = c2dNonlin(plant_fxu, Delta)
    plant_h = lambda x: x[plant_pars['yindices']]

    # Vary each manipulated input to get the corresponding xs.
    ssCurveData_list = []
    for usind in range(plant_pars['Nu']):

        ssCurveData = getSSCurveData(fxu=plant_f, hx=plant_h,
                             model_pars=plant_pars, Nus=100, usind=usind)
        ssCurveData_list += [ssCurveData]

    # Save.
    with open('styrenePoly_ss_curve.pickle', "wb") as stream:
        pickle.dump(ssCurveData_list, stream)

main()