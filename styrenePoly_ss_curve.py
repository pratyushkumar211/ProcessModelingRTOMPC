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
    ulb = model_pars['us']
    uub = model_pars['us']
    #ulb[usind] = plant_pars['ulb'][usind]
    #uub[usind] = plant_pars['uub'][usind]
    us_list = list(np.linspace(ulb, uub, Nus))
    
    # Create list to store the steady state xs.
    xs_list = []

    # Number of guesses to solve the ss rootfinding problem. 
    numGuess = 3

    # Compute SS cost.
    for us in us_list:
        
        for _ in range(numGuess):

            # Get guess.
            xguess = model_pars['xs']
            #xguess[3] = 500
            xguess = np.array([1e+02, 2.26, 4.93e+1, 4e+02,
                               3.01e+02, 2.46e+02, 4.04e+03, 1.14e+05])
            breakpoint()
            # Get the xs and sscost.
            xs, _, _ = getXsYsSscost(fxu=fxu, hx=hx, lxu=None, 
                                     us=us, parameters=model_pars, 
                                     xguess=xguess)
            breakpoint()
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

    # Number of ss inputs at which to compute the cost.
    ssCurveData = getSSCurveData(fxu=plant_f, hx=plant_h,
                             model_pars=plant_pars, Nus=100, usind=0)

    # Save.
    with open('styrenePoly_ss_curve.pickle', "wb") as stream:
        pickle.dump(ssCurveData, stream)

main()