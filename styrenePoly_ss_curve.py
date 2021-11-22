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

def get_xguess(*, model_type, plant_pars, model_pars):
    """ Get x guess based on model type. """
    if model_type == 'Plant':
        xs = plant_pars['xs']
    elif model_type == 'Black-Box-NN':
        None
    elif model_type == 'Hybrid-FullGb':
        None
    elif model_type == 'Hybrid-PartialGb':
        None
    else:
        None
    # Return.
    return xs

def getSSCurveData(*, plant_fxu, plant_hx, plant_pars):
    """ Function to get the plant steady-state profile. """

    # Get a linspace of steady-state u values.
    ulb, uub = plant_pars['ulb'], plant_pars['uub']
    us_list = list(np.linspace(ulb, uub, Nus))

    # Create lists to store xs and sscosts. 
    xs_list = []

    # Compute SS cost.
    for us in us_list:
            
        # Get guess.
        xguess = get_xguess(model_type=model_type, 
                                plant_pars=plant_pars, 
                                model_pars=model_pars)
            
        # Get the xs and sscost.
        xs, _, sscost = getXsYsSscost(fxu=fxu, hx=hx, lxu=cost_lxup, 
                                          us=us, parameters=model_pars, 
                                          xguess=xguess)
        model_xs += [xs]
        model_sscost += [sscost]

        # Model steady states and costs.
        model_xs = np.asarray(model_xs)
        model_sscost = np.asarray(model_sscost)

        # Store steady states and costs in lists.
        xs_list += [model_xs]
        sscosts += [model_sscost]

    # Convert to a rank 1 array.
    us = np.asarray(us_list)[:, 0]

    # Create a dictionary and save.
    reac_ssopt = dict(us=us, xs=xs_list, sscosts=sscosts)

    # Return.
    return reac_ssopt

def main():
    """ Main function to be executed. """

    # Load data.
    with open('styrenePoly_parameters.pickle', "wb") as stream:
        styrenePoly_parameters = pickle.load(styrenePoly_parameters, stream)

    # Get plant and hybrid model parameters.
    plant_pars = styrenePoly_parameters['plant_pars']

    # Plant function handles.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_f = c2dNonlin(plant_fxu, Delta)
    plant_h = lambda x: x[plant_pars['yindices']]

    # Number of ss inputs at which to compute the cost.
    Nus = 100
    xs_list = getSSCurveData(plant_fxu=plant_f,
                                 plant_hx=plant_h,
                                 plant_pars=plant_pars)

    # Save.
    with open('styrenePoly_ss_curve.pickle', "wb") as stream:
        pickle.dump(styrenePoly_parameters, stream)

main()