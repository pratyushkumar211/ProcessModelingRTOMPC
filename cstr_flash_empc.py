# [depends] %LIB%/hybridid.py %LIB%/HybridModelLayers.py
# [depends] cstr_flash_parameters.pickle
# [makes] pickle
""" Script to perform closed-loop simulations  
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import (PickleTool, NonlinearEMPCController,
                      SimData, get_cstr_flash_empc_pars,
                      online_simulation)
from cstr_flash_parameters import _plant_ode, _greybox_ode, 
                                  _get_model, _measurement

def get_controller(model_ode, model_pars, lxup_xindices, empc_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get models.
    ps = model_pars['ps']
    fxu = lambda x, u: model_ode(x, u, ps, model_pars)
    hx = lambda x: _measurement(x, model_pars)

    # Get the sizes.
    (Nx, Nu, Ny) = (model_pars['Nx'], model_pars['Nu'], 
                    model_pars['Nd'])
    Nd = Ny

    # Get the disturbance models.
    Bd, Cd

    # Get steady state.

    # Return the NN controller.
    return NonlinearEMPCController(
        fxu=fxu, hx=hx, lxup, Bd, Cd, sample_time,
                     Nx, Nu, Ny, Nd,
                     xs, us, ds, ys,
                     empc_pars,
                     ulb, uub, Nmpc,
                     Qwx, Qwd, Rv, Nmhe)
    

def get_plant(*, parameters):
    """ Return a nonlinear plant simulator object. """
    return _get_model(parameters=parameters, plant=True)

def stage_cost(x, u, p, pars, xindices):
    """ Custom stage cost for the CSTR/Flash system. """
    CAf = pars['CAf']
    Td = pars['Td']
    pho = pars['pho']
    Cp = pars['Cp']
    kb = pars['kb']    
    # Get inputs, parameters, and states.
    F, Qr, D, Qb = u[0:4]
    ce, ca, cb = p[0:1]
    Hb, CBb, Tb = x[xindices]
    Fb = kb*np.sqrt(Hb)
    # Return.
    return ca*F*CAf + ce*Qr + ce*Qb + ce*D*pho*Cp*(Td-Tb) - cb*Fb*CBb

def collect_cl_data(plant):

    return

def collect_performance_losses():
    """ Collect the performance losses 
        of the trained neural network controllers. """

    return

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                            'cstr_flash_parameters.pickle',
                                            type='read')
    plant_pars = cstr_flash_parameters['plant_pars']
    greybox_pars = cstr_flash_parameters['greybox_pars']

    cl_data_list = []
    model_odes = [_plant_ode, _greybox_ode]
    model_pars = [plant_pars, greybox_pars]
    for (model_ode, model_par) in zip(model_odes, model_pars):
        plant = get_plant(parameters=plant_pars)
        controller = get_controller(model_ode, model_par)


    # Save data.
    PickleTool.save(data_object=cstr_flash_training_data,
                    filename='cstr_flash_empc.pickle')

main()