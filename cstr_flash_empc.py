# [depends] %LIB%/hybridid.py %LIB%/HybridModelLayers.py
# [depends] cstr_flash_parameters.pickle
# [makes] pickle
""" Script to perform closed-loop simulations  
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import casadi
import numpy as np
from hybridid import (PickleTool, NonlinearEMPCController,
                      SimData, get_cstr_flash_empc_pars,
                      online_simulation)
from cstr_flash_parameters import (_plant_ode, _greybox_ode, 
                                    _get_model, _measurement)

def get_controller(model_ode, model_pars, model_type, empc_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get models.
    ps = model_pars['ps']
    fxu = lambda x, u: model_ode(x, u, ps, model_pars)
    hx = lambda x: _measurement(x, model_pars)

    # Get the stage cost.
    if model_type == 'plant':
        lxup_xindices = [5, 7, 9]
    else:
        lxup_xindices = [4, 6, 7]
    lxup = lambda x, u, p: stage_cost(x, u, p, model_pars, lxup_xindices)
    
    # Get the sizes/sample time.
    (Nx, Nu, Ny) = (model_pars['Nx'], model_pars['Nu'], 
                    model_pars['Ny'])
    Nd = Ny
    Np = 3
    Delta = model_pars['Delta']

    # Get the disturbance models.
    Bd = np.zeros((Nx, Nd))
    if model_type == 'plant':
        Bd[1, 0] = 1.
        Bd[2, 1] = 1.
        Bd[4, 2] = 1.
        Bd[6, 3] = 1.
        Bd[7, 4] = 1.
        Bd[9, 5] = 1.
    else:
        Bd[1, 0] = 1.
        Bd[2, 1] = 1.
        Bd[3, 2] = 1.
        Bd[5, 3] = 1.
        Bd[6, 4] = 1.
        Bd[7, 5] = 1.
    Cd = np.zeros((Ny, Nd))

    # Get steady states.
    xs = model_pars['xs']
    us = model_pars['us']
    ds = np.zeros((Nd,))
    ys = hx(xs)

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Fictitious noise covariances for MHE.
    Qwx = np.eye(Nx)
    Qwd = 4*np.eye(Nd)
    Rv = np.eye(Ny)

    # Horizon lengths.
    Nmpc = 120
    Nmhe = 30

    # Return the NN controller.
    return NonlinearEMPCController(fxu=fxu, hx=hx,
                                   lxup=lxup, Bd=Bd, Cd=Cd,
                                   sample_time=Delta,
                                   Nx=Nx, Nu=Nu, Ny=Ny, Nd=Nd, Np=Np,
                                   xs=xs, us=us, ds=ds, ys=ys,
                                   empc_pars=empc_pars,
                                   ulb=ulb, uub=uub, Nmpc=Nmpc,
                                   Qwx=Qwx, Qwd=Qwd, Rv=Rv, Nmhe=Nmhe)

def get_plant(*, parameters):
    """ Return a nonlinear plant simulator object. """
    return _get_model(parameters=parameters, plant=True)

def stage_cost(x, u, p, pars, xindices):
    """ Custom stage cost for the CSTR/Flash system. """
    CAf = pars['ps'][0]
    Td = pars['Td']
    pho = pars['pho']
    Cp = pars['Cp']
    kb = pars['kb']    

    # Get inputs, parameters, and states.
    F, Qr, D, Qb = u[0:4]
    ce, ca, cb = p[0:3]
    Hb, CBb, Tb = x[xindices]
    Fb = kb*np.sqrt(Hb)

    # Compute and return cost.
    return ca*F*CAf + ce*Qr + ce*Qb + ce*D*pho*Cp*(Tb-Td) - cb*Fb*CBb

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
    empc_pars = get_cstr_flash_empc_pars(num_days=2, 
                                         sample_time=plant_pars['Delta'])
    cl_data_list = []
    model_odes = [_plant_ode, _greybox_ode]
    model_pars = [plant_pars, greybox_pars]
    model_types = ['plant', 'greybox']
    for (model_ode, 
         model_par, model_type) in zip(model_odes, model_pars, model_types):
        plant = get_plant(parameters=plant_pars)
        controller = get_controller(model_ode, model_par, model_type, empc_pars)


    # Save data.
    PickleTool.save(data_object=cstr_flash_training_data,
                    filename='cstr_flash_empc.pickle')

main()