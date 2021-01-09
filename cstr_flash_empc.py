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
                      online_simulation, NonlinearPlantSimulator)
from hybridid import _cstr_flash_plant_ode as _plant_ode
from hybridid import _cstr_flash_greybox_ode as _greybox_ode
from hybridid import _cstr_flash_measurement as _measurement

def get_controller(model_ode, model_pars, model_type, 
                   cost_pars, mhe_noise_tuning):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get models.
    ps = model_pars['ps']
    fxu = lambda x, u: model_ode(x, u, ps, model_pars)
    hx = lambda x: _measurement(x, model_pars)

    # Get the stage cost.
    if model_type == 'plant':
        lxup_xindices = [5, 7, 9]
        Nx = model_pars['Nx']
    else:
        lxup_xindices = [4, 6, 7]
        Nx = model_pars['Ng']
    lxup = lambda x, u, p: stage_cost(x, u, p, model_pars, lxup_xindices)
    
    # Get the sizes/sample time.
    (Nu, Ny) = (model_pars['Nu'], model_pars['Ny'])
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
        Bd[0, 0] = 1.
        Bd[1, 1] = 1.
        Bd[3, 2] = 1.
        Bd[4, 3] = 1.
        Bd[5, 4] = 1.
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
    Qwx, Qwd, Rv = mhe_noise_tuning

    # Horizon lengths.
    Nmpc = 60
    Nmhe = 30

    # Return the NN controller.
    return NonlinearEMPCController(fxu=fxu, hx=hx,
                                   lxup=lxup, Bd=Bd, Cd=Cd,
                                   sample_time=Delta,
                                   Nx=Nx, Nu=Nu, Ny=Ny, Nd=Nd, Np=Np,
                                   xs=xs, us=us, ds=ds, ys=ys,
                                   empc_pars=cost_pars,
                                   ulb=ulb, uub=uub, Nmpc=Nmpc,
                                   Qwx=Qwx, Qwd=Qwd, Rv=Rv, Nmhe=Nmhe)

def get_plant(*, parameters):
    """ Return a nonlinear plant simulator object. """
    measurement = lambda x: _measurement(x, parameters)
    # Construct and return the plant.
    plant_ode = lambda x, u, p: _plant_ode(x, u, p, parameters)
    xs = parameters['xs'][:, np.newaxis]
    return NonlinearPlantSimulator(fxup = plant_ode,
                                    hx = measurement,
                                    Rv = parameters['Rv'], 
                                    Nx = parameters['Nx'], 
                                    Nu = parameters['Nu'], 
                                    Np = parameters['Np'], 
                                    Ny = parameters['Ny'],
                                    sample_time = parameters['Delta'], 
                                    x0 = xs)

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

#def collect_cl_data(plant):
#
#    return

#def collect_performance_losses():
#    """ Collect the performance losses 
#        of the trained neural network controllers. """
#    return

def get_mhe_noise_tuning(model_type, model_par):
    # Get MHE tuning.
    if model_type == 'plant':
        Qwx = 1e-6*np.eye(model_par['Nx'])
        Qwd = 1e-6*np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    else:
        Qwx = 1e-3*np.eye(model_par['Ng'])
        Qwd = np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    return (Qwx, Qwd, Rv)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                            'cstr_flash_parameters.pickle',
                                            type='read')
    plant_pars = cstr_flash_parameters['plant_pars']
    greybox_pars = cstr_flash_parameters['greybox_pars']
    cost_pars, disturbances = get_cstr_flash_empc_pars(num_days=2,
                                         sample_time=plant_pars['Delta'], 
                                         plant_pars=plant_pars)

    # Run simulations for different model.
    cl_data_list, avg_stage_costs_list, openloop_sol_list = [], [], []
    model_odes = [_plant_ode, _greybox_ode]
    model_pars = [plant_pars, greybox_pars]
    model_types = ['plant', 'grey-box']
    plant_lxup = lambda x, u, p: stage_cost(x, u, p, plant_pars, [5, 7, 9])
    for (model_ode,
         model_par, model_type) in zip(model_odes, model_pars, model_types):
        mhe_noise_tuning = get_mhe_noise_tuning(model_type, model_par)
        plant = get_plant(parameters=plant_pars)
        controller = get_controller(model_ode, model_par, model_type,
                                    cost_pars, mhe_noise_tuning)
        cl_data, avg_stage_costs, openloop_sol = online_simulation(plant, 
                                         controller,
                                         plant_lxup=plant_lxup,
                                         Nsim=24*60, disturbances=disturbances,
                                         stdout_filename='cstr_flash_empc.txt')
        cl_data_list += [cl_data]
        avg_stage_costs_list += [avg_stage_costs]
        openloop_sol_list += [openloop_sol]

    # Save data.
    PickleTool.save(data_object=dict(cl_data_list=cl_data_list,
                                     cost_pars=cost_pars,
                                     disturbances=disturbances,
                                     avg_stage_costs=avg_stage_costs_list,
                                     openloop_sols=openloop_sol_list),
                    filename='cstr_flash_empc.pickle')

main()