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
import copy
import numpy as np
from hybridid import (PickleTool, NonlinearEMPCController,
                      SimData, get_cstr_flash_empc_pars,
                      online_simulation, NonlinearPlantSimulator)
from hybridid import _cstr_flash_plant_ode as _plant_ode
from hybridid import _cstr_flash_greybox_ode as _greybox_ode
from hybridid import _cstr_flash_measurement as _measurement

def get_controller(model_func, model_pars, model_type, 
                   cost_pars, mhe_noise_tuning):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Get models.
    ps = model_pars['ps']
    Delta = model_pars['Delta']

    # State-space model (discrete time).
    if model_type == 'hybrid':
        fxu = lambda x, u: model_func(x, u, model_pars)
        breakpoint()
    else:
        fxu = lambda x, u: model_func(x, u, ps, model_pars)
        fxu = c2dNonlin(fxu, Delta)

    # Measurement function.
    hx = lambda x: _measurement(x, model_pars)

    # Get the stage cost.
    if model_type == 'plant':
        lxup_xindices = [5, 7, 9]
        Nx = model_pars['Nx']
    elif model_type == 'grey-box':
        lxup_xindices = [4, 6, 7]
        Nx = model_pars['Ng']
    else:
        lxup_xindices = [4, 6, 7]
        Nx = model_pars['Nx']
    lxup = lambda x, u, p: stage_cost(x, u, p, model_pars, lxup_xindices)
    
    # Get the sizes/sample time.
    (Nu, Ny) = (model_pars['Nu'], model_pars['Ny'])
    Nd = Ny

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
                                   Nx=Nx, Nu=Nu, Ny=Ny, Nd=Nd,
                                   xs=xs, us=us, ds=ds, ys=ys,
                                   empc_pars=cost_pars,
                                   ulb=ulb, uub=uub, Nmpc=Nmpc,
                                   Qwx=Qwx, Qwd=Qwd, Rv=Rv, Nmhe=Nmhe)

def c2dNonlin(fxu, Delta):
    """ Write a quick function to 
        convert a ode to discrete
        time using the RK4 method.
        
        xdot is a function such that 
        dx/dt = f(x, u)
        assume zero-order hold on the input.
    """
    # Get k1, k2, k3, k4.
    k1 = fxu
    k2 = lambda x, u: fxu(x + Delta*(k1(x, u)/2), u)
    k3 = lambda x, u: fxu(x + Delta*(k2(x, u)/2), u)
    k4 = lambda x, u: fxu(x + Delta*k3(x, u), u)
    # Final discrete time function.
    xplus = lambda x, u: x + (Delta/6)*(k1(x, u) + 
                                        2*k2(x, u) + 2*k3(x, u) + k4(x, u))
    return xplus

def get_hybrid_pars(*, greybox_pars, Npast, fnn_weights, xuyscales):
    """ Get the hybrid model parameters. """

    hybrid_pars = copy.deepcopy(greybox_pars)
    # Update sizes.
    Nu, Ny = greybox_pars['Nu'], greybox_pars['Ny']
    hybrid_pars['Nx'] = greybox_pars['Ng'] + Npast*(Nu + Ny)

    # Update steady state.
    ys = _measurement(greybox_pars['xs'], greybox_pars)
    yspseq = np.tile(ys, (Npast, ))
    us = greybox_pars['us']
    uspseq = np.tile(us, (Npast, ))
    xs = greybox_pars['xs']
    hybrid_pars['xs'] = np.concatenate((xs, yspseq, uspseq))
    
    # NN pars.
    hybrid_pars['Npast'] = Npast
    hybrid_pars['fnn_weights'] = fnn_weights 

    # Scaling.
    hybrid_pars['xuyscales'] = xuyscales

    # Return.
    return hybrid_pars

def _fnn(xG, z, u, Npast, xuyscales, fnn_weights):
    """ Compute the NN output. """
    #xmean, xstd = xuyscales['xscale']
    #umean, ustd = xuyscales['uscale']
    #ymean, ystd = xuyscales['yscale']
    #xGzumean = np.concatenate((xmean,
    #                           np.tile(ymean, (Npast, )), 
    #                           np.tile(umean, (Npast+1, ))))
    #xGzustd = np.concatenate((xstd,
    #                          np.tile(ystd, (Npast, )), 
    #                          np.tile(ustd, (Npast+1, ))))
    nn_output = np.concatenate((xG, z, u))#- xGzumean)/xGzustd
    nn_output = nn_output[:, np.newaxis]
    for i in range(0, len(fnn_weights)-2, 2):
        (W, b) = fnn_weights[i:i+2]
        nn_output = np.tanh(W.T @ nn_output + b[:, np.newaxis])
    (Wf, bf) = fnn_weights[-2:]
    nn_output = (Wf.T @ nn_output + bf[:, np.newaxis])[:, 0]
    # Return.
    return nn_output

def interpolate(yseq, Npast, Ny):
    """ y is of dimension: (None, (Np+1)*p)
        Return y of dimension: (None, Np*p). """
    yseq_interp = []
    for t in range(Npast):
        yseq_interp.append(0.5*(yseq[t*Ny:(t+1)*Ny] + yseq[(t+1)*Ny:(t+2)*Ny]))
    # Return.
    return np.concatenate(yseq_interp)

def _hybrid_func(xGz, u, parameters):
    """ The augmented continuous time model. """

    # Extract a few parameters.
    Ng = parameters['Ng']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    ps = parameters['ps']
    Npast = parameters['Npast']
    Delta = parameters['Delta']
    fnn_weights = parameters['fnn_weights']
    xuyscales = parameters['xuyscales']
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']
    xGzmean = np.concatenate((xmean,
                               np.tile(ymean, (Npast, )), 
                               np.tile(umean, (Npast, ))))
    xGzstd = np.concatenate((xstd,
                             np.tile(ystd, (Npast, )), 
                             np.tile(ustd, (Npast, ))))

    # Get some vectors.
    xGz = (xGz - xGzmean)/xGzstd
    u = (u-umean)/ustd
    xG, ypseq, upseq = xGz[:Ng], xGz[Ng:Ng+Npast*Ny], xGz[-Npast*Nu:]
    z = xGz[Ng:]
    hxG = _measurement(xG, parameters)
    
    # Get k1.
    k1 = _greybox_ode(xG*xstd + xmean, u*ustd + umean, ps, parameters)/xstd
    k1 +=  _fnn(xG, z, u, Npast, xuyscales, fnn_weights)

    # Interpolate for k2 and k3.
    ypseq_interp = interpolate(np.concatenate((ypseq, hxG)), Npast, Ny)
    z = np.concatenate((ypseq_interp, upseq))
    
    # Get k2.
    k2 = _greybox_ode((xG + Delta*(k1/2))*xstd + xmean, u*ustd + umean, 
                       ps, parameters)/xstd
    k2 += _fnn(xG + Delta*(k1/2), z, u, Npast, xuyscales, fnn_weights)

    # Get k3.
    k3 = _greybox_ode((xG + Delta*(k2/2))*xstd + xmean, u*ustd + umean, 
                       ps, parameters)/xstd
    k3 += _fnn(xG + Delta*(k2/2), z, u, Npast, xuyscales, fnn_weights)

    # Get k4.
    ypseq_shifted = np.concatenate((ypseq[Ny:], hxG))
    z = np.concatenate((ypseq_shifted, upseq))
    k4 = _greybox_ode((xG + Delta*k3)*xstd + xmean, u*ustd + umean, 
                       ps, parameters)/xstd
    k4 += _fnn(xG + Delta*k3, z, u, Npast, xuyscales, fnn_weights)
    
    # Get the current output/state and the next time step.
    xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
    zplus = np.concatenate((ypseq_shifted, upseq[Nu:], u))
    xGzplus = np.concatenate((xGplus, zplus))
    xGzplus = xGzplus*xGzstd + xGzmean

    # Return the sum.
    return xGzplus

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

def sim_hybrid(hybrid_func, uval, hybrid_pars, greybox_processed_data):
    """ Hybrid validation simulation to make 
        sure the above programmed function is 
        the same is what tensorflow is training. """
    
    # Get initial state.
    t = hybrid_pars['tsteps_steady']
    Np = hybrid_pars['Npast']
    Ny = hybrid_pars['Ny']
    Nu = hybrid_pars['Nu']
    y = greybox_processed_data.y
    u = greybox_processed_data.u
    x = greybox_processed_data.x
    yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[:, np.newaxis]
    up0seq = u[t-Np:t:, ].reshape(Np*Nu, )[:, np.newaxis]
    z0 = np.concatenate((yp0seq, up0seq))
    xG0 = x[t, :][:, np.newaxis]
    xGz0 = np.concatenate((xG0, z0))

    # Start the validation simulation.
    uval = uval[t:, :]
    Nval = uval.shape[0]
    hx = lambda x: _measurement(x, hybrid_pars)
    fxu = lambda x, u: hybrid_func(x, u, hybrid_pars)
    x = xGz0[:, 0]
    yval, xGval = [], []
    xGval.append(x)
    for t in range(Nval):
        yval.append(hx(x))
        x = fxu(x, uval[t, :].T)
        xGval.append(x)
    yval = np.asarray(yval)
    xGval = np.asarray(xGval)[:-1, :8]
    # Return.
    return yval, xGval

def get_mhe_noise_tuning(model_type, model_par):
    # Get MHE tuning.
    if model_type == 'plant':
        Qwx = 1e-6*np.eye(model_par['Nx'])
        Qwd = 1e-6*np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    elif model_type == 'grey-box':
        Qwx = 1e-3*np.eye(model_par['Ng'])
        Qwd = np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    else:
        Qwx = 1e-3*np.eye(model_par['Nx'])
        Qwd = np.eye(model_par['Ny'])
        Rv = 1e-3*np.eye(model_par['Ny'])
    return (Qwx, Qwd, Rv)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                            'cstr_flash_parameters.pickle',
                                            type='read')
    cstr_flash_train = PickleTool.load(filename=
                                            'cstr_flash_train.pickle',
                                            type='read')

    # Get parameters.
    plant_pars = cstr_flash_parameters['plant_pars']
    greybox_pars = cstr_flash_parameters['greybox_pars']
    cost_pars, disturbances = get_cstr_flash_empc_pars(num_days=2,
                                         sample_time=plant_pars['Delta'], 
                                         plant_pars=plant_pars)

    # Get NN weights and the hybrid ODE.
    Np = cstr_flash_train['Nps'][0]
    fnn_weights = cstr_flash_train['trained_weights'][0][0]
    xuyscales = cstr_flash_train['xuyscales']
    hybrid_pars = get_hybrid_pars(greybox_pars=greybox_pars,
                                  Npast=Np,
                                  fnn_weights=fnn_weights,
                                  xuyscales=xuyscales)

    # Check the hybrid function.
    uval = cstr_flash_parameters['training_data'][-1].u
    ytfval = cstr_flash_train['val_predictions'][0].y
    xGtfval = cstr_flash_train['val_predictions'][0].x
    greybox_processed_data = cstr_flash_parameters['greybox_processed_data'][-1]
    yval, xGval = sim_hybrid(_hybrid_func, uval, 
                             hybrid_pars, greybox_processed_data)

    # Run simulations for different model.
    cl_data_list, avg_stage_costs_list, openloop_sol_list = [], [], []
    model_odes = [_plant_ode, _hybrid_func]
    model_pars = [plant_pars, hybrid_pars]
    model_types = ['plant', 'hybrid']
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
                                         Nsim=0, disturbances=disturbances,
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