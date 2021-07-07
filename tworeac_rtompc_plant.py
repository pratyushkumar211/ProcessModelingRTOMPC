# [depends] %LIB%/economicopt.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] tworeac_parameters.pickle
# [makes] pickle
""" Script to perform closed-loop simulations
    with the trained models.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import mpctools as mpc
import casadi
import copy
import numpy as np
from hybridid import PickleTool, SimData, measurement
from linNonlinMPC import RTOLinearMPController, get_model, c2dNonlin
from tworeac_funcs import plant_ode, cost_yup, getEconDistPars
from economicopt import online_simulation

def getMPCController(fxup, hx, model_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Nx = model_pars['Nx']
    Nu = model_pars['Nu']
    Ny = model_pars['Ny']
    Delta = model_pars['Delta']

    # Get dynamic model in discrete time.
    f = c2dNonlin(fxup, Delta, p=True)

    # MHE parameters.
    Qw = 1e-4*np.eye(Nx)
    Rv = model_pars['Rv']

    # Steady states.
    xs = model_pars['xs']
    us = model_pars['us']
    ps = model_pars['ps']

    # RTO optimization parameters.
    rto_type = 'dynmodel_optimization'
    tssOptFreq = 240
    econPars, distPars = getEconDistPars()
    
    # MPC tuning.
    Q = np.eye(Nx) @ np.diag(1/xs**2)
    R = np.eye(Nu) @ np.diag(1/us**2)
    S = 0.1*np.eye(Nu) @ np.diag(1/us**2)
    Nmpc = 120

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Return the NN controller.
    mpccontroller = RTOLinearMPController(fxup=f, hx=hx, 
                                          lyup=cost_yup, econPars=econPars, 
                                          distPars=distPars, rto_type=rto_type, 
                                          tssOptFreq=tssOptFreq, 
                                          picnn_lyup=None, picnn_parids=None, 
                                          Nx=Nx, Nu=Nu, Ny=Ny, xs=xs, us=us, 
                                          ps=ps, Q=Q, R=R, S=S, ulb=ulb, 
                                          uub=uub, Nmpc=Nmpc, Qw=Qw, Rv=Rv)
    
    # Return Controller.
    return mpccontroller

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']

    # Get the plant function handle.
    Delta = plant_pars['Delta']
    plant_fxup = lambda x, u, p: plant_ode(x, u, p, plant_pars)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Get MPC Controller.
    mpccontroller = getMPCController(plant_fxup, plant_hx, plant_pars)

    # Get plant.
    plant_pars['Rv'] = 0*plant_pars['Rv']
    plant = get_model(ode=plant_ode, parameters=plant_pars)

    # Run closed-loop simulation.
    Nsim = 6*24*60
    disturbances = mpccontroller.empcPars[:Nsim, :plant_pars['Np']]

    # Run closed-loop simulation.
    clData, avgStageCosts = online_simulation(plant, mpccontroller,
                                        Nsim=Nsim, disturbances=disturbances,
                                    stdout_filename='tworeac_rtompc_plant.txt')

    # Save data.
    PickleTool.save(data_object=dict(clData=clData,
                                     avgStageCosts=avgStageCosts),
                    filename='tworeac_rtompc_plant.pickle')

main()