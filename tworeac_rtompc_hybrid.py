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
from TwoReacHybridFuncs import get_hybrid_pars, hybrid_fxup, hybrid_hx

def getMPCController(fxup, hx, model_pars, plant_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Nx = model_pars['Nx']
    Nu = model_pars['Nu']
    Ny = model_pars['Ny']
    Delta = model_pars['Delta']

    # MHE parameters.
    Qw = 1e-4*np.eye(Nx)
    Rv = plant_pars['Rv']

    # Steady states.
    xs = plant_pars['xs']
    us = plant_pars['us']
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
    mpccontroller = RTOLinearMPController(fxup=fxup, hx=hx, 
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
    tworeac_hybtrain = PickleTool.load(filename='tworeac_hybtrain.pickle',
                                         type='read')

    # Get the dynamic model function handle.
    hyb_greybox_pars = tworeac_parameters['hyb_greybox_pars']
    hyb_pars = get_hybrid_pars(train=tworeac_hybtrain, 
                                hyb_greybox_pars=hyb_greybox_pars)
    fxup = lambda x, u, p: hybrid_fxup(x, u, p, hyb_pars)
    hx = hybrid_hx

    # Get MPC Controller.
    mpccontroller = getMPCController(fxup, hx, hyb_pars, plant_pars)

    # Get plant.
    plant_pars['Rv'] = 0*plant_pars['Rv']
    plant = get_model(ode=plant_ode, parameters=plant_pars)

    # Run closed-loop simulation.
    Nsim = 6*24*60
    disturbances = mpccontroller.empcPars[:Nsim, :plant_pars['Np']]

    # Run closed-loop simulation.
    clData, avgStageCosts = online_simulation(plant, mpccontroller,
                                        Nsim=Nsim, disturbances=disturbances,
                                    stdout_filename='tworeac_rtompc_hybrid.txt')

    # Save data.
    PickleTool.save(data_object=dict(clData=clData,
                                     avgStageCosts=avgStageCosts),
                    filename='tworeac_rtompc_hybrid.pickle')

main()