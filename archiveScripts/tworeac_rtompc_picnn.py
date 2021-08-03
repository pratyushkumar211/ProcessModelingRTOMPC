# [depends] %LIB%/economicopt.py %LIB%/tworeac_funcs.py
# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] tworeac_parameters.pickle
# [makes] pickle
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
from InputConvexFuncs import get_picnn_pars, picnn_lyup

def getMPCController(fxup, hx, dynmodel_pars, picnn_pars, picnn_lup, 
                     plant_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # PICNN parameter IDs. 
    picnn_parids = [1]

    # Some sizes.
    Nx = dynmodel_pars['Nx']
    Nu = dynmodel_pars['Nu']
    Ny = dynmodel_pars['Ny']
    Delta = dynmodel_pars['Delta']

    # MHE parameters.
    Qw = 1e-4*np.eye(Nx)
    Rv = plant_pars['Rv']

    # Steady states.
    xs = plant_pars['xs']
    us = plant_pars['us']
    ps = dynmodel_pars['ps']

    # RTO optimization parameters.
    rto_type = 'picnn_optimization'
    tssOptFreq = 240
    econPars, distPars = getEconDistPars()
    
    # MPC tuning.
    Q = np.eye(Nx) @ np.diag(1/xs**2)
    R = np.eye(Nu) @ np.diag(1/us**2)
    S = 0.1*np.eye(Nu) @ np.diag(1/us**2)
    Nmpc = 120

    # Get upper and lower bounds.
    ulb = dynmodel_pars['ulb']
    uub = dynmodel_pars['uub']

    # Return the NN controller.
    mpccontroller = RTOLinearMPController(fxup=fxup, hx=hx, 
                                          lyup=cost_yup, econPars=econPars, 
                                          distPars=distPars, rto_type=rto_type, 
                                          tssOptFreq=tssOptFreq, 
                                          picnn_lyup=picnn_lup, 
                                          picnn_parids=picnn_parids, 
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
    tworeac_hybtrain = PickleTool.load(filename='tworeac_hybtrain.pickle',
                                         type='read')
    tworeac_picnntrain = PickleTool.load(filename='tworeac_picnntrain.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']

    # Get the dynamic model function handle.
    hyb_greybox_pars = tworeac_parameters['hyb_greybox_pars']
    hyb_pars = get_hybrid_pars(train=tworeac_hybtrain, 
                                hyb_greybox_pars=hyb_greybox_pars)
    fxup = lambda x, u, p: hybrid_fxup(x, u, p, hyb_pars)
    hx = hybrid_hx

    # Get the PICNN function handle and parameters.
    picnn_pars = get_picnn_pars(train=tworeac_picnntrain, 
                                plant_pars=plant_pars)
    picnn_lup = lambda u, p: picnn_lyup(u, p, picnn_pars)

    # Get MPC Controller.
    mpccontroller = getMPCController(fxup, hx, hyb_pars, 
                                     picnn_pars, picnn_lup, plant_pars)

    # Get plant.
    plant_pars['Rv'] = 0*plant_pars['Rv']
    plant = get_model(ode=plant_ode, parameters=plant_pars)

    # Run closed-loop simulation.
    Nsim = 6*24*60
    disturbances = mpccontroller.empcPars[:Nsim, :plant_pars['Np']]

    # Run closed-loop simulation.
    clData, avgStageCosts = online_simulation(plant, mpccontroller,
                                        Nsim=Nsim, disturbances=disturbances,
                                    stdout_filename='tworeac_rtompc_picnn.txt')

    # Save data.
    PickleTool.save(data_object=dict(clData=clData,
                                     avgStageCosts=avgStageCosts),
                    filename='tworeac_rtompc_picnn.pickle')

main()