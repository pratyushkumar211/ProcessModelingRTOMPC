# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/tworeac_funcs.py %LIB%/economicopt.py
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
from hybridid import PickleTool, SimData, measurement, get_model
from linNonlinMPC import NonlinearEMPCController, c2dNonlin
from tworeac_funcs import plant_ode, cost_yup, getEconDistPars
from economicopt import online_simulation

def getMPCController(fxup, hx, model_pars):
    """ Construct the controller object comprised of 
        EMPC regulator and MHE estimator. """

    # Some sizes.
    Nx = model_pars['Nx']
    Nu = model_pars['Nu']
    Ny = model_pars['Ny']
    Nd = Ny
    Delta = model_pars['Delta']

    # MHE parameters.
    Qwx = 1e-4*np.eye(Nx)
    Qwd = 1e-4*np.eye(Nd)
    Rv = model_pars['Rv']
    Nmhe = 60

    # Disturbance model.
    Cd = np.zeros((Ny, Ny))        
    Bd = np.zeros((Nx, Nd))

    # MPC parameters.
    econPars, distPars = getEconDistPars()
    Nmpc = 60
    econPars = np.concatenate((econPars,
                np.repeat(econPars[-1:, :], Nmpc, axis=0)))
    distPars = np.concatenate((distPars,
                np.repeat(distPars[-1:, :], Nmpc, axis=0)))

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Steady states.
    xs = model_pars['xs']
    us = model_pars['us']
    ps = model_pars['ps']
    ds = np.zeros((Nd, ))

    # Return the NN controller.
    mpccontroller = NonlinearEMPCController(fxup=fxup, hx=hx, lyup=cost_yup, 
                                            Bd=Bd, Cd=Cd, Delta=Delta, Nx=Nx,
                                            Nu=Nu, Ny=Ny, Nd=Nd, xs=xs, us=us, 
                                            ds=ds, ps=ps, econPars=econPars, 
                                            distPars=distPars, ulb=ulb, 
                                            uub=uub, Nmpc=Nmpc, Qwx=Qwx, 
                                            Qwd=Qwd, Rv=Rv, Nmhe=Nmhe)
    
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
    plant = get_model(ode=plant_ode, parameters=plant_pars)

    # Run closed-loop simulation.
    Nsim = 12*60
    disturbances = mpccontroller.empcPars[:Nsim, :plant_pars['Np']]
    clData, avgStageCosts = online_simulation(plant, mpccontroller,
                                        Nsim=Nsim, disturbances=disturbances,
                                        stdout_filename='tworeac_empc.txt')

    # Save data.
    PickleTool.save(data_object=dict(clData=clData,
                                     avgStageCosts=avgStageCosts),
                    filename='tworeac_empc.pickle')

main()