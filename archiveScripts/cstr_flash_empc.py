# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [depends] %LIB%/economicopt.py %LIB%/cstr_flash_funcs.py
# [depends] cstr_flash_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import time
import mpctools as mpc
import casadi
import numpy as np
from hybridid import PickleTool, SimData, measurement
from linNonlinMPC import NonlinearEMPCController, c2dNonlin, get_model
from cstr_flash_funcs import plant_ode, cost_yup, getEconDistPars
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
    Nmhe = 5

    # Disturbance model.
    Cd = np.zeros((Ny, Ny))        
    Bd = np.zeros((Nx, Nd))

    # MPC parameters.
    econPars, distPars = getEconDistPars()
    Nmpc = 5
    econPars = np.concatenate((econPars,
                np.repeat(econPars[-1:, :], Nmpc, axis=0)))
    distPars = np.concatenate((distPars,
                np.repeat(distPars[-1:, :], Nmpc, axis=0)))

    # Get the stage cost.
    lyup = lambda y, u, p: cost_yup(y, u, p, model_pars)

    # Get upper and lower bounds.
    ulb = model_pars['ulb']
    uub = model_pars['uub']

    # Steady states.
    xs = model_pars['xs']
    us = model_pars['us']
    ps = model_pars['ps']
    ds = np.zeros((Nd, ))

    # Construct an MPC controller.
    mpccontroller = NonlinearEMPCController(fxup=fxup, hx=hx, lyup=lyup, 
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
    cstr_flash_parameters = PickleTool.load(filename=
                                         'cstr_flash_parameters.pickle',
                                         type='read')
    plant_pars = cstr_flash_parameters['plant_pars']

    # Get the plant function handle.
    Delta = plant_pars['Delta']
    plant_f = lambda x, u, p: plant_ode(x, u, p, plant_pars)
    plant_fxup = c2dNonlin(plant_f, Delta, p=True)
    plant_hx = lambda x: measurement(x, plant_pars)

    # Get MPC Controller.
    plant_pars['xs'] = np.array([ 33.06250061,   0.74028955,   1.29378651,   
                                0.77042495,
       318.47113593,  25.00000045,   0.77531398,   1.87503834,
         1.11654923, 318.95422771])
    plant_pars['us'] = np.array([15.00000013,  8.00000008])
    plant_pars['ps'] = np.array([6., 301.59290954])
    mpccontroller = getMPCController(plant_fxup, plant_hx, plant_pars)

    # Get plant.
    plant_pars['Rv'] = 0*plant_pars['Rv']
    plant = get_model(ode=plant_ode, parameters=plant_pars)

    # Run closed-loop simulation.
    Nsim = 60
    disturbances = mpccontroller.empcPars[:Nsim, :plant_pars['Np']]
    clData, avgStageCosts = online_simulation(plant, mpccontroller,
                                        plant_lyup=controller.lyup,
                                        Nsim=Nsim, disturbances=disturbances,
                                        stdout_filename='cstr_flash_empc.txt')

    # Save data.
    PickleTool.save(data_object=dict(clData=clData,
                                     avgStageCosts=avgStageCosts),
                    filename='cstr_flash_empc.pickle')

main()