# [depends] %LIB%/hybridid.py %LIB%/linNonlinMPC.py
# [makes] pickle
""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import mpctools as mpc
import numpy as np
import scipy.linalg
from hybridid import PickleTool, c2d, sample_prbs_like, SimData
from tworeac_nonlin_funcs import get_parameters, get_model, get_rectified_xs

def gen_train_val_data(*, parameters, num_traj, 
                           Nsim_train, Nsim_trainval, 
                           Nsim_val, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    data_list = []
    ulb = parameters['ulb']
    uub = parameters['uub']
    tsteps_steady = parameters['tsteps_steady']
    p = parameters['ps'][:, np.newaxis]

    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = get_model(parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tsteps_steady, 1))
        
        # Get input trajectories for different simulatios.
        if traj == num_traj-1:
            "Get input for train val simulation."
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=24, num_steps=Nsim_val, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+1)
        elif traj == num_traj-2:
            "Get input for validation simulation."
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=24, num_steps=Nsim_trainval, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=2, seed=seed+2)
        else:
            "Get input for training simulation."
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=12, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=6, seed=seed+3)

        seed += 1
        umid = 0.5*(ulb + uub)[0]
        u = np.where(u<umid, ulb, uub)

        # Complete input profile and run open-loop simulation.
        u = np.concatenate((us_init, u), axis=0)
        for t in range(tsteps_steady + Nsim):
            plant.step(u[t:t+1, :], p)
        data_list.append(SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u).squeeze(axis=-1),
                y=np.asarray(plant.y[0:-1]).squeeze()))
    # Return the data list.
    return data_list

def get_greybox_val_preds(*, parameters, training_data):
    """ Use the input profile to compute 
        the prediction of the grey-box model
        on the validation data. """
    #model = get_model(parameters=parameters, plant=False)
    #p = parameters['ps'][:, np.newaxis]
    u = training_data[-1].u
    y = training_data[-1].y
    x = training_data[-1].x
    t = training_data[-1].t
    #Nsim = u.shape[0]
    # Run the open-loop simulation.
    #for t in range(Nsim):
    #    model.step(u[t:t+1, :], p)
    #x = np.asarray(model.x[0:-1]).squeeze()
    #data = SimData(t=np.asarray(model.t[0:-1]), x=x, u=u,
    #               y=np.asarray(model.y[:-1]).squeeze())
    data = SimData(t=t, x=x, u=u, y=y)
    # Return data.
    return data

def main():
    """ Get the parameters/training/validation data."""
    
    # Get parameters.
    parameters = get_parameters()
    parameters['xs'] = get_rectified_xs(parameters=parameters)
    
    # Generate training data.
    training_data = gen_train_val_data(parameters=parameters,
                                        num_traj=18, Nsim_train=360,
                                        Nsim_trainval=720, Nsim_val=720,
                                        seed=100)
    greybox_val_data = get_greybox_val_preds(parameters=
                                            parameters, 
                                            training_data=training_data)
    
    # Create a dict and save.
    tworeac_parameters = dict(parameters=parameters, 
                              training_data=training_data,
                              greybox_val_data=greybox_val_data)
    
    # Save data.
    PickleTool.save(data_object=tworeac_parameters, 
                    filename='tworeac_parameters_nonlin.pickle')

main()