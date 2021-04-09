# [depends] %LIB%/hybridid.py %LIB%/tworeac_funcs.py
# [makes] pickle
""" Script to generate the necessary 
    parameters and training data for the 
    three reaction example.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool, sample_prbs_like, SimData
from hybridid import get_rectified_xs, get_model
from tworeac_funcs import get_plant_pars, plant_ode

def gen_train_val_data(*, parameters, num_traj, 
                          Nsim_train, Nsim_trainval, 
                          Nsim_val, seed):
    """ Simulate the plant model 
        and generate training and validation data."""
    # Get the data list.
    data_list = []
    ulb, uub = parameters['ulb'], parameters['uub']
    tsteps_steady = parameters['tsteps_steady']
    p = parameters['ps'][:, np.newaxis]

    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = get_model(ode=plant_ode, parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tsteps_steady, 1))
        
        # Get input trajectories for different simulatios.
        if traj == num_traj-1:
            "Get input for train val simulation."
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=6, num_steps=Nsim_val, 
                                 lb=ulb, ub=uub,
                                 mean_change=60, sigma_change=10, seed=seed+1)
        elif traj == num_traj-2:
            "Get input for validation simulation."
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=6, num_steps=Nsim_trainval, 
                                 lb=ulb, ub=uub,
                                 mean_change=60, sigma_change=10, seed=seed+2)
        else:
            "Get input for training simulation."
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=6, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=60, sigma_change=10, seed=seed+3)

        seed += 1

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

# def get_greybox_val_preds(*, parameters, training_data):
#     """ Use the input profile to compute 
#         the prediction of the grey-box model
#         on the validation data. """
#     model = get_model(ode=greybox_ode, parameters=parameters, plant=False)
#     p = parameters['ps'][:, np.newaxis]
#     u = training_data[-1].u
#     Nsim = u.shape[0]
#     # Run the open-loop simulation.
#     for t in range(Nsim):
#         model.step(u[t:t+1, :], p)
#     x = np.asarray(model.x[0:-1]).squeeze()
#     x = np.insert(x, [2], np.nan, axis=1)
#     data = SimData(t=np.asarray(model.t[0:-1]), x=x, u=u,
#                    y=np.asarray(model.y[:-1]).squeeze())
#     # Return data.
#     return data

def main():
    """ Get the parameters/training/validation data."""
    
    # Get parameters.
    plant_pars = get_plant_pars()
    plant_pars['xs'] = get_rectified_xs(ode=plant_ode, parameters=plant_pars)
    
    # Generate training data.
    training_data = gen_train_val_data(parameters=plant_pars,
                                        num_traj=8, Nsim_train=360,
                                        Nsim_trainval=360, Nsim_val=360,
                                        seed=103)
    
    # Create a dict and save.
    tworeac_parameters = dict(plant_pars=plant_pars,
                              training_data=training_data)
    
    # Save data.
    PickleTool.save(data_object=tworeac_parameters,
                    filename='tworeac_parameters.pickle')

main()