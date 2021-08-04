# [depends] %LIB%/hybridId.py %LIB%/tworeacFuncs.py
# [depends] %LIB%/linNonlinMPC.py
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool, sample_prbs_like, SimData
from hybridId import get_rectified_xs, genPlantSsdata
from linNonlinMPC import get_model
from tworeacFuncs import get_plant_pars, plant_ode, get_hyb_greybox_pars

def gen_train_val_data(*, parameters, num_traj, 
                          Nsim_train, Nsim_trainval, 
                          Nsim_val, seed):
    """ Simulate the plant model and generate training and validation data."""

    # Create a list to store data and get some parameters.
    data_list = []
    ulb, uub = parameters['ulb'], parameters['uub']
    tthrow = 10
    p = parameters['ps'][:, np.newaxis]

    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = get_model(ode=plant_ode, parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tthrow, 1))
        
        # Get input trajectories for different simulations.
        if traj == num_traj-1:
            " Get input for train val simulation. "
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=6, num_steps=Nsim_val, 
                                 lb=ulb, ub=uub,
                                 mean_change=60, sigma_change=10, 
                                 seed=seed+1)
        elif traj == num_traj-2:
            " Get input for validation simulation. "
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=8, num_steps=Nsim_trainval, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=10, 
                                 seed=seed+2)
        else:
            " Get input for training simulation. "
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=10, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=30, sigma_change=10, 
                                 seed=seed+3)

        seed += 1

        # Complete input profile and run open-loop simulation.
        u = np.concatenate((us_init, u), axis=0)
        for t in range(tthrow + Nsim):
            plant.step(u[t:t+1, :], p)
        data_list.append(SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u).squeeze(axis=-1),
                y=np.asarray(plant.y[0:-1]).squeeze()))

    # Return the data list.
    return data_list

def main():
    """ Get the parameters, training, and validation data."""
    
    # Get parameters.
    plant_pars = get_plant_pars()
    plant_pars['xs'] = get_rectified_xs(ode=plant_ode, 
                                        parameters=plant_pars)
    hyb_greybox_pars = get_hyb_greybox_pars(plant_pars=plant_pars)

    # Get steady-state training data.
    hx = lambda x: x[plant_pars['yindices']]
    fxu = lambda x, u: plant_ode(x, u, plant_pars['ps'], plant_pars)
    training_ss = genPlantSsdata(fxu=fxu, hx=hx, parameters=plant_pars, 
                                 Ndata=300, xguess=plant_pars['xs'], seed=10)

    # Generate training data.
    training_dyn = gen_train_val_data(parameters=plant_pars,
                                      num_traj=4, Nsim_train=300,
                                      Nsim_trainval=240, Nsim_val=360,
                                      seed=103)

    # Get the dictionary.
    tworeac_parameters = dict(plant_pars = plant_pars,
                              hyb_greybox_pars = hyb_greybox_pars,
                              training_ss = training_ss, 
                              training_dyn = training_dyn)
    
    # Save data.
    PickleTool.save(data_object=tworeac_parameters,
                    filename='tworeac_parameters.pickle')

main()