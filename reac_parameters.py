# [depends] %LIB%/hybridId.py %LIB%/reacFuncs.py
# [depends] %LIB%/linNonlinMPC.py
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool, sample_prbs_like, SimData
from hybridId import get_rectified_xs
from linNonlinMPC import get_model
from reacFuncs import get_plant_pars, plant_ode
from reacFuncs import get_hyb_pars

def gen_train_val_data(*, parameters, Nthrow, num_traj,
                          Nsim_train, Nsim_trainval, Nsim_val, seed):
    """ Generate data for training and validation. """

    # Create a list to store data and get some parameters.
    data_list = []
    ulb, uub = parameters['ulb'], parameters['uub']
    np.random.seed(seed) # (So that the first us is reproducible).

    # Steady state disturbance.
    Np = parameters['Np']
    ps = parameters['ps'][:, np.newaxis]

    # Start to generate data.
    for traj in range(num_traj):
        
        # Get the plant and initial steady input.
        plant = get_model(ode=plant_ode, parameters=parameters, plant=True)
        us_init = np.tile(np.random.uniform(ulb, uub), (tthrow, 1))
        
        # Get input trajectories for different simulations.
        if traj == num_traj-1:
            " Get input for train val simulation. "
            Nsim = Nsim_val
            u = sample_prbs_like(num_change=9, num_steps=Nsim_val, 
                                 lb=ulb, ub=uub,
                                 mean_change=40, sigma_change=5,
                                 num_constraint=2, seed=seed+1)
        elif traj == num_traj-2:
            " Get input for validation simulation. "
            Nsim = Nsim_trainval
            u = sample_prbs_like(num_change=6, num_steps=Nsim_trainval, 
                                 lb=ulb, ub=uub,
                                 mean_change=40, sigma_change=5, 
                                 num_constraint=2, seed=seed+4)
        else:
            " Get input for training simulation. "
            Nsim = Nsim_train
            u = sample_prbs_like(num_change=6, num_steps=Nsim_train, 
                                 lb=ulb, ub=uub,
                                 mean_change=40, sigma_change=5, 
                                 num_constraint=2, seed=seed+7)
        
        # Get the disturbance signal.
        p = np.tile(ps.T, (tthrow + Nsim, Np))
        seed += 1

        # Complete input profile and run open-loop simulation.
        u = np.concatenate((us_init, u), axis=0)
        for t in range(tthrow + Nsim):
            plant.step(u[t:t+1, :], p[t:t+1, :])

        # Create a simdata.
        data_list.append(SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                                x=np.asarray(plant.x[0:-1]).squeeze(),
                                u=np.asarray(plant.u).squeeze(axis=-1),
                                y=np.asarray(plant.y[0:-1]).squeeze(), 
                                p=np.asarray(plant.p).squeeze(axis=-1)))

    # Return.
    return data_list

def main():
    """ Get the parameters, training, and validation data."""
    
    # Get parameters.
    plant_pars = get_plant_pars()
    plant_pars['xs'] = get_rectified_xs(ode=plant_ode,
                                        parameters=plant_pars)
    
    # Grey-Box model parameters.
    hyb_fullgb_pars = get_hyb_pars(plant_pars=plant_pars, Nx=3)
    hyb_partialgb_pars = get_hyb_pars(plant_pars=plant_pars, Nx=2)
    
    # Generate training data.
    training_data_dyn = gen_train_val_data(parameters=plant_pars,
                                            num_traj=6, Nsim_train=240,
                                            Nsim_trainval=240, Nsim_val=360,
                                            seed=0)

    # Get the dictionary.
    reac_parameters = dict(plant_pars = plant_pars,
                           training_data_dyn = training_data_dyn,
                           hyb_fullgb_pars=hyb_fullgb_pars,
                           hyb_partialgb_pars=hyb_partialgb_pars)
    
    # Save data.
    PickleTool.save(data_object=reac_parameters,
                    filename='reac_parameters.pickle')

main()