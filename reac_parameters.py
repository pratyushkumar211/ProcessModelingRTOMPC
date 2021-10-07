# [depends] %LIB%/hybridId.py %LIB%/reacFuncs.py
# [depends] %LIB%/linNonlinMPC.py
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from hybridId import PickleTool, sample_prbs_like, SimData
from hybridId import get_rectified_xs
from linNonlinMPC import get_plant_model
from reacFuncs import get_plant_pars, plant_ode
from reacFuncs import get_hyb_pars

def gen_train_val_data(*, parameters, Ntz, num_traj,
                          Nsim_train, Nsim_trainval, Nsim_val, seed):
    """ Generate data for training and validation. """

    # Numpy seed.
    np.random.seed(seed)

    # Sizes.
    Nx, Np = parameters['Nx'], parameters['Np']

    # Initial concentration limits.
    x0lb, x0ub = np.zeros((Nx, )), np.ones((Nx, ))

    # List to store simdata objects.
    data_list = []

    # Input constraint limits.
    ulb, uub = parameters['ulb'], parameters['uub']

    # Steady-state disturbance.
    ps = parameters['ps'][:, np.newaxis]

    # Loop over the number of trajectories.
    for traj in range(num_traj):
        
        # Get a random initial state.
        x0 = (x0ub - x0lb)*np.random.rand((Nx, 1)) + x0lb

        # Get a plant simulator object.
        plant = get_model(ode=plant_ode, parameters=parameters, x0=x0)
        
        # Get input trajectories for different simulations.
        if traj == num_traj-1:

            " Get input for train val simulation. "
            Nsim = Ntz + Nsim_val
            u = sample_prbs_like(num_change=9, num_steps=Nsim, 
                                 lb=ulb, ub=uub,
                                 mean_change=40, sigma_change=5)

        elif traj == num_traj-2:

            " Get input for validation simulation. "
            Nsim = Ntz + Nsim_trainval
            u = sample_prbs_like(num_change=6, num_steps=Nsim, 
                                 lb=ulb, ub=uub,
                                 mean_change=40, sigma_change=5)

        else:

            " Get input for training simulation. "
            Nsim = Ntz + Nsim_train
            u = sample_prbs_like(num_change=6, num_steps=Nsim,
                                 lb=ulb, ub=uub,
                                 mean_change=40, sigma_change=5)
        
        # Create the steady-state disturbance signal.
        p = np.tile(ps.T, (Nsim, Np))

        # Run open-loop simulation.
        for t in range(Nsim):
            plant.step(u[t:t+1, :], p[t:t+1, :])

        # Create a simdata object.
        simdata = SimData(t=np.asarray(plant.t[0:-1]).squeeze(axis=-1),
                            x=np.asarray(plant.x[0:-1]).squeeze(axis=-1),
                            u=np.asarray(plant.u).squeeze(axis=-1),
                            y=np.asarray(plant.y[0:-1]).squeeze(axis=-1), 
                            p=np.asarray(plant.p).squeeze(axis=-1))

        # Append data to a list.
        data_list += [simdata]

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