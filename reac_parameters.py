# [depends] %LIB%/hybridId.py %LIB%/reacFuncs.py
# [depends] %LIB%/linNonlinMPC.py
# [makes] pickle
import sys
sys.path.append('lib/')
import random
import numpy as np
from hybridId import PickleTool, sample_prbs_like, SimData
from hybridId import get_rectified_xs
from linNonlinMPC import get_plant_model
from reacFuncs import get_plant_pars, plant_ode

# Numpy seed.
np.random.seed(12)

# Random package (Used to sample distinct mixed-integers) seed.
random.seed(6)

def get_known_hyb_pars(*, plant_pars, hybtype=None):
    """ Grey-Box parameters for the hybrid model. """
    
    # Parameters.
    parameters = {}

    # Volume, steady-state disturbance, and sample time.
    parameters['V'] = plant_pars['V']
    parameters['ps'] = plant_pars['ps']
    parameters['Delta'] = plant_pars['Delta']

    # Number of states based on model type.
    if hybtype == 'fullgb':
        Ng = 3
    elif hybtype == 'partialgb':
        Ng = 2
    else:
        raise ValueError("Model type not present")

    # Sizes.
    parameters['Ng'] = Ng
    parameters['Nu'] = plant_pars['Nu']
    parameters['Ny'] = plant_pars['Ny']
    parameters['Np'] = plant_pars['Np']

    # Return.
    return parameters

def gen_train_val_data(*, parameters, Ntstart, num_traj,
                          Nsim_train, Nsim_trainval, Nsim_val, 
                          x0lb, x0ub):
    """ Generate data for training and validation. """

    # Sizes.
    Nx, Np = parameters['Nx'], parameters['Np']

    # Input constraint limits.
    ulb, uub = parameters['ulb'], parameters['uub']

    # List to store simdata objects.
    data_list = []

    # Steady-state disturbance.
    ps = parameters['ps'][:, np.newaxis]

    # Loop over the number of trajectories.
    for traj in range(num_traj):
        
        # Get a random initial state.
        x0 = (x0ub - x0lb)*np.random.rand(Nx, 1) + x0lb

        # Get a plant simulator object.
        plant = get_plant_model(ode=plant_ode, parameters=parameters, x0=x0)
        
        # Get input trajectories for different simulations.
        if traj == num_traj-1:

            " Generate useq for validation simulation. "
            Nsim = Ntstart + Nsim_val
            u = sample_prbs_like(num_change=9, num_steps=Nsim, 
                                 lb=ulb, ub=uub, mean_change=40, 
                                 sigma_change=5, num_constraint=2)

        elif traj == num_traj-2:

            " Generate useq for train val simulation. "
            Nsim = Ntstart + Nsim_trainval
            u = sample_prbs_like(num_change=6, num_steps=Nsim, 
                                 lb=ulb, ub=uub, mean_change=40, 
                                 sigma_change=5, num_constraint=2)

        else:

            " Generate useq for training simulation. "
            Nsim = Ntstart + Nsim_train
            u = sample_prbs_like(num_change=6, num_steps=Nsim,
                                 lb=ulb, ub=uub, mean_change=40, 
                                 sigma_change=5, num_constraint=2)
        
        # Create the steady-state disturbance signal.
        # Change this later if we need to do simulations
        # with varying unmeasured disturbance signals.
        p = np.tile(ps.T, (Nsim, Np))

        # Run an open-loop simulation based on u and p.
        for t in range(Nsim):
            plant.step(u[t:t+1, :], p[t:t+1, :])

        # Create a simdata object.
        simdata = SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                          x=np.asarray(plant.x[0:-1]).squeeze(axis=-1),
                          u=np.asarray(plant.u).squeeze(axis=-1),
                          y=np.asarray(plant.y[0:-1]).squeeze(axis=-1), 
                          p=np.asarray(plant.p).squeeze(axis=-1))

        # Append data to a list.
        data_list += [simdata]

    # Return.
    return data_list

def get_training_data(*, plant_pars, Ntstart):
    """ Generate training data. """

    # Sizes.
    Nx, Ny = plant_pars['Nx'], plant_pars['Ny']

    # Number of simulation trajectories and steps.
    num_traj = 4
    Nsim_train = 240
    Nsim_trainval = 240
    Nsim_val = 360

    # Noise covariance. 
    Rv = plant_pars['Rv']

    # Range of initial conditions.
    x0lb = np.array([0.2, 0.3, 0.1])[:, np.newaxis]
    x0ub = np.array([0.7, 0.6, 0.6])[:, np.newaxis]

    # Get data without noise. 
    plant_pars['Rv'] = 0*np.eye(Ny)
    training_data_nonoise = gen_train_val_data(parameters=plant_pars, 
                                               Ntstart=Ntstart,
                                               num_traj=num_traj, 
                                               Nsim_train=Nsim_train,
                                               Nsim_trainval=Nsim_trainval, 
                                               Nsim_val=Nsim_val,
                                               x0lb=x0lb, x0ub=x0ub)

    # Get data with noise. 
    plant_pars['Rv'] = Rv
    training_data_withnoise = gen_train_val_data(parameters=plant_pars, 
                                                 Ntstart=Ntstart,
                                                 num_traj=num_traj, 
                                                 Nsim_train=Nsim_train,
                                                 Nsim_trainval=Nsim_trainval, 
                                                 Nsim_val=Nsim_val,
                                                 x0lb=x0lb, x0ub=x0ub)
    
    # Return.
    return training_data_nonoise, training_data_withnoise

def main():
    """ Get the parameters, training, and validation data."""
    
    # Get parameters.
    plant_pars = get_plant_pars()
    plant_pars['xs'] = get_rectified_xs(ode=plant_ode,
                                        parameters=plant_pars)

    # Grey-Box model parameters.
    hyb_fullgb_pars = get_known_hyb_pars(plant_pars=plant_pars,
                                         hybtype='fullgb')
    hyb_partialgb_pars = get_known_hyb_pars(plant_pars=plant_pars,
                                            hybtype='partialgb')
    
    # Get training data. 
    Ntstart = 2
    (training_data_nonoise, 
     training_data_withnoise) = get_training_data(plant_pars=plant_pars, 
                                                  Ntstart=Ntstart)

    # Get a dictionary to return.
    reac_parameters = dict(plant_pars=plant_pars, Ntstart=Ntstart,
                           training_data_nonoise=training_data_nonoise,
                           training_data_withnoise=training_data_withnoise,
                           hyb_fullgb_pars=hyb_fullgb_pars,
                           hyb_partialgb_pars=hyb_partialgb_pars)
    
    # Save data.
    PickleTool.save(data_object=reac_parameters,
                    filename='reac_parameters.pickle')

main()