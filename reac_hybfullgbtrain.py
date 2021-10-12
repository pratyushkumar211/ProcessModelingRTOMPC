# [depends] %LIB%/hybridId.py %LIB%/ReacHybridFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_train_val_data
from BlackBoxFuncs import get_weights_from_tflayers
from ReacHybridFullGbFuncs import (create_model, train_model, 
                                   get_val_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """
    
    # Load data.
    reac_parameters = PickleTool.load(filename=
                                      'reac_parameters.pickle',
                                      type='read')

    # Sample time and parameters.
    Delta = reac_parameters['plant_pars']['Delta']
    hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']

    # Sizes.
    Nu = hyb_fullgb_pars['Nu']
    Ny = hyb_fullgb_pars['Ny']

    # Raw training data.
    training_data_list = [reac_parameters['training_data_nonoise'], 
                          reac_parameters['training_data_withnoise']]

    # Create some parameters.
    Ntstart = reac_parameters['Ntstart']
    Np = reac_parameters['Ntstart']
    r1Dims = [1, 8, 1]
    r2Dims = [2, 8, 1]
    estC0Dims = [Np*(Ny + Nu), 4, 1]

    # Filenames.
    ckpt_path = 'reac_hybfullgbtrain.ckpt'
    stdout_filename = 'reac_hybfullgbtrain.txt'

    # Train on both the types of training data, create lists to store.
    reac_train_list = []

    # Loop over both types of training data. 
    for training_data in training_data_list:

        # Get scaling.
        xuyscales = get_scaling(data=training_data[0])
        ymean, ystd = xuyscales['yscale']
        xmean, xstd = xuyscales['xscale']
        xmean = np.concatenate((ymean, ymean[-1:])) # (Update Cc_mean)
        xstd = np.concatenate((ystd, ystd[-1:])) # (Update Cc_std)
        xuyscales['xscale'] = (xmean, xstd)
        
        # Get scaling and the training data.
        (train_data, 
        trainval_data, val_data) = get_train_val_data(Ntstart=Ntstart, Np=Np,
                                                    xuyscales=xuyscales,
                                                    data_list=training_data)

        # Create model.
        model = create_model(r1Dims=r1Dims, r2Dims=r2Dims, 
                            estC0Dims=estC0Dims, Np=Np, 
                            xuyscales=xuyscales, 
                            hyb_fullgb_pars=hyb_fullgb_pars)

        # Train.
        train_model(model=model, epochs=10, batch_size=1, 
                        train_data=train_data, trainval_data=trainval_data,
                        stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        val_predictions = get_val_predictions(model=model,
                                            val_data=val_data, xuyscales=xuyscales,
                                            ckpt_path=ckpt_path, Delta=Delta)

        # Get weights to store.
        r1Weights = get_weights_from_tflayers(model.r1Layers)
        r2Weights = get_weights_from_tflayers(model.r2Layers)
        estC0Weights = get_weights_from_tflayers(model.estC0Layers)

        # Save the weights.
        reac_train = dict(Np=Np, r1Dims=r1Dims, r2Dims=r2Dims, 
                        estC0Dims=estC0Dims, r1Weights=r1Weights,
                        r2Weights=r2Weights, estC0Weights=estC0Weights,
                        val_predictions=val_predictions,
                        xuyscales=xuyscales)
        
        # Store the list into dictionaries. 
        reac_train_list += [reac_train]

    # Save data.
    PickleTool.save(data_object=reac_train_list,
                    filename='reac_hybfullgbtrain.pickle')

main()