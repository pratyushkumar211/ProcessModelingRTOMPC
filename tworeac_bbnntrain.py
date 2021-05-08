# [depends] %LIB%/hybridid.py %LIB%/BlackBoxFuncs.py
# [depends] tworeac_parameters.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import PickleTool, get_scaling, get_train_val_data
from BlackBoxFuncs import (create_bbNNmodel, train_bbmodel, 
                           get_bbval_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')

    # Get sizes/raw training data.
    plant_pars = tworeac_parameters['plant_pars']
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    training_data = tworeac_parameters['training_data']
    
    # Number of samples.
    num_samples = [hours*60 for hours in [6]]

    # Create some parameters.
    xinsert_indices = []
    tthrow = 10
    Np = 0
    tanhScale = 1
    fNDims = [Ny + Np*(Ny+Nu), 64, Ny]

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_bbNNtrain.ckpt'
    stdout_filename = 'tworeac_bbNNtrain.txt'

    # Get scaling and the training data.
    xuyscales = get_scaling(data=training_data[0])
    (train_data, trainval_data, val_data) = get_train_val_data(tthrow=tthrow,
                                            Np=Np, xuyscales=xuyscales,
                                            data_list=training_data)

    # Loop over the number of samples.
    for num_sample in num_samples:
        
        # Create model.
        model = create_bbNNmodel(Np=Np, Ny=Ny, Nu=Nu, fNDims=fNDims, 
                                 tanhScale=tanhScale)
        
        # Use num samples to adjust here the num training samples.
        train_samples = dict(yz0=train_data['yz0'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])

        # Train.
        train_bbmodel(model=model, epochs=3000, batch_size=2, 
                      train_data=train_samples, trainval_data=trainval_data, 
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        (val_prediction, val_metric) = get_bbval_predictions(model=model,
                                        val_data=val_data, xuyscales=xuyscales, 
                                        xinsert_indices=xinsert_indices, 
                                        ckpt_path=ckpt_path)

        # Get weights to store.
        fNWeights = model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        trained_weights.append(fNWeights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    tworeac_train = dict(Np=Np, fNDims=fNDims,
                         trained_weights=trained_weights,
                         val_predictions=val_predictions,
                         val_metrics=val_metrics,
                         num_samples=num_samples,
                         xuyscales=xuyscales,
                         tanhScale=tanhScale)
    
    # Save data.
    PickleTool.save(data_object=tworeac_train,
                    filename='tworeac_bbNNtrain.pickle')

main()