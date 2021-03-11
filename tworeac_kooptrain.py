# [depends] %LIB%/hybridid.py %LIB%/HybridModelLayers.py
# [depends] %LIB%/../tworeac_nonlin_funcs.py
# [depends] tworeac_parameters_nonlin.pickle
# [makes] pickle
""" Script to train the deep Koopman model for the
    three reaction system.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import PickleTool, get_scaling, get_train_val_data
from training_funcs import create_koopmodel, train_koopmodel
from training_funcs import get_koopval_predictions

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')

    # Get sizes/raw training data.
    parameters = tworeac_parameters['parameters']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    training_data = tworeac_parameters['training_data']
    
    # Number of samples.
    num_samples = [hours*60 for hours in [6]]

    # Create some parameters.
    xinsert_indices = [2]
    tthrow = 10
    Np = 2
    fN_dims = [Np*(Ny+Nu), 16, 2]

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_kooptrain.ckpt'
    stdout_filename = 'tworeac_kooptrain.txt'

    # Get scaling and the training data.
    xuyscales = get_scaling(data=training_data[0])
    (train_data, trainval_data, val_data) = get_train_val_data(tthrow=tthrow, 
                                            Np=Np, xuyscales=xuyscales, 
                                            data_list=training_data)

    # Loop over the number of samples.
    for num_sample in num_samples:
        
        # Create model.
        model = create_koopmodel(Np=Np, Ny=Ny, Nu=Nu, fN_dims=fN_dims)
        
        # Use num samples to adjust here the num training samples.
        train_samples = dict(yz0=train_data['yz0'],
                             yz=train_data['yz'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])

        # Train.
        train_koopmodel(model=model, epochs=100, batch_size=3, 
                      train_data=train_samples, trainval_data=trainval_data, 
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        (val_prediction, val_metric) = get_koopval_predictions(model=model,
                                    val_data=val_data, xuyscales=xuyscales, 
                                    xinsert_indices=xinsert_indices, 
                                    ckpt_path=ckpt_path)

        # Get weights to store.
        fN_weights = model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        trained_weights.append(fN_weights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    tworeac_training_data = dict(Np=Np,
                                 fN_dims=fN_dims,
                                 trained_weights=trained_weights,
                                 val_predictions=val_predictions,
                                 val_metrics=val_metrics,
                                 num_samples=num_samples,
                                 xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_training_data,
                    filename='tworeac_kooptrain.pickle')

main()