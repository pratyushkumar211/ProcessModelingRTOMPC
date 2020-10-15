# [depends] %LIB%/hybridid.py tworeac_parameters_lin.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import (PickleTool, get_tworeac_train_val_data, 
                      SimData)
from HybridModelLayers import TwoReacModel

# Set the tensorflow graph-level seed.
tf.random.set_seed(1)

def create_tworeac_model(*, Np, fnn_dims, tworeac_parameters):
    """ Create/compile the two reaction model for training. """
    tworeac_model = TwoReacModel(Np=Np,
                                 fnn_dims=fnn_dims,
                                 tworeac_parameters=tworeac_parameters)
    # Compile the nn controller.
    tworeac_model.compile(optimizer='adam', 
                          loss='mean_squared_error')
    # Return the compiled model.
    return tworeac_model

def train_model(model, train_data, trainval_data, val_data,
                stdout_filename, ckpt_path):
    """ Function to train the NN controller."""
    # Std out.
    sys.stdout = open(stdout_filename, 'w')
    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    # Call the fit method to train.
    tstart = time.time()
    model.fit(x=[train_data['inputs'], train_data['x0']], 
              y=train_data['outputs'], 
            epochs=1000, batch_size=1,
            validation_data = ([trainval_data['inputs'], trainval_data['x0']], 
                                trainval_data['outputs']),
            callbacks = [checkpoint_callback])
    tend = time.time()
    training_time = tend - tstart
    
    # Get predictions on validation data.
    model.load_weights(ckpt_path)
    model_predictions = model.predict(x=[val_data['inputs'], val_data['x0']])
    val_predictions = SimData(t=None, x=None, u=None,
                              y=model_predictions.squeeze())

    # Return the NN controller.
    return (model, training_time, val_predictions)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_lin.pickle',
                                         type='read')
    # Create the hybrid model.
    Np = 3
    fnn_dims = [8, 2]
    tworeac_model = create_tworeac_model(Np=Np, fnn_dims=fnn_dims,
                    tworeac_parameters=tworeac_parameters['parameters'])
    # Get the training data.
    (train_data, trainval_data, val_data) = get_tworeac_train_val_data(Np=Np,
                                parameters=tworeac_parameters['parameters'],
                                data_list=tworeac_parameters['training_data'])
    (tworeac_model, training_time, 
          val_predictions) = train_model(tworeac_model, 
                                         train_data, trainval_data, val_data,
                                         'tworeac_train_lin.txt', 
                                         'tworeac_train_lin.ckpt')
    fnn_weights = tworeac_model.get_weights()
    # Save the weights.
    tworeac_training_data = dict(fnn_weights=fnn_weights,
                                   val_predictions=val_predictions,
                                   training_time=training_time)
    # Save data.
    PickleTool.save(data_object=tworeac_training_data, 
                    filename='tworeac_train_lin.pickle')

main()