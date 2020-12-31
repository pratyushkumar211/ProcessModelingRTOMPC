# [depends] %LIB%/hybridid.py %LIB%/HybridModelLayers.py
# [depends] cstr_flash_parameters.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import (PickleTool, get_cstr_flash_train_val_data, 
                      SimData)
from HybridModelLayers import CstrFlashModel

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(2)

def create_model(*, Np, fnn_dims, xuyscales, cstr_flash_parameters, model_type):
    """ Create/compile the two reaction model for training. """
    cstr_flash_model = CstrFlashModel(Np=Np,
                                   fnn_dims=fnn_dims,
                                   xuyscales=xuyscales,
                                   cstr_flash_parameters=cstr_flash_parameters,
                                   model_type=model_type)
    # Compile the nn controller.
    cstr_flash_model.compile(optimizer='adam', 
                             loss='mean_squared_error')
    # Return the compiled model.
    return cstr_flash_model

def train_model(model, x0key, xuyscales, train_data, trainval_data, val_data,
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
    model.fit(x=[train_data['inputs'], train_data[x0key]],
              y=train_data['outputs'], 
              epochs=1000, batch_size=2,
        validation_data = ([trainval_data['inputs'], trainval_data[x0key]], 
                            trainval_data['outputs']),
        callbacks = [checkpoint_callback])

    # Get predictions on validation data.
    model.load_weights(ckpt_path)
    model_predictions = model.predict(x=[val_data['inputs'], val_data[x0key]])
    val_predictions = SimData(t=None, x=None, u=None,
                              y=model_predictions.squeeze()*xuyscales['yscale'])
    # Get prediction error on the validation data.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data[x0key]],
                                y = val_data['outputs'])
    # Return the NN controller.
    return (model, val_predictions, val_metric)

def main():
    """ Main function to be executed. """
    # Load data.
    
    # Save data.
    PickleTool.save(data_object=cstr_flash_training_data,
                    filename='cstr_flash_train.pickle')

main()