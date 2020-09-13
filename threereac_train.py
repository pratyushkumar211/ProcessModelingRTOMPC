""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
from hybridid import PickleTool
from HybridModelLayers import get_threereac_model
from threereac_parameters import _get_threereac_parameters

def create_hybrid_model(*, threereac_parameters):
    """ Create/compile the hybrid model for training. """
    bb_dims = [5, 16, 5]
    hybrid_model = get_threereac_model(threereac_parameters=
                                       threereac_parameters, 
                                       bb_dims=bb_dims)
    # Compile the nn controller.
    hybrid_model.compile(optimizer='adam', loss='mean_squared_error')
    breakpoint()
    # Return the compiled model.
    return hybrid_model

def get_data_for_training():
    """ Get the data for training. """


    return 

def train_nn_controller(hybrid_model, data, 
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
    nn_controller.fit(x=[data['x'], data['uprev'], data['xs'], data['us']], 
                      y=[data['u']], 
                      epochs=500,
                      batch_size=1,
                      validation_split=0.1,
                      callbacks = [checkpoint_callback])
    tend = time.time()
    training_time = tend - tstart
    # Return the NN controller.
    return (nn_controller, training_time)

def main():
    """ Main function to be executed. """
    threereac_parameters = PickleTool.load(filename=
                                    'threereac_parameters.pickle',
                                    type='read')
    
    create_hybrid_model(threereac_parameters=threereac_parameters['parameters'])

main()

