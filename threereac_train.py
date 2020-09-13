""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import tensorflow as tf
import numpy as np
from hybridid import PickleTool
from HybridModelLayers import get_threereac_model
from threereac_parameters import _get_threereac_parameters

def create_hybrid_model(*, threereac_parameters):
    """ Create/compile the hybrid model for training."""
    bb_dims = [9, 16, 5]
    hybrid_model = get_threereac_model(threereac_parameters=
                                       threereac_parameters, 
                                       bb_dims=bb_dims)
    # Compile the nn controller.
    hybrid_model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return hybrid_model

def get_data_for_training(*, train_val_datum):
    """ Get the data for training. """
    train_data = dict(u=train_val_datum[0].Ca0[np.newaxis, :, np.newaxis],
                      y=train_val_datum[0].Cc[np.newaxis, :, np.newaxis])
    trainval_data = dict(u=train_val_datum[1].Ca0[np.newaxis, :, np.newaxis],
                         y=train_val_datum[1].Cc[np.newaxis, :, np.newaxis])
    #val_data = dict(u=train_val_datum[2].Ca0[np.newaxis, :, np.newaxis],
    #            y=train_val_datum[2].Cc[np.newaxis, :, np.newaxis])
    return (train_data, trainval_data)

def train_nn_controller(hybrid_model, train_data, trainval_data, 
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
    hybrid_model.fit(x=[train_data['u']], 
                      y=train_data['y'], 
                epochs=10, batch_size=1,
                validation_data = ([trainval_data['u']], trainval_data['y']),
                callbacks = [checkpoint_callback])
    tend = time.time()
    training_time = tend - tstart
    # Return the NN controller.
    return (hybrid_model, training_time)

def main():
    """ Main function to be executed. """
    # Load data.
    threereac_parameters = PickleTool.load(filename=
                                    'threereac_parameters.pickle',
                                    type='read')
    # Create the hybrid model.
    hybrid_model = create_hybrid_model(threereac_parameters=
                                       threereac_parameters['parameters'])
    # Get the training data.
    (train_data, trainval_data) = get_data_for_training(train_val_datum=
                                    threereac_parameters['train_val_datum'])
    (hybrid_model, training_time) = train_nn_controller(hybrid_model, 
                                             train_data, trainval_data, 
                                            'threereac_train.txt', 
                                            'threereac_train.ckpt')
    hybrid_model.load_weights('threereac_train.ckpt')
    breakpoint()
    print("Hi")

main()