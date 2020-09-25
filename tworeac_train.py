""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import time
import tensorflow as tf
import numpy as np
from hybridid import PickleTool
from HybridModelLayers import TwoReacModel

# Set the tensorflow graph-level seed.
tf.random.set_seed(1)

def create_tworeac_model(*, Np, fnn_dims, tworeac_parameters):
    """ Create/compile the two reaction model for training."""
    tworeac_model = TwoReacModel(Np=Np,
                                 fnn_dims=fnn_dims,
                                 tworeac_parameters=tworeac_parameters)
    # Compile the nn controller.
    tworeac_model.compile(optimizer='adam', 
                         loss='mean_squared_error')
    breakpoint()
    # Return the compiled model.
    return tworeac_model

#def get_data_for_training(*, train_val_datum, num_batches=32):
#    """ Get the data for training. """
#    inputs, outputs = [], []
#    for i in range(num_batches):
#        inputs += [train_val_datum[i].Ca0[np.newaxis, :, np.newaxis]]
#        outputs += [train_val_datum[i].Cc[np.newaxis, :, np.newaxis]]
#    train_data = dict(u=np.concatenate(inputs, axis=0),
#                      y=np.concatenate(outputs, axis=0))
#    trainval_data = dict(u=train_val_datum[32].Ca0[np.newaxis, :, np.newaxis],
#                         y=train_val_datum[32].Cc[np.newaxis, :, np.newaxis])
#    return (train_data, trainval_data)

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
                epochs=5, batch_size=32,
                validation_data = ([trainval_data['u']], trainval_data['y']),
                callbacks = [checkpoint_callback])
    breakpoint()
    tend = time.time()
    training_time = tend - tstart
    # Return the NN controller.
    return (hybrid_model, training_time)

#def get_prediction_data(hybrid_model, val_data, 
#                        train_val_datum):
#    """ Predict for plotting. """
#    pred = hybrid_model.predict(x=[val_data['u']]).squeeze()
#    return ModelSimData(time=train_val_datum[2].time, 
#                Ca=train_val_datum[2].Ca, Cc=pred, 
#                Cd=train_val_datum[2].Cd,
#                Ca0=train_val_datum[2].Ca0)

def main():
    """ Main function to be executed. """

    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')
    # Create the hybrid model.
    Np = 5
    fnn_dims = [14, 64, 64, 2]
    hybrid_model = create_tworeac_model(Np=Np, fnn_dims=fnn_dims,
                    tworeac_parameters=tworeac_parameters['parameters'])
    breakpoint()
    # Get the training data.
    (train_data, 
        trainval_data) = get_data_for_training(train_val_datum=
                                    threereac_parameters['train_val_datum'])
    (hybrid_model, training_time) = train_nn_controller(hybrid_model, 
                                             train_data, trainval_data, 
                                            'threereac_train.txt', 
                                            'threereac_train.ckpt')
    hybrid_model.load_weights('threereac_train.ckpt')
    bb_weights = hybrid_model.get_weights()
    #hybrid_pred = get_prediction_data(hybrid_model, val_data, 
    #                                  threereac_parameters['train_val_datum'])
    # Save the weights.
    threereac_training_data = dict(bb_weights=bb_weights,
                        training_time=training_time)
    # Save data.
    PickleTool.save(data_object=threereac_training_data, 
                    filename='threereac_train.pickle')

main()