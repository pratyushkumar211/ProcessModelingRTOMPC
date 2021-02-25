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
from hybridid import PickleTool, SimData
from tworeac_nonlin_funcs import get_tworeac_train_val_data
from HybridModelLayers import KoopmanModel

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def create_model(*, Np, fnn_dims, xuscales,
                    tworeac_parameters):
    """ Create/compile the two reaction model for training. """

    Ny, Nu = tworeac_parameters['Ny'], tworeac_parameters['Nu']
    model = KoopmanModel(Np=Np, Ny=Ny, Nu=Nu, fnn_dims=fnn_dims)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  loss_weights = [1., 0.])

    # Return the compiled model.
    return model

def train_model(model, train_data, trainval_data, 
                stdout_filename, ckpt_path):
    """ Function to train the Koopman model."""
    # Std out.
    sys.stdout = open(stdout_filename, 'w')

    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Call the fit method to train.
    model.fit(x=[train_data['inputs'], train_data['xGz0']], 
              y=[train_data['yz'], train_data['outputs']], 
              epochs=3000, batch_size=1,
        validation_data = ([trainval_data['inputs'], trainval_data['xGz0']], 
                            [trainval_data['yz'], trainval_data['outputs']]),
            callbacks = [checkpoint_callback])

    # Return.
    return 

def get_val_predictions(model, val_data, xuscales, ckpt_path):
    """ Get validation predictions and metric. """

    model.load_weights(ckpt_path)
    model_predictions = model.predict(x=[val_data['inputs'], val_data['xGz0']])
    xmean, xstd = xuscales['xscale']
    umean, ustd = xuscales['uscale']
    ypredictions = model_predictions[1].squeeze()*xstd + xmean
    xpredictions = ypredictions
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean
    val_predictions = SimData(t=np.arange(0, uval.shape[0], 1), 
                              x=xpredictions, u=uval,
                              y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['xGz0']], 
                                y=[val_data['yz'], val_data['outputs']])

    # Return validation predictions and metric.
    return (val_predictions, val_metric)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_nonlin.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    training_data = tworeac_parameters['training_data']
    
    # Number of samples.
    num_samples = [hours*60 for hours in [6]]

    # Create lists.
    Np = 0
    fnn_dims = [3, 32, 64]
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_kooptrain_nonlin.ckpt'
    stdout_filename = 'tworeac_kooptrain_nonlin.txt'

    # Get the training data.
    (train_data, 
     trainval_data, 
     val_data, xuscales) = get_tworeac_train_val_data(Np=Np,
                                                parameters=parameters, 
                                                data_list=training_data)

    # Loop over the number of samples.
    for num_sample in num_samples:
            
        # Create model.
        tworeac_model = create_model(Np=Np, fnn_dims=fnn_dims,
                                     xuscales=xuscales,
                                     tworeac_parameters=parameters)
        train_samples = dict(xGz0=train_data['xGz0'],
                             inputs=train_data['inputs'],
                             yz=train_data['yz'],
                             outputs=train_data['outputs'])

        # Train.
        train_model(tworeac_model, train_samples, trainval_data,
                    stdout_filename, ckpt_path)

        # Validate.
        (val_prediction, 
         val_metric) = get_val_predictions(tworeac_model, val_data, 
                                               xuscales, ckpt_path)

        # Get weights.
        fnn_weights = tworeac_model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        trained_weights.append(fnn_weights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    tworeac_training_data = dict(Np=Np,
                                 fnn_dims=fnn_dims,
                                 trained_weights=trained_weights,
                                 val_predictions=val_predictions,
                                 val_metrics=val_metrics,
                                 num_samples=num_samples,
                                 xuscales=xuscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_training_data,
                    filename='tworeac_kooptrain_nonlin.pickle')

main()