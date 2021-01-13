# [depends] %LIB%/hybridid.py %LIB%/HybridModelLayers.py
# [depends] tworeac_parameters_nonlin.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import PickleTool, SimData, get_tworeac_train_val_data
from HybridModelLayers import TwoReacModel

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def create_tworeac_model(*, Np, fnn_dims, xuscales, 
                            tworeac_parameters, model_type):
    """ Create/compile the two reaction model for training. """
    tworeac_model = TwoReacModel(Np=Np,
                                 fnn_dims=fnn_dims,
                                 xuscales=xuscales,
                                 tworeac_parameters=tworeac_parameters, 
                                 model_type=model_type)
    # Compile the nn controller.
    tworeac_model.compile(optimizer='adam', 
                          loss='mean_squared_error')
    # Return the compiled model.
    return tworeac_model

def train_model(model, train_data, trainval_data, val_data,
                xuscales, stdout_filename, ckpt_path):
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
    model.fit(x=[train_data['inputs'], train_data['xGz0']], 
              y=train_data['outputs'], 
              epochs=1000, batch_size=1,
        validation_data = ([trainval_data['inputs'], trainval_data['xGz0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

    # Get predictions on validation data.
    model.load_weights(ckpt_path)
    model_predictions = model.predict(x=[val_data['inputs'], val_data['xGz0']])
    xmean, xstd = xuscales['xscale']
    umean, ustd = xuscales['uscale']
    ypredictions = model_predictions.squeeze()*xstd + xmean
    xpredictions = np.insert(ypredictions, [2], 
                             np.nan*np.ones((ypredictions.shape[0], 1)), axis=1)
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean
    val_predictions = SimData(t=np.arange(0, uval.shape[0], 1), 
                              x=xpredictions, u=uval,
                              y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['xGz0']], 
                               y=val_data['outputs'])

    # Return the trained model/predictions/metrics.
    return (model, val_predictions, val_metric)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_nonlin.pickle',
                                         type='read')
    parameters = tworeac_parameters['parameters']
    training_data = tworeac_parameters['training_data']
    
    # Number of samples.
    num_samples = [hour*60 for hour in [6]]

    # Create lists.
    Nps = [2, 2]
    fnn_dims = [[9, 16, 2], [8, 16, 2]]
    model_types = ['black-box', 'hybrid']
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_train_nonlin.ckpt'
    stdout_filename = 'tworeac_train_nonlin.txt'

    # Loop over the model choices.
    for (model_type, fnn_dim, Np) in zip(model_types, fnn_dims, Nps):
        
        model_trained_weights = []
        model_val_metrics = []

        # Get the training data.
        (train_data, 
         trainval_data, 
         val_data, xuscales) = get_tworeac_train_val_data(Np=Np,
                                                parameters=parameters, 
                                                data_list=training_data)

        # Loop over the number of samples.
        for num_sample in num_samples:
            
            # Create model.
            tworeac_model = create_tworeac_model(Np=Np, fnn_dims=fnn_dim,
                                                 xuscales=xuscales,
                                                 tworeac_parameters=parameters, 
                                                 model_type=model_type)
            train_samples = dict(xGz0=train_data['xGz0'],
                            inputs=train_data['inputs'][:, :num_sample, :],
                            outputs=train_data['outputs'][:, :num_sample, :])
            (tworeac_model, 
             val_prediction, 
             val_metric) = train_model(tworeac_model, 
                                        train_samples, trainval_data, val_data,
                                        xuscales, stdout_filename, ckpt_path)
            fnn_weights = tworeac_model.get_weights()

            # Save info.
            model_trained_weights.append(fnn_weights)
            model_val_metrics.append(val_metric)

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(np.asarray(model_val_metrics))
        trained_weights.append(model_trained_weights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    tworeac_training_data = dict(Nps=Nps,
                                 model_types=model_types,
                                 fnn_dims=fnn_dims,
                                 trained_weights=trained_weights,
                                 val_predictions=val_predictions,
                                 val_metrics=val_metrics,
                                 num_samples=num_samples, 
                                 xuscales=xuscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_training_data, 
                    filename='tworeac_train_nonlin.pickle')

main()