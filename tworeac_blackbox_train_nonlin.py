# [depends] %LIB%/hybridid.py %LIB%/HybridModelLayers.py
# [depends] %LIB%/../tworeac_nonlin_funcs.py
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
from hybridid import PickleTool, SimData, get_scaling
from hybridid import get_train_val_data
from HybridModelLayers import TwoReacModel

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def create_model(*, Np, fnn_dims, xuyscales, tworeac_parameters):
    """ Create/compile the two reaction model for training. """
    model = TwoReacModel(Np=Np, fnn_dims=fnn_dims,
                         xuyscales=xuyscales,
                         tworeac_parameters=tworeac_parameters, 
                         model_type='black-box')
    # Compile the nn controller.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def train_model(model, train_data, trainval_data, stdout_filename, ckpt_path):
    """ Function to train the NN controller. """
    # Std out.
    sys.stdout = open(stdout_filename, 'w')
    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    # Call the fit method to train.
    model.fit(x=[train_data['inputs'], train_data['yz0']], 
              y=train_data['outputs'], 
              epochs=3000, batch_size=2,
        validation_data = ([trainval_data['inputs'], trainval_data['yz0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

def get_val_predictions(model, val_data, xuyscales, ckpt_path):
    """ Get the validation predictions. """

    # Get predictions on validation data.
    model.load_weights(ckpt_path)
    model_predictions = model.predict(x=[val_data['inputs'], val_data['yz0']])
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    ypredictions = model_predictions.squeeze()*ystd + ymean
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean
    xpredictions = np.insert(ypredictions, [2], np.nan, axis=1)
    Nt = uval.shape[0]
    val_predictions = SimData(t=np.arange(0, Nt, 1), 
                              x=xpredictions, u=uval,
                              y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['yz0']], 
                                y=val_data['outputs'])

    # Return predictions and metric.
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
    Np = 2
    fnn_dims = [8, 16, 2]
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_blackbox_train_nonlin.ckpt'
    stdout_filename = 'tworeac_blackbox_train_nonlin.txt'

    # Get scaling and the training data.
    xuyscales = get_scaling(data=training_data[0])
    (train_data, trainval_data, val_data) = get_train_val_data(Np=Np,
                                                xuyscales=xuyscales,
                                                parameters=parameters,
                                                data_list=training_data)
    # Loop over the number of samples.
    for num_sample in num_samples:
            
        # Create model.
        tworeac_model = create_model(Np=Np, fnn_dims=fnn_dims,
                                     xuyscales=xuyscales,
                                     tworeac_parameters=parameters)
        
        # Use num samples to adjust here the num training samples.
        train_samples = dict(yz0=train_data['yz0'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])

        # Train.
        train_model(tworeac_model, train_samples, trainval_data, 
                    stdout_filename, ckpt_path)

        # Validate.
        (val_prediction, val_metric) = get_val_predictions(tworeac_model, 
                                                           val_data, 
                                                           xuyscales, ckpt_path)

        # Get weights to store.
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
                                 xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_training_data,
                    filename='tworeac_blackbox_train_nonlin.pickle')

main()