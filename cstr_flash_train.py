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
                             loss='mean_squared_error', 
                             loss_weights=[1., 0.])
    # Return the compiled model.
    return cstr_flash_model

def train_model(model, x0key, xuyscales, train_data, trainval_data, val_data,
                stdout_filename, ckpt_path):
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
    model.fit(x = [train_data['inputs'], train_data[x0key]],
              y = [train_data['outputs'], train_data['xG']], 
              epochs=300, batch_size=2,
        validation_data = ([trainval_data['inputs'], trainval_data[x0key]], 
                           [trainval_data['outputs'], trainval_data['xG']]),
        callbacks = [checkpoint_callback])

    # Load best weights.
    model.load_weights(ckpt_path)

    # Get predictions on validation data and scale back to physical variables.
    model_predictions = model.predict(x=[val_data['inputs'], val_data[x0key]])
    ymean, ystd = xuyscales['yscale']
    xmean, xstd = xuyscales['xscale']
    ypredictions = model_predictions[0].squeeze()*ystd + ymean
    xpredictions = model_predictions[1].squeeze()*xstd + xmean
    xpredictions = np.insert(xpredictions, [3, 7], 
                             np.nan*np.ones((xpredictions.shape[0], 2)), axis=1)
    val_predictions = SimData(t=None, x=xpredictions, u=None,
                              y=ypredictions)

    # Evaluate metric.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data[x0key]],
                                y = [val_data['outputs'], val_data['xG']])

    # Return model/predictions/metric.
    return (model, val_predictions, val_metric)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                            'cstr_flash_parameters.pickle',
                                            type='read')
    (greybox_pars,
     greybox_processed_data) = (cstr_flash_parameters['greybox_pars'],
                                cstr_flash_parameters['greybox_processed_data'])

    # Number of samples.
    num_train_traj = len(greybox_processed_data) - 2
    num_batches = [num_train_traj]
    Nsim_train = greybox_processed_data[0].x.shape[0] 
    Nsim_train -= greybox_pars['tsteps_steady']
    num_samples = [batch*Nsim_train for batch in num_batches]

    # Create lists.
    Nps = [9]
    fnn_dims = [[102, 32, 32, 32, 8]]
    model_types = ['hybrid']
    trained_weights = []
    val_metrics = []
    val_predictions = []
    
    # Filenames.
    ckpt_path = 'cstr_flash_train.ckpt'
    stdout_filename = 'cstr_flash_train.txt'

    # Loop over the model choices.
    for (model_type, fnn_dim, Np) in zip(model_types, fnn_dims, Nps):
        
        model_trained_weights = []
        model_val_metrics = []

        # Get the training data.
        (train_data,
         trainval_data,
         val_data, xuyscales) = get_cstr_flash_train_val_data(Np=Np,
                                parameters=greybox_pars,
                                greybox_processed_data=greybox_processed_data)

        # Loop over the number of samples.
        for num_batch in num_batches:
            
            # Create model.
            cstr_flash_model = create_model(Np=Np, fnn_dims=fnn_dim,
                                            xuyscales=xuyscales,
                                            cstr_flash_parameters=greybox_pars,
                                            model_type=model_type)

            # Get the training samples.
            if model_type == 'black-box':
                x0key = 'yz0'
            else:
                x0key = 'xGz0'
            train_samples=dict(inputs=train_data['inputs'][:num_batch, ...],
                               outputs=train_data['outputs'][:num_batch, ...], 
                               xG=train_data['xG'][:num_batch, ...])
            train_samples[x0key] = train_data[x0key][:num_batch, :]
            (cstr_flash_model,
             val_prediction,
             val_metric) = train_model(cstr_flash_model, x0key, xuyscales,
                                       train_samples, trainval_data, val_data,
                                       stdout_filename, ckpt_path)
            fnn_weights = cstr_flash_model.get_weights()

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
    cstr_flash_training_data = dict(Nps=Nps,
                                    model_types=model_types,
                                    fnn_dims=fnn_dims,
                                    trained_weights=trained_weights,
                                    val_predictions=val_predictions,
                                    val_metrics=val_metrics,
                                    num_samples=num_samples,
                                    xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=cstr_flash_training_data,
                    filename='cstr_flash_train.pickle')
    
main()