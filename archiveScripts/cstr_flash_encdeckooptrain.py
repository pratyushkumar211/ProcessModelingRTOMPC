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
from hybridid import (PickleTool, SimData)
from HybridModelLayers import KoopmanEncDecModel
from hybridid import get_scaling, get_train_val_data

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(2)

def create_model(*, Np, enc_dims, dec_dims, cstr_flash_parameters):
    """ Create/compile the two reaction model for training. """

    Ny, Nu = cstr_flash_parameters['Ny'], cstr_flash_parameters['Nu']
    model = KoopmanEncDecModel(Np=Np, Ny=Ny, Nu=Nu,
                               enc_dims=enc_dims, dec_dims=dec_dims)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  loss_weights = [1., 0.])
    
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
    model.fit(x = [train_data['inputs'], train_data['yz0']],
              y = [train_data['yz'], train_data['outputs']],
              epochs=10000, batch_size=24,
        validation_data = ([trainval_data['inputs'], trainval_data['yz0']], 
                           [trainval_data['yz'], trainval_data['outputs']]),
        callbacks = [checkpoint_callback])

def get_val_predictions(model, val_data, xuyscales, ckpt_path):
    """ Get the validation predictions. """
    
    # Load best weights during training.
    model.load_weights(ckpt_path)
    
    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], 
                                         val_data['yz0']])
    ymean, ystd = xuyscales['yscale']
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    u = val_data['inputs'][0, ...]*ustd + umean
    t = np.arange(0, val_data['inputs'].shape[1], 1)
    ypredictions = model_predictions[1].squeeze()*ystd + ymean
    xpredictions = np.nan*np.ones((ypredictions.shape[0], 10))
    val_predictions = SimData(t=t, x=xpredictions, u=u, y=ypredictions)

    # Get the validation metric.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data['yz0']],
                                y = [val_data['yz'], val_data['outputs']])

    # Return predictions and metric.
    return (val_predictions, val_metric)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                            'cstr_flash_parameters.pickle',
                                            type='read')
    plant_pars = cstr_flash_parameters['plant_pars']
    training_data = cstr_flash_parameters['training_data']
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']

    # Number of samples.
    num_train_traj = len(training_data) - 2
    num_batches = [num_train_traj]
    Nsim_train = training_data[0].x.shape[0]
    Nsim_train -= plant_pars['tsteps_steady']
    num_samples = [batch*Nsim_train for batch in num_batches]

    # Create lists.
    Np = 6
    enc_dims = [Ny + Np*(Ny + Nu), 256, 64]
    dec_dims = [64, 256, Ny + Np*(Ny + Nu)]
    #trained_weights = []
    val_metrics = []
    val_predictions = []
    
    # Filenames.
    ckpt_path = 'cstr_flash_encdeckooptrain.ckpt'
    stdout_filename = 'cstr_flash_encdeckooptrain.txt'
    
    # Get the training data.
    xuyscales = get_scaling(data=training_data[0])
    (train_data, trainval_data, val_data) = get_train_val_data(Np=Np,
                                                xuyscales=xuyscales,
                                                parameters=plant_pars,
                                                data_list=training_data)

    # Loop over the number of samples.
    for num_batch in num_batches:
            
        # Create model.
        cstr_flash_model = create_model(Np=Np, enc_dims=enc_dims,
                                        dec_dims=dec_dims,
                                        cstr_flash_parameters=plant_pars)

        train_samples=dict(inputs=train_data['inputs'][:num_batch, ...],
                           outputs=train_data['outputs'][:num_batch, ...], 
                           yz=train_data['yz'][:num_batch, ...],
                           yz0=train_data['yz0'][:num_batch, ...])
        
        # Train.
        train_model(cstr_flash_model, train_samples, trainval_data,
                    stdout_filename, ckpt_path)

        # Validate.
        (val_prediction,
         val_metric) = get_val_predictions(cstr_flash_model, val_data,
                                           xuyscales, ckpt_path)

        # Load weights.
        #fnn_weights = cstr_flash_model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        #trained_weights.append(fnn_weights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    cstr_flash_training_data = dict(Np=Np,
                                    enc_dims=enc_dims,
                                    dec_dims=dec_dims,
                                    val_predictions=val_predictions,
                                    val_metrics=val_metrics,
                                    num_samples=num_samples,
                                    xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=cstr_flash_training_data,
                    filename='cstr_flash_encdeckooptrain.pickle')
    
main()