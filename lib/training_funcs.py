# [depends] HybridModelLayers.py hybridid.py
"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf
from HybridModelLayers import BlackBoxModel, KoopmanModel
from hybridid import SimData

def create_koopmodel(*, Np, Ny, Nu, fN_dims):
    """ Create/compile the two reaction model for training. """
    model = KoopmanModel(Np, Ny, Nu, fN_dims)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error', 
                  loss_weights=[0., 1.])
    # Return the compiled model.
    return model

def train_koopmodel(*, model, epochs, batch_size, train_data, trainval_data, 
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
    model.fit(x=[train_data['inputs'], train_data['yz0']], 
              y=[train_data['yz'], train_data['outputs']], 
              epochs=epochs, batch_size=batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['yz0']], 
                           [trainval_data['yz'], trainval_data['outputs']]),
            callbacks = [checkpoint_callback])

def get_koopval_predictions(*, model, val_data, xuyscales, 
                               xinsert_indices, ckpt_path):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['yz0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    ypredictions = model_predictions[1].squeeze()*ystd + ymean
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean

    # Get xpredictions.
    xpredictions = np.insert(ypredictions, xinsert_indices, np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    val_predictions = SimData(t=np.arange(0, Nt, 1), x=xpredictions, 
                              u=uval, y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['yz0']], 
                                y=[val_data['yz'], val_data['outputs']])

    # Return predictions and metric.
    return (val_predictions, val_metric)