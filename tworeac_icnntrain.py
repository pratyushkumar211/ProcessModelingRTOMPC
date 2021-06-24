# [depends] %LIB%/hybridid.py %LIB%/training_funcs.py
# [depends] tworeac_parameters.pickle
# [makes] pickle
""" Script to train the hybrid model for the 
    three reaction system. 
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import PickleTool
from InputConvexFuncs import (get_scaling, get_train_val_data, 
                              create_model, train_model, 
                              get_val_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    tworeac_icnndata = PickleTool.load(filename=
                                        'tworeac_icnndata.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']

    # Create some parameters.
    zDims = [None, 32, 1]
    uDims = None
    Nu = plant_pars['Nu']

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_icnntrain.ckpt'
    stdout_filename = 'tworeac_icnntrain.txt'

    # Get scaling and the training data.
    ulpscales = get_scaling(u=tworeac_icnndata['u'], 
                            lyup=tworeac_icnndata['lyup'])
    datasize_fracs = [0.7, 0.15, 0.15]
    (train_data, 
     trainval_data, val_data) = get_train_val_data(u=tworeac_icnndata['u'], 
                                        lyup=tworeac_icnndata['lyup'], 
                                        ulpscales=ulpscales, 
                                        datasize_fracs=datasize_fracs)

    # Create model.
    model = create_model(Nu=Nu, zDims=zDims, uDims=None)

    # Use num samples to adjust here the num training samples.
    train_samples = dict(inputs=train_data['inputs'],
                         output=train_data['output'])

    # Train.
    train_model(model=model, epochs=1000, batch_size=32, 
                      train_data=train_samples, trainval_data=trainval_data,
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

    # Validate.
    val_prediction, val_metric = get_val_predictions(model=model,
                                val_data=val_data, ulpscales=ulpscales, 
                                ckpt_path=ckpt_path)

    # Get weights to store.
    fNWeights = model.get_weights()

    # Save info.
    val_predictions.append(val_prediction)
    val_metrics.append(val_metric)
    trained_weights.append(fNWeights)
    
    # Save the weights.
    tworeac_icnntrain = dict(zDims=zDims,
                         trained_weights=trained_weights,
                         val_predictions=val_predictions,
                         val_metrics=val_metrics,
                         num_samples=None,
                         ulpscales=ulpscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_icnntrain,
                    filename='tworeac_icnntrain.pickle')

main()