# [depends] %LIB%/hybridId.py %LIB%/ReacHybridPartialGbFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_train_val_data
from BlackBoxFuncs import get_weights_from_tflayers
from ReacHybridPartialGbFuncs import (create_model,
                                      train_model, get_val_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """
    
    # Load data.
    reac_parameters = PickleTool.load(filename=
                                      'reac_parameters.pickle',
                                      type='read')

    # Get sizes/raw training data.
    Delta = reac_parameters['plant_pars']['Delta']
    hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']
    Nu = hyb_partialgb_pars['Nu']
    Ny = hyb_partialgb_pars['Ny']

    # Raw training data.
    training_data = reac_parameters['training_data']

    # Create some parameters.
    Ntstart = reac_parameters['Ntstart']
    Np = reac_parameters['Ntstart']
    unmeasXIndices = [2]
    r1Dims = [1, 8, 1]
    r2Dims = [1 + Np*(Ny + Nu), 32, 1]

    # Filenames.
    ckpt_path = 'reac_hybpartialgbtrain_dyndata.ckpt'
    stdout_filename = 'reac_hybpartialgbtrain_dyndata.txt'

    # Get scaling.
    xuyscales = get_scaling(data=training_data[0])
    
    # Get the the three types of data.
    (train_data, 
     trainval_data, val_data) = get_train_val_data(Ntstart=Ntstart, Np=Np,
                                                   xuyscales=xuyscales,
                                                   data_list=training_data)

    # Create model.
    model = create_model(r1Dims=r1Dims, r2Dims=r2Dims, 
                         Np=Np, xuyscales=xuyscales,
                         hyb_partialgb_pars=hyb_partialgb_pars)

    # Train.
    train_model(model=model, epochs=10, batch_size=1, 
                    train_data=train_data, trainval_data=trainval_data,
                    stdout_filename=stdout_filename, ckpt_path=ckpt_path)

    # Validate.
    val_prediction = get_val_predictions(model=model,
                                    val_data=val_data, xuyscales=xuyscales,
                                    unmeasXIndices=unmeasXIndices,
                                    ckpt_path=ckpt_path, Delta=Delta)

    # Get weights to store.
    r1Weights = get_weights_from_tflayers(model.r1Layers)
    r2Weights = get_weights_from_tflayers(model.r2Layers)

    # Save the weights.
    reac_train = dict(Np=Np, r1Dims=r1Dims, r2Dims=r2Dims,
                      r1Weights=r1Weights, r2Weights=r2Weights,
                      val_prediction=val_prediction,
                      xuyscales=xuyscales)

    # Save data.
    PickleTool.save(data_object=reac_train,
                    filename='reac_hybpartialgbtrain_dyndata.pickle')

main()