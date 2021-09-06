# [depends] %LIB%/hybridId.py %LIB%/ReacHybridPartialGbFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_train_val_data
from ReacHybridFullGbFuncs import get_weights
from ReacHybridPartialGbFuncs import (create_model,
                                      train_model, get_val_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)
np.random.seed(2)

def main():
    """ Main function to be executed. """
    
    # Load data.
    reac_parameters = PickleTool.load(filename=
                                      'reac_parameters.pickle',
                                      type='read')

    # Get sizes/raw training data.
    Delta = reac_parameters['plant_pars']['Delta']
    hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']
    Nx = hyb_partialgb_pars['Nx']
    Nu = hyb_partialgb_pars['Nu']
    Ny = hyb_partialgb_pars['Ny']
    training_data = reac_parameters['training_data_dyn']

    # Create some parameters.
    Np = 2
    tthrow = 10
    r1Dims = [1, 8, 1]
    r2Dims = [1, 8, 1]
    r3Dims = [Np*(Nx+Nu), 32, 1]

    # Lists.
    val_predictions = []
    val_metrics = []
    trained_r1Weights = []
    trained_r2Weights = []
    trained_r3Weights = []

    # Filenames.
    ckpt_path = 'reac_hybpartialgbtrain_dyndata.ckpt'
    stdout_filename = 'reac_hybpartialgbtrain_dyndata.txt'

    # Get scaling.
    xuyscales = get_scaling(data=training_data[0])

    # Scaled unmeasured grey-box state.
    unmeasGbx0_list = [None for _ in range(len(training_data))]
    
    # Get the the three types of data.
    (train_data, 
     trainval_data, val_data) = get_train_val_data(tthrow=tthrow, 
                                                   Np=Np, xuyscales=xuyscales, 
                                                   data_list=training_data,
                                                unmeasGbx0_list=unmeasGbx0_list) 
    # Create model.
    model = create_model(r1Dims=r1Dims, r2Dims=r2Dims, 
                        r3Dims=r3Dims, Np=Np, 
                        xuyscales=xuyscales, 
                        hyb_partialgb_pars=hyb_partialgb_pars)

    # Train.
    train_model(model=model, epochs=5, batch_size=1, 
                    train_data=train_data, trainval_data=trainval_data,
                    stdout_filename=stdout_filename, ckpt_path=ckpt_path)

    # Validate.
    (val_prediction, 
     val_metric) = get_val_predictions(model=model,
                                    val_data=val_data, xuyscales=xuyscales,
                                    ckpt_path=ckpt_path, Delta=Delta)

    # Get weights to store.
    r1Weights = get_weights(model.r1Layers)
    r2Weights = get_weights(model.r2Layers)
    r3Weights = get_weights(model.r3Layers)

    # Save info.
    val_predictions.append(val_prediction)
    val_metrics.append(val_metric)
    trained_r1Weights.append(r1Weights)
    trained_r2Weights.append(r2Weights)
    trained_r3Weights.append(r3Weights)
    
    # Save the weights.
    reac_train = dict(Np=Np, r1Dims=r1Dims, r2Dims=r2Dims, r3Dims=r3Dims,
                      trained_r1Weights=trained_r1Weights,
                      trained_r2Weights=trained_r2Weights,
                      trained_r3Weights=trained_r3Weights,
                      val_predictions=val_predictions,
                      val_metrics=val_metrics,
                      xuyscales=xuyscales)

    # Save data.
    PickleTool.save(data_object=reac_train,
                    filename='reac_hybpartialgbtrain_dyndata.pickle')

main()