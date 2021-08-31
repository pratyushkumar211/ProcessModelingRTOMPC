# [depends] %LIB%/hybridId.py %LIB%/ReacHybridFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_train_val_data
from ReacHybridFuncs import (create_fullgb_model, 
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
    hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']
    Nx = hyb_fullgb_pars['Nx']
    Nu = hyb_fullgb_pars['Nu']
    Ny = hyb_fullgb_pars['Ny']
    training_data = reac_parameters['training_data_dyn']

    # Indices used for the training the Full Grey-Box model.
    extraUnmeasGbCostScale = 0
    yi = reac_parameters['plant_pars']['yindices']
    unmeasGbPredi = [2]
    unmeasGbEsti = [3]

    # Number of samples.
    num_samples = [hours*60 for hours in [6]]

    # Create some parameters.
    xinsert_indices = []
    Np = 2
    tthrow = 10
    r1Dims = [1, 8, 1]
    r2Dims = [1, 8, 1]
    r3Dims = [1, 8, 1]
    estCDims = [Np*(Ny + Nu), 8, 1]

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'reac_hybfullgbtrain_dyndata.ckpt'
    stdout_filename = 'reac_hybfullgbtrain_dyndata.txt'

    # Get scaling and the training data.
    xuyscales = get_scaling(data=training_data[0])
    unmeasGbx0 = np.array([0.2])
    (train_data, 
     trainval_data, val_data) = get_train_val_data(tthrow=tthrow, 
                                                   Np=Np, xuyscales=xuyscales, 
                                                   data_list=training_data, 
                                                   unmeasGbx0=unmeasGbx0)

    # Loop over the number of samples.
    for num_sample in num_samples:
        
        # Create model.
        model = create_fullgb_model(r1Dims=r1Dims, r2Dims=r2Dims, 
                            r3Dims=r3Dims, estCDims=estCDims, Np=Np, 
                            xuyscales=xuyscales, hyb_fullgb_pars=hyb_fullgb_pars, 
                        extraUnmeasGbCostScale=extraUnmeasGbCostScale, yi=yi, 
                        unmeasGbPredi=unmeasGbPredi, unmeasGbEsti=unmeasGbEsti)

        # Use num samples to adjust here the num training samples.
        train_samples = dict(x0=train_data['x0'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])
        
        # Train.
        train_model(model=model, epochs=8000, batch_size=1, 
                      train_data=train_samples, trainval_data=trainval_data,
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        (val_prediction, val_metric) = get_val_predictions(model=model,
                                    val_data=val_data, xuyscales=xuyscales, 
                                    xinsert_indices=xinsert_indices, 
                                    ckpt_path=ckpt_path, Delta=Delta, fullGb=True)

        # Get weights to store.
        fNWeights = model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        trained_weights.append(fNWeights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    reac_train = dict(Np=Np, r1Dims=r1Dims, r2Dims=r2Dims, 
                      r3Dims=r3Dims, estCDims=estCDims,
                      trained_weights=trained_weights,
                      val_predictions=val_predictions,
                      val_metrics=val_metrics,
                      num_samples=num_samples,
                      xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=reac_train,
                    filename='reac_hybfullgbtrain_dyndata.pickle')

main()