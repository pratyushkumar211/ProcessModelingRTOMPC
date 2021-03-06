# [depends] %LIB%/hybridId.py %LIB%/ReacHybridFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_train_val_data
from ReacHybridFullGbFuncs import get_weights
from ReacHybridFullGbFuncsWithExtraLoss import (create_model,
                                            train_model, get_val_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)
np.random.seed(3)

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

    # Create some parameters.
    Np = 2
    tthrow = 10
    r1Dims = [1, 16, 1]
    r2Dims = [1, 16, 1]
    estCDims = [Np*(Ny + Nu), 64, 1]

    # Parameters for the extra loss.
    lam = 1
    yi_list = [0, 1]
    unmeasGbi_list = [2]
    unmeasGbi_NN_list = [3]

    # Lists.
    val_predictions = []
    val_metrics = []
    trained_r1Weights = []
    trained_r2Weights = []
    trained_estCWeights = []

    # Filenames.
    ckpt_path = 'reac_hybfullgbWithExtraLosstrain_dyndata.ckpt'
    stdout_filename = 'reac_hybfullgbWithExtraLosstrain_dyndata.txt'

    # Get scaling.
    xuyscales = get_scaling(data=training_data[0])
    ymean, ystd = xuyscales['yscale']
    xmean, xstd = xuyscales['xscale']
    xmean = np.concatenate((ymean, ymean[-1:]))
    xstd = np.concatenate((ystd, ystd[-1:]))
    xuyscales['xscale'] = (xmean, xstd)

    # Scaled unmeasured grey-box state.
    unmeasGbx0_list = [(np.random.rand(1, 1)-ymean[-1])/ystd[-1] 
                        for _ in range(len(training_data))]
    
    # Get the the three types of data.
    (train_data, 
     trainval_data, val_data) = get_train_val_data(tthrow=tthrow, 
                                                   Np=Np, xuyscales=xuyscales, 
                                                   data_list=training_data,
                                                unmeasGbx0_list=unmeasGbx0_list)   
    # Create model.
    model = create_model(r1Dims=r1Dims, r2Dims=r2Dims, 
                         estCDims=estCDims, Np=Np, 
                         xuyscales=xuyscales, 
                         hyb_fullgb_pars=hyb_fullgb_pars, 
                         lam=lam, yi_list=yi_list, 
                         unmeasGbi_list=unmeasGbi_list, 
                         unmeasGbi_NN_list=unmeasGbi_NN_list)

    # Train.
    train_model(model=model, epochs=10, batch_size=1, 
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
    estCWeights = get_weights(model.estCLayers)

    # Save info.
    val_predictions.append(val_prediction)
    val_metrics.append(val_metric)
    trained_r1Weights.append(r1Weights)
    trained_r2Weights.append(r2Weights)
    trained_estCWeights.append(estCWeights)

    # Save the weights.
    reac_train = dict(Np=Np, r1Dims=r1Dims, r2Dims=r2Dims, 
                      estCDims=estCDims,
                      trained_r1Weights=trained_r1Weights,
                      trained_r2Weights=trained_r2Weights,
                      trained_estCWeights=trained_estCWeights,
                      unmeasGbx0_list=unmeasGbx0_list,
                      val_predictions=val_predictions,
                      val_metrics=val_metrics,
                      xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=reac_train,
                    filename='reac_hybfullgbWithExtraLosstrain_dyndata.pickle')

main()