# [depends] %LIB%/hybridid.py %LIB%/CstrFlashHybridFuncs.py
# [depends] cstr_flash_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridid import PickleTool, get_scaling, get_train_val_data
from CstrFlashHybridFuncs import create_model, train_model, get_val_predictions

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """
    # Load data.
    cstr_flash_parameters = PickleTool.load(filename=
                                         'cstr_flash_parameters.pickle',
                                         type='read')

    # Get sizes/raw training data.
    hyb_greybox_pars = cstr_flash_parameters['hyb_greybox_pars']
    plant_pars = cstr_flash_parameters['plant_pars']
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    training_data = cstr_flash_parameters['training_data']
    
    # Number of samples.
    num_samples = [hours*60 for hours in [6]]

    # Create some parameters.
    ypred_xinsert_indices = []
    tthrow = 10
    Np = 0
    fNDims = [Ny + Np*(Ny+Nu), 4, 2]

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'cstr_flash_hybtrain.ckpt'
    stdout_filename = 'cstr_flash_hybtrain.txt'

    # Get scaling and the training data.
    xuyscales = get_scaling(data=training_data[-1])
    (train_data, trainval_data, val_data) = get_train_val_data(tthrow=tthrow, 
                                            Np=Np, xuyscales=xuyscales, 
                                            data_list=training_data)
    
    # Loop over the number of samples.
    for num_sample in num_samples:
        
        # Create model.
        model = create_model(fNDims=fNDims, 
                             xuyscales=xuyscales, 
                             hyb_greybox_pars=hyb_greybox_pars)

        # Use num samples to adjust here the num training samples.
        train_samples = dict(x0=train_data['x0'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])

        # Train.
        train_model(model=model, epochs=20000, batch_size=8, 
                    train_data=train_samples, trainval_data=trainval_data, 
                    stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        (val_prediction, val_metric) = get_val_predictions(model=model,
                                    val_data=val_data, xuyscales=xuyscales, 
                                    xinsert_indices=ypred_xinsert_indices, 
                                    ckpt_path=ckpt_path)

        # Get weights to store.
        fNWeights = model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        trained_weights.append(fNWeights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    cstr_flash_train = dict(Np=Np, fNDims=fNDims,
                            trained_weights=trained_weights,
                            val_predictions=val_predictions,
                            val_metrics=val_metrics,
                            num_samples=num_samples,
                            xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=cstr_flash_train,
                    filename='cstr_flash_hybtrain.pickle')

main()