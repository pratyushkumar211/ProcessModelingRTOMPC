# [depends] %LIB%/hybridId.py %LIB%/BlackBoxFuncs.py
# [depends] tworeac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_ss_train_val_data
from BlackBoxFuncs import create_model, train_model, get_val_predictions

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """

    # Load data.
    tworeac_parameters = PickleTool.load(filename='tworeac_parameters.pickle',
                                         type='read')

    # Get the sizes and raw training data.
    plant_pars = tworeac_parameters['plant_pars']
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    training_data = tworeac_parameters['training_data_ss']
    Delta = plant_pars['Delta']

    # Number of samples.
    num_samples = [300]

    # Create some parameters.
    ypred_xinsert_indices = []
    Np = 0
    fNDims = [Ny + Np*(Ny+Nu), 8, 8, Ny]

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_bbnntrain_ssdata.ckpt'
    stdout_filename = 'tworeac_bbnntrain_ssdata.txt'

    # Get scaling and the training data.
    datasize_fracs = [0.7, 0.15, 0.15]
    xuyscales = get_scaling(data=training_data)
    (train_data, trainval_data, 
     val_data) = get_ss_train_val_data(xuyscales=xuyscales,
                                       training_data=training_data, 
                                       datasize_fracs=datasize_fracs)

    # Make dictionary keys compatible with black box functions.
    train_data['yz0'] = train_data.pop('x0')
    trainval_data['yz0'] = trainval_data.pop('x0')
    val_data['yz0'] = val_data.pop('x0')

    # Loop over the number of samples.
    for num_sample in num_samples:
        
        # Create model.
        model = create_model(Np=Np, Ny=Ny, Nu=Nu, fNDims=fNDims)
        
        # Use num samples to adjust here the num training samples.
        train_samples = dict(yz0=train_data['yz0'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])

        # Train.
        train_model(model=model, epochs=1000, batch_size=64, 
                      train_data=train_samples, trainval_data=trainval_data, 
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        (val_prediction, val_metric) = get_val_predictions(model=model,
                                        val_data=val_data, xuyscales=xuyscales, 
                                        xinsert_indices=ypred_xinsert_indices, 
                                        ckpt_path=ckpt_path, Delta=Delta)

        # Get weights to store.
        fNWeights = model.get_weights()

        # Save info.
        val_predictions.append(val_prediction)
        val_metrics.append(val_metric)
        trained_weights.append(fNWeights)

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    tworeac_train = dict(Np=Np, fNDims=fNDims,
                         trained_weights=trained_weights,
                         val_predictions=val_predictions,
                         val_metrics=val_metrics,
                         num_samples=num_samples,
                         xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_train,
                    filename='tworeac_bbnntrain_ssdata.pickle')

main()