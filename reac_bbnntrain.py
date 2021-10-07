# [depends] %LIB%/hybridId.py %LIB%/BlackBoxFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import tensorflow as tf
import time
import numpy as np
from hybridId import PickleTool, get_scaling, get_train_val_data
from BlackBoxFuncs import (create_model, train_model, 
                           get_val_predictions)

# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def main():
    """ Main function to be executed. """

    # Load data.
    reac_parameters = PickleTool.load(filename='reac_parameters.pickle',
                                         type='read')

    # Sizes and sample time.
    plant_pars = reac_parameters['plant_pars']
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    Delta = plant_pars['Delta']

    # Raw training data.
    training_data = reac_parameters['training_data']

    # Create some parameters.
    unmeasXindices = [2]
    tthrow = 10
    Np = 2
    fNDims = [Ny + Nu + Np*(Ny+Nu), 16, Ny]

    # Filenames.
    ckpt_path = 'reac_bbnntrain.ckpt'
    stdout_filename = 'reac_bbnntrain.txt'

    # Get scaling and the training data.
    xuyscales = get_scaling(data=training_data[0])
    unmeasGbx0_list = [None for _ in range(len(training_data))]
    train_data, trainval_data, val_data = get_train_val_data(tthrow=tthrow,
                                                    Np=Np, xuyscales=xuyscales,
                                                data_list=training_data, 
                                                unmeasGbx0_list=unmeasGbx0_list)

    # Loop over the number of samples.
    for num_sample in num_samples:
        
        # Create model.
        model = create_model(Np=Np, Ny=Ny, Nu=Nu, fNDims=fNDims)
        
        # Use num samples to adjust here the num training samples.
        train_samples = dict(yz0=train_data['yz0'],
                             inputs=train_data['inputs'],
                             outputs=train_data['outputs'])

        # Train.
        train_model(model=model, epochs=8000, batch_size=1, 
                      train_data=train_samples, trainval_data=trainval_data, 
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

        # Validate.
        val_prediction = get_val_predictions(model=model,
                                        val_data=val_data, xuyscales=xuyscales, 
                                        unmeasXIndices=unmeasXindices, 
                                        ckpt_path=ckpt_path, Delta=Delta)

        # Get weights to store.
        fNWeights = model.get_weights()

    # Num samples array for quick plotting.
    num_samples = np.asarray(num_samples) + trainval_data['inputs'].shape[1]

    # Save the weights.
    reac_train = dict(Np=Np, fNDims=fNDims,
                        fNWeights=fNWeights,
                        val_prediction=val_prediction,
                        val_metrics=val_metrics,
                        xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=reac_train,
                    filename='reac_bbnntrain.pickle')

main()