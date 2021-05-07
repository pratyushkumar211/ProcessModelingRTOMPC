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
from tworeac_funcs import cost_yup
from economicopt import get_sscost
from TwoReacHybridFuncs import (tworeacHybrid_fxu,
                                tworeacHybrid_hx,
                                get_tworeacHybrid_pars)         
from InputConvexFuncs import (get_scaling, get_train_val_data, 
                              create_model, train_model, 
                              get_val_predictions_metric)


# Set the tensorflow global and graph-level seed.
tf.random.set_seed(123)

def generate_data(*, hyb_fxu, hyb_hx, hyb_pars):
    """ Function to generate data to train the ICNN. """

    p = [100, 200]
    cost_yu = lambda y, u: cost_yup(y, u, p)

    # Get a list of random inputs.
    Nu = hyb_pars['Nu']
    ulb, uub = hyb_pars['ulb'], hyb_pars['uub']
    Ndata = 1000
    us_list = list((uub-ulb)*np.random.rand(Ndata, Nu) + ulb)

    # Get the corresponding steady state costs.
    ss_costs = []
    for us in us_list:
        ss_cost = get_sscost(fxu=hyb_fxu, hx=hyb_hx, lyu=cost_yu, 
                             us=us, parameters=hyb_pars)
        ss_costs += [ss_cost]
    lyu = np.array(ss_costs)
    u = np.array(us_list)

    # Return.
    return u, lyu

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                        'tworeac_parameters.pickle',
                                         type='read')
    tworeac_hybtrain = PickleTool.load(filename=
                                        'tworeac_hybtrain.pickle',
                                         type='read')
    plant_pars = tworeac_parameters['plant_pars']
    greybox_pars = tworeac_parameters['greybox_pars']

    # Get the black-box model parameters and function handles.
    hyb_pars = get_tworeacHybrid_pars(train=tworeac_hybtrain, 
                                      greybox_pars=greybox_pars)
    hyb_fxu = lambda x, u: tworeacHybrid_fxu(x, u, hyb_pars)
    hyb_hx = lambda x: tworeacHybrid_hx(x)

    # Generate data.
    utrain, lyutrain = generate_data(hyb_fxu=hyb_fxu, hyb_hx=hyb_hx, 
                                     hyb_pars=hyb_pars)    

    # Create some parameters.
    zDims = [None, 32, 32, 1]
    uDims = None

    # Create lists to store data.
    trained_weights = []
    val_metrics = []
    val_predictions = []

    # Filenames.
    ckpt_path = 'tworeac_icnntrain.ckpt'
    stdout_filename = 'tworeac_icnntrain.txt'

    # Get scaling and the training data.
    ulpscales = get_scaling(u=utrain, lyup=lyutrain)
    datasize_fracs = [0.7, 0.15, 0.15]
    (train_data, 
     trainval_data, val_data) = get_train_val_data(u=utrain, 
                                        lyup=lyutrain, ulpscales=ulpscales, 
                                        datasize_fracs=datasize_fracs)
    breakpoint()
    # Create model.
    model = create_model(Nu=Nu, zDims=zDims, uDims=None)
    
    # Use num samples to adjust here the num training samples.
    train_samples = dict(inputs=train_data['inputs'],
                         output=train_data['output'])

    # Train.
    train_model(model=model, epochs=10, batch_size=2, 
                      train_data=train_samples, trainval_data=trainval_data,
                      stdout_filename=stdout_filename, ckpt_path=ckpt_path)

    # Validate.
    val_prediction, val_metric = get_val_predictions_metric(model=model,
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
    PickleTool.save(data_object=tworeac_train,
                    filename='tworeac_icnntrain.pickle')

main()