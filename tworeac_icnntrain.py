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
from TwoReacHybridFuncs import (create_tworeac_model, train_hybrid_model, 
                               get_hybrid_predictions, tworeacHybrid_fxu,
                               tworeacHybrid_hx,
                               get_tworeacHybrid_pars)         

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

def get_icnndata_scaling(*, u, lyup, p=None):

    # Umean.
    umean = np.mean(u, axis=0)
    ustd = np.std(u, axis=0)
    
    # lyupmean.
    lyupmean = np.mean(lyup, axis=0)
    lyupstd = np.std(lyup, axis=0)
    
    # Get dictionary.
    ulpscales = dict(uscale = (umean, ustd), 
                     lyupscale = (lyupmean, lyupstd))

    # Get means of p and update dict if necessary.
    if p is not None:
        pmean = np.mean(p, axis=0)
        pstd = np.std(p, axis=0)
        ulpscales['pscale'] = (pmean, pstd)

    # Return.
    return ulpscales

def get_icnn_train_val_data(*, u, lyup, ulpscales, datasize_fracs, p=None):
    """ Return train, train val, and validation data for ICNN training. """

    # Get scaling.
    umean, ustd = ulpscales['uscale']
    lyupmean, lyupstd = ulpscales['lyupscale']

    # Do the scaling.
    u = (u - umean)/ustd
    lyup = (lyup - lyupmean)/lyupstd
    if p is not None:
        pmean, pstd = ulpscales['pscale']
        p = (p-pmean)/pstd

    # Get the corresponding fractions of data. 
    train_frac, trainval_frac, val_frac = datasize_fracs
    Ndata = u.shape[0]
    Ntrain = int(Ndata*train_frac)
    Ntrainval = int(Ndata*trainval_frac)
    Nval = int(Ndata*val_frac)

    # Get the three types of data.
    u = np.split(u, [Ntrain, Ntrain + Ntrainval, ], axis=0)
    lyup = np.split(lyup, [Ntrain, Ntrain + Ntrainval, ], axis=0)

    # Get dictionaries of data types.
    train_data = dict(u=u[0], lyup=lyup[0])
    trainval_data = dict(u=u[1], lyup=lyup[1])
    val_data = dict(u=u[2], lyup=lyup[2])
    if p is not None:
        p = np.split(p, [Ntrain, Ntrain + Ntrainval, ], axis=0)
        train_data['p'] = p[0]
        trainval_data['p'] = p[1]
        val_data['p'] = p[2]

    # Return.
    return train_data, trainval_data, val_data

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
    ulpscales = get_icnndata_scaling(u=utrain, lyup=lyutrain)
    datasize_fracs = [0.7, 0.15, 0.15]
    (train_data, 
     trainval_data, val_data) = get_icnn_train_val_data(u=utrain, 
                                        lyup=lyutrain, ulpscales=ulpscales, 
                                        datasize_fracs=datasize_fracs)
        
    # Create model.
    model = create_tworeac_model(zDims=zDims, greybox_pars=greybox_pars)
    
    # Use num samples to adjust here the num training samples.
    train_samples = dict(x0=train_data['x0'],
                            inputs=train_data['inputs'],
                            outputs=train_data['outputs'])

    # Train.
    train_hybrid_model(model=model, epochs=5000, batch_size=2, 
                    train_data=train_samples, trainval_data=trainval_data,
                    stdout_filename=stdout_filename, ckpt_path=ckpt_path)

    # Validate.
    (val_prediction, val_metric) = get_hybrid_predictions(model=model,
                                val_data=val_data, xuyscales=xuyscales, 
                                xinsert_indices=xinsert_indices, 
                                ckpt_path=ckpt_path)

    # Get weights to store.
    fNWeights = model.get_weights()

    # Save info.
    val_predictions.append(val_prediction)
    val_metrics.append(val_metric)
    trained_weights.append(fNWeights)

    # Save the weights.
    tworeac_icnntrain = dict(fNDims=fNDims,
                         trained_weights=trained_weights,
                         val_predictions=val_predictions,
                         val_metrics=val_metrics,
                         num_samples=num_samples,
                         xuyscales=xuyscales)
    
    # Save data.
    PickleTool.save(data_object=tworeac_train,
                    filename='tworeac_icnntrain.pickle')

main()