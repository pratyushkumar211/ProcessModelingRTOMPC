#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [depends] hvacnndata.mat
# [makes] mat
"""
@author: pkumar

Code for a 3 state 2 input nonLinear MPC example.
"""
from plottools.matio import loadmat, savemat
import custompath
custompath.add()
from tfnet import tfnet

data=loadmat('hvacnndata.mat',scalararray=False,asdict=True)
X = data['xnn']
Y = data['ynn']
del data

tfeg=tfnet(data={'X':X,'Y':Y},          
           options={'layer_dims':[6, 12, 4],
                    'frac_train':0.9,
                    'learning_rate':1e-2,
                    'num_epochs':20000,
                    'normalize_data':True,
                    'optimization_algorithm':'adam',
                    'activation':'tanh',
                    'minibatch_size':X.shape[1],
                    'print_cost':True,
                    'np_seed':10,
                    'tf_seed':1023
                    }           
           )

data=tfeg.model()
savemat('hvacnntrain.mat', data, appendmat=False, oned_as="column",
            list_as_cell=True)
