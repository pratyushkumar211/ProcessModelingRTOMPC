"""
Custom Neural Network layers for performing
experiments and comparisions with the MPC optimization problem.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import time
import math
import numpy as np
import tensorflow as tf
from linearMPC import LinearMPCController
from python_utils import PickleTool
tf.keras.backend.set_floatx('float64')

class ThreeReacHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell.
    [xG, dN]^+ = [fG(xG, u) + BN*dN;fN(xG, dN, u)], y = [0, 0, 1]xG
    """
    def __init__(self, units, **kwargs):
      super(ThreeReacHybridCell, self).__init__(**kwargs)
      self.units = units

    @property
    def state_size(self):
      return self.units

    def build(self, input_shape):
      self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='uniform',
                                    name='kernel')
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          initializer='uniform',
          name='recurrent_kernel')
      self.built = True

    def call(self, inputs, states):
      prev_output = states[0]
      h = K.dot(inputs, self.kernel)
      output = h + K.dot(prev_output, self.recurrent_kernel)
      return output, output

class RegulatorModel(tf.keras.Model):
    """Custom regulator model, assumes 
        by default that the NN would take 
        uprev as an input."""
    def __init__(self, Nx, Nu, regulator_dims, nnwithuprev=True):
        
        inputs = [tf.keras.Input(name='x', shape=(Nx,)),
                  tf.keras.Input(name='xs', shape=(Nx,)),
                  tf.keras.Input(name='us', shape=(Nu,))]
        if nnwithuprev:
            inputs.insert(1, tf.keras.Input(name='uprev', shape=(Nu,)))
            regulator = RegulatorLayerWithUprev(layer_dims=regulator_dims[1:])
        else:
            regulator = RegulatorLayerWithoutUprev(layer_dims=
                                                   regulator_dims[1:])
        outputs = regulator(inputs)
        super().__init__(inputs=inputs, outputs=outputs)