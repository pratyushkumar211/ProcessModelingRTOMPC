"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
sys.path.append('')
import time
import numpy as np
import tensorflow as tf
from tworeac_parameters import _get_tworeac_parameters
tf.keras.backend.set_floatx('float64')

class TwoReacCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_{NN}(y_{k+1-N_p:k}, u_{k+1-N_p:k-1}), y = xG
    """
    def __init__(self, fnn_layers, tworeac_parameters, **kwargs):
        super(TwoReacHybridCell, self).__init__(**kwargs)
        self.fnn_layers = fnn_layers
        self.tworeac_parameters = tworeac_parameters

    @property
    def state_size(self):
        return self.tworeac_parameters['Ng']
    
    @property
    def output_size(self):
        return self.tworeac_parameters['Ng']        

    def _fg(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. """

        # Extract the parameters.
        k1 = self.tworeac_parameters['k1']
        tau = self.tworeac_parameters['tau']

        # The concentrations.
        (Ca, Cd, Cc) = (x[..., 0:1], x[..., 1:2], x[..., 2:3])
        Ca0 = u[..., 0:1]
        
        # Define a constant.
        one = tf.constant(1., shape=(), dtype='float64')
        sqrtoneplusbetaCa = tf.math.sqrt(tf.math.add(one, beta*Ca))

        # Write the ODEs.
        dCabydt = F*tf.math.subtract(Ca0, Ca)/Vr
        dCabydt = tf.math.subtract(dCabydt, k1*Ca)

        dCdbydt1 = 0.5*k1*Ca
        dCdbydt2 = tf.math.divide(tf.math.subtract(sqrtoneplusbetaCa, one), 
                                  tf.math.add(sqrtoneplusbetaCa, one))
        dCdbydt = tf.math.multiply(dCdbydt1, dCdbydt2) 
        dCdbydt = tf.math.subtract(dCdbydt, F*Cd/Vr)

        dCcbydt = 2*k1*tf.math.divide(Ca,  tf.math.add(one, sqrtoneplusbetaCa)) 
        dCcbydt = tf.math.subtract(dCcbydt, F*Cc/Vr)

        # Return the derivative.
        return tf.concat((dCabydt, dCdbydt, dCcbydt), axis=-1)

    def _fnn(self, yseq, useq):
        """ Compute the output of the feedforward network. """
        fnn_output = tf.concat((yseq, useq), axis=-1)
        for layer in self.fnn_layers:
            fnn_output = layer(fnn_output)
        return fnn_output

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of xG: (None, Ng)
            Dimension of u: (None, Nu)
            Dimension of yseq: (None, (Np-1)*p)
            Dimension of useq: (None, (Np-1)*m)
        """
        # Extract important variables.
        [xG] = states
        [u, yseq, useq] = inputs
        Delta = self.threereac_parameters['sample_time']

        # Write-down the RK4 step for the NN grey-box augmentation.
        yseq_fnn = tf.concat((xG, yseq), axis=-1)
        k1 = self._fg(xG, u) + self._fnn(yseq_fnn, useq)

        yseq_interp = self.interp_layer(yseq_fnn)
        yseq_fnn = tf.concat((xG + Delta*(k1/2), yseq_interp), axis=-1)
        k2 = self._fg(xG + Delta*(k1/2), u) + self._fnn(yseq_fnn, useq)
        
        yseq_fnn = tf.concat((xG + Delta*(k2/2), yseq_interp), axis=-1)
        k3 = self._fg(xG + Delta*(k2/2), u) + self._fnn(yseq_fnn, useq)
        
        yseq_fnn = tf.concat((xG + Delta*k3, yseq_interp), axis=-1)
        k4 = self._fg(xG + Delta*k3, u) + self._fnn(yseq_fnn, useq)
        
        # Get the current output/state and the next time step.
        y = xG
        xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Return output and states at the next time-step.
        return (y, xGplus)

class InterpolationLayer(tf.keras.layers.Layer):
    """
    The layer to perform interpolation for RK4 predictions.
    """
    def __init__(self, p, Np, trainable=False, name=None):
        super(InterpolationLayer, self).__init__(trainable, name)
        self.p = p
        self.Np = Np

    def call(self, yseq):
        """ The main call function of the interpolation layer. 
        y is of dimension: (None, Np*p)
        Return y of dimension: (None, (Np-1)*p)
        """
        yseq_interp = []
        for t in range(self.Np-1):
            yseq_interp.append(0.5*(yseq[..., t*n:(t+1)*n] + 
                                    yseq[..., (t+1)*n:(t+2)*n]))
        return tf.concat(yseq_interp, axis=-1)

    def get_config(self):
        return super().get_config()

class TwoReacModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, fnn_dims, tworeac_parameters):
        """ Create the dense layers for the NN, and 
            construct the overall model. """
        (Ng, Nu) = (tworeac_parameters['Ng'], tworeac_parameters['Nu'])

        # Input layers.
        inputs = [tf.keras.Input(name='u', shape=(None, Nu)),
                  tf.keras.Input(name='yseq', shape=(None, Ng)),
                  tf.keras.Input(name='useq', shape=(None, Nu))]

        # Dense layers for the black-box NN.
        fnn_layers = []
        for dim in fnn_dims[1:-1]:
            fnn_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
        fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1]))

        # Construct the RNN layer and the model.
        tworeac_cell = TwoReacCell(fnn_layers, tworeac_parameters)
        tworeac_layer = tf.keras.layers.RNN(tworeac_cell, return_sequences=True)
        outputs = tworeac_layer(inputs)
        super().__init__(inputs=inputs, outputs=outputs)