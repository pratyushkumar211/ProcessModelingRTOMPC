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
tf.keras.backend.set_floatx('float64')

class TwoReacCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_{NN}(y_{k+1-N_p:k}, u_{k+1-N_p:k-1}), y = xG
    """
    def __init__(self, Np, interp_layer, 
                       fnn_layers, tworeac_parameters, **kwargs):
        super(TwoReacCell, self).__init__(**kwargs)
        self.Np = Np
        self.interp_layer = interp_layer
        self.fnn_layers = fnn_layers
        self.tworeac_parameters = tworeac_parameters
        (self.Ng, self.Nu) = (tworeac_parameters['Ng'],
                              tworeac_parameters['Nu'])

    @property
    def state_size(self):
        return self.Ng
    
    @property
    def output_size(self):
        return self.Ng        

    def _fg(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. """

        # Extract the parameters.
        k1 = self.tworeac_parameters['k1']
        tau = self.tworeac_parameters['ps'].squeeze()
        
        # Get the state and control.
        (Ca, Cb) = (x[..., 0:1], x[..., 1:2])
        Ca0 = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = (Ca0 - Ca)/tau - k1*Ca
        dCbbydt = k1*Ca - Cb/tau

        # Return the derivative.
        return tf.concat([dCabydt, dCbbydt], axis=-1)

    def _fnn(self, ypseq, upseq):
        """ Compute the output of the feedforward network. """
        fnn_output = tf.concat((ypseq, upseq), axis=-1)
        for layer in self.fnn_layers:
            fnn_output = layer(fnn_output)
        return fnn_output

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of xG: (None, Ng)
            Dimension of input: (None, Nu+(Np-1)*(Ng+Nu))
        """
        # Extract important variables.
        [xG] = states
        [u, ypseq, upseq] = tf.split(inputs, [self.Nu, 
                            (self.Np-1)*self.Ng, (self.Np-1)*self.Nu], axis=-1)
        Delta = self.tworeac_parameters['sample_time']

        # Write-down the RK4 step for the NN grey-box augmentation.
        ypseq_fnn = tf.concat((xG, ypseq), axis=-1)
        k1 = self._fg(xG, u) + self._fnn(ypseq_fnn, upseq)
        
        ypseq_interp = self.interp_layer(ypseq_fnn)
        ypseq_fnn = tf.concat((xG + Delta*(k1/2), ypseq_interp), axis=-1)
        k2 = self._fg(xG + Delta*(k1/2), u) + self._fnn(ypseq_fnn, upseq)
        
        ypseq_fnn = tf.concat((xG + Delta*(k2/2), ypseq_interp), axis=-1)
        k3 = self._fg(xG + Delta*(k2/2), u) + self._fnn(ypseq_fnn, upseq)
        
        ypseq_fnn = tf.concat((xG + Delta*k3, ypseq_interp), axis=-1)
        k4 = self._fg(xG + Delta*k3, u) + self._fnn(ypseq_fnn, upseq)
        
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
            yseq_interp.append(0.5*(yseq[..., t*self.p:(t+1)*self.p] + 
                                    yseq[..., (t+1)*self.p:(t+2)*self.p]))
        return tf.concat(yseq_interp, axis=-1)

    def get_config(self):
        return super().get_config()

class TwoReacModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, Np, fnn_dims, tworeac_parameters):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        (Ng, Nu) = (tworeac_parameters['Ng'], tworeac_parameters['Nu'])
        layer_input = tf.keras.Input(name='u_ypseq_upseq', 
                        shape=(None, Nu+(Np-1)*(Ng+Nu)))
        initial_state = tf.keras.Input(name='x0', shape=(Ng, ))

        # Get the interpolation layer.
        interp_layer = InterpolationLayer(p=Ng, Np=Np)

        # Dense layers for the black-box NN.
        fnn_layers = []
        for dim in fnn_dims[1:-1]:
            fnn_layers.append(tf.keras.layers.Dense(dim, activation='tanh'))
        fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1], 
                                                kernel_initializer='zeros',
                                                use_bias=False))

        # Construct the RNN layer and the model.
        tworeac_cell = TwoReacCell(Np, interp_layer, 
                                   fnn_layers, tworeac_parameters)
        tworeac_layer = tf.keras.layers.RNN(tworeac_cell, return_sequences=True)
        layer_output = tworeac_layer(inputs=layer_input, 
                                initial_state=[initial_state])
        super().__init__(inputs=[layer_input, initial_state], 
                         outputs=layer_output)