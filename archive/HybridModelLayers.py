"""
Custom Neural Network layers for performing
experiments and comparisions with the MPC optimization problem.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
sys.path.append('')
import time
import numpy as np
import tensorflow as tf
from threereac_parameters import _get_threereac_parameters
tf.keras.backend.set_floatx('float64')

class ThreeReacHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    [xG, dN]^+ = [fG(xG, u) + BN*dN;fN(xG, dN, u)], y = [0, 0, 1]xG
    """
    def __init__(self, Ng, Nb, Ny, BN, bb_layers, 
                       threereac_parameters, **kwargs):
        super(ThreeReacHybridCell, self).__init__(**kwargs)
        self.Ng = Ng
        self.Nb = Nb
        self.Ny = Ny
        self.BN = BN
        self.bb_layers = bb_layers
        self.threereac_parameters = threereac_parameters
    
    @property
    def state_size(self):
        return self.Nb + self.Ng
    
    @property
    def output_size(self):
        return self.Ny        

    def _threereac_greybox_ode(self, x, u):
        """ Function to compute the ODE value of the grey-box 
            model. """

        # Extract the parameters.
        k1 = self.threereac_parameters['k1']
        beta = self.threereac_parameters['beta']
        F = self.threereac_parameters['F']
        Vr = self.threereac_parameters['Vr']

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

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell."""

        (xG, dN) = tf.split(states[0], [self.Ng, self.Nb], -1)
        u = inputs
        h = self.threereac_parameters['sample_time']

        # Get the current output.
        y = xG[..., -1:]

        # Write-down the RK4 step and NN Grey-box augmentation.
        k1 = self._threereac_greybox_ode(xG, u)
        k2 = self._threereac_greybox_ode(tf.math.add(xG, h*(k1/2)), u)
        k3 = self._threereac_greybox_ode(tf.math.add(xG, h*(k2/2)), u)
        k4 = self._threereac_greybox_ode(tf.math.add(xG, h*k3), u)
        xGplus = tf.math.add(k1, tf.math.add(2*k2, tf.math.add(2*k3, k4)))
        xGplus = tf.math.add(xG, tf.math.add((h/6)*xGplus, dN @ self.BN.T))

        # Input to the black-box layer and compute one-step ahead
        # of the black-box layer.
        bb_output = tf.concat((xG, dN, u), axis=-1)
        for layer in self.bb_layers:
            bb_output = layer(bb_output)
        dNplus = bb_output

        # Concatenate to get all the states.
        xplus = tf.concat((xGplus, dNplus), axis=-1)

        # Return output and states at the next time-step.
        return (y, xplus)

def get_threereac_model(*, threereac_parameters, bb_dims):
    """ Get the Hybrid model which can be trained from data. """

    # Get the BN matrix.
    Nx = threereac_parameters['Nx'] - 1
    Nu = threereac_parameters['Nu']
    Ny = threereac_parameters['Ny']
    Nb = bb_dims[-1]
    BN = np.array([[0., 0.], 
                   [1., 0.], 
                   [0., 1.]])

    # Get the black-box layers.
    bb_layers = []
    for dim in bb_dims[1:]:
        bb_layers.append(tf.keras.layers.Dense(dim, 
                                    activation='relu',
                            kernel_initializer=tf.keras.initializers.Zeros(),
                            bias_initializer='zeros'))

    # Get the initial states.
    xG0 = threereac_parameters['xs'][np.newaxis, (0, 2, 3)]
    xG0 = tf.constant(xG0, shape=xG0.shape)
    dN0 = np.zeros((1, Nb))
    dN0 = tf.constant(dN0, shape=dN0.shape)
    x0 = tf.concat((xG0, dN0), axis=-1)

    # Get instances of the RNN cell and layer.
    HybridCell = ThreeReacHybridCell(Nx, Nb, Ny, BN, 
                                     bb_layers, threereac_parameters)
    HybridLayer = tf.keras.layers.RNN(HybridCell, 
                                      return_sequences=True)
    
    # Create a sequential model and compute the output.
    model_input = tf.keras.Input(name='u', shape=(None, Nu))
    model_output = HybridLayer(inputs=model_input, 
                  initial_state=[x0])
    model = tf.keras.Model(model_input, model_output)

    # Return the model.
    return model