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
        return [self.Ng, self.Nb]
    
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
        sqrtoneplusbetaCa = tf.math.sqrt(1 + beta*Ca)

        # Write the ODEs.
        dCabydt = F*(Ca0-Ca)/Vr - k1*Ca
        dCdbydt = 0.5*k1*Ca*(-1+sqrtoneplusbetaCa)/(1+sqrtoneplusbetaCa) 
        dCdbydt = dCdbydt - F*Cd/Vr
        dCcbydt = 2*k1*Ca/(1 + sqrtoneplusbetaCa) - F*Cc/Vr

        # Return the derivative.
        return tf.concat((dCabydt, dCdbydt, dCcbydt), axis=-1)

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell."""
        [xG, dN] = states
        u = inputs
        h = self.threereac_parameters['sample_time']

        # Get the current output.
        y = xG[..., -1:]

        # Write-down the RK4 step and NN Grey-box augmentation.
        k1 = self._threereac_greybox_ode(xG, u)
        k2 = self._threereac_greybox_ode(xG + h*(k1/2), u)
        k3 = self._threereac_greybox_ode(xG + h*(k2/2), u)
        k4 = self._threereac_greybox_ode(xG + h*k3, u)
        xGplus = xG + (h/6)*(k1 + 2*k2 + 2*k3 + k4) + dN @ self.BN.T

        # Input to the black-box layer and compute one-step ahead
        # of the black-box layer.
        bb_output = tf.concat((xG, dN, u), axis=-1)
        for layer in self.bb_layers:
            bb_output = layer(bb_output)
        dNplus = bb_output

        # Return output and states at the next time-step.
        return (y, [xGplus, dNplus])

def get_threereac_model(*, threereac_parameters, bb_dims):
    """ Get the Hybrid model which can be trained from data. """

    # Get the BN matrix.
    Nx = threereac_parameters['Nx'] - 1
    Nu = threereac_parameters['Nu']
    BN = np.concatenate((np.eye(Nx), np.ones((Nx, bb_dims[0] - Nx))), axis=-1)

    # Get the black-box layers.
    bb_layers = []
    for dim in bb_dims[1:]:
        bb_layers.append(tf.keras.layers.Dense(dim, activation='relu'))

    # Get instances of the RNN cell and layer.
    HybridCell = ThreeReacHybridCell(3, 5, 1, BN, 
                                    bb_layers, threereac_parameters)
    HybridLayer = tf.keras.layers.RNN(HybridCell, return_sequences=True)

    
    # Create a sequential model and compute the output.
    model_input = tf.keras.Input(name='u', shape=(None, Nu))
    model = tf.keras.Sequential()
    model.add(HybridLayer)
    model_output = model(model_input)

    # Return the model.
    return model