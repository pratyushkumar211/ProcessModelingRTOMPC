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
    def __init__(self, Ng, Nb, Ny, BN, threereac_parameters, **kwargs):
        super(ThreeReacHybridCell, self).__init__(**kwargs)
        self.Ng = Ng
        self.Nb = Nb
        self.Ny = Ny
        self.BN = BN
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

        # Define a constant.
        sqrtoneplusbetaCa = np.sqrt(1 + beta*Ca)

        # Write the ODEs.
        dCabydt = F*(Ca0-Ca)/Vr - k1*Ca
        dCdbydt = 0.5*k1*Ca*(-1+sqrtoneplusbetaCa)/(1+sqrtoneplusbetaCa) 
        dCdbydt = dCdbydt - F*Cd/Vr
        dCcbydt = 2*k1*Ca/(1 + sqrtoneplusbetaCa) - F*Cc/Vr

        # Return the derivative.
        return np.array([dCabydt, dCdbydt, dCcbydt])

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
        xGplus = xG + (h/6)*(k1 + 2*k2 + 2*k3 + k4) + self.BN*dN

        # Input to the black-box layer and compute one-step ahead
        # of the black-box layer.
        bb_output = tf.concat((xG, dN, u), axis=-1)
        for layer in self._layers:
            bb_output = layer(bb_output)
        dNplus = bb_output

        # Return output and states at the next time-step.
        return (y, [xGplus, dNplus])


#class RegulatorModel(tf.keras.Model):
#    """Custom regulator model, assumes 
#        by default that the NN would take 
#        uprev as an input."""
#    def __init__(self, Nx, Nu, regulator_dims, nnwithuprev=True):
#        
#        inputs = [tf.keras.Input(name='x', shape=(Nx,)),
#                  tf.keras.Input(name='xs', shape=(Nx,)),
#                  tf.keras.Input(name='us', shape=(Nu,))]
#        if nnwithuprev:
#            inputs.insert(1, tf.keras.Input(name='uprev', shape=(Nu,)))
#            regulator = RegulatorLayerWithUprev(layer_dims=regulator_dims[1:])
#        else:
#            regulator = RegulatorLayerWithoutUprev(layer_dims=
#                                                   regulator_dims[1:])
#        outputs = regulator(inputs)
#        super().__init__(inputs=inputs, outputs=outputs)

threereac_parameters = _get_threereac_parameters()
BN = np.concatenate((np.eye(3), np.ones((3, 5))), axis=-1)
HybridCell = ThreeReacHybridCell(3, 5, 1, BN, threereac_parameters)