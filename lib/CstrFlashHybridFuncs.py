"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf
from BlackBoxFuncs import fnnTF

class CstrFlashHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_N(xG, y_{k-N_p:k-1}, u_{k-N_p:k-1}), y = xG
    """
    def __init__(self, Np, interpLayer, fNLayers, xuyscales, 
                       grey_box_pars, **kwargs):
        super(CstrFlashHybridCell, self).__init__(**kwargs)

        # Save attributes.
        self.Np = Np
        self.interpLayer = interpLayer
        self.fNLayers = fNLayers
        self.parameters = grey_box_pars
        self.xuyscales = xuyscales
        self.Nx, self.Ny, self.Nu = (grey_box_pars['Ng'],
                                     grey_box_pars['Ny'],
                                     grey_box_pars['Nu'])
    
    @property
    def state_size(self):
        return self.Ng + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny        

    def _fg(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. """
        
        # Extract the parameters.
        pho = self.parameters['pho']
        Cp = self.parameters['Cp']
        Ar = self.parameters['Ar']
        Ab = self.parameters['Ab']
        kr = self.parameters['kr']
        kb = self.parameters['kb']
        Td = self.parameters['Td']
        ps = self.parameters['ps']

        # Scale back to physical states.
        xmean, xstd = self.xuyscales['xscale']
        umean, ustd = self.xuyscales['uscale']
        x = x*xstd + xmean
        u = u*ustd + umean

        # Extract the plant states into meaningful names.
        Hr, CAr, CBr, Tr = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4] 
        Hb, CAb, CBb, Tb = x[..., 4:5], x[..., 5:6], x[..., 6:7], x[..., 7:8] 
        F, D = u[..., 0:1], u[..., 1:2]
        CAf, Tf = ps[0], ps[1]
        
        # The outlet mass flow rates.
        Fr = kr*tf.math.sqrt(Hr)
        Fb = kb*tf.math.sqrt(Hb)
        
        # Write the CSTR odes.
        dHrbydt = (F + D - Fr)/Ar
        dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
        dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
        dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
        dTrbydt = dTrbydt + (r1*delH1)/(pho*Cp)
        dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

        # Write the flash odes.
        dHbbydt = (Fr - Fb - D)/Ab
        dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
        dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
        dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

        # Get the scaled derivative.
        xdot = tf.concat([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
                          dHbbydt, dCAbbydt, dCBbbydt, dTbbydt], axis=-1)/xstd

        # Return the derivative.
        return xdot

    def _hxg(self, xg):
        """ Measurement function. """
        yindices = self.parameters['yindices']
        # Return the measured grey-box states.
        return tf.gather(xg, yindices, axis=-1)

    def _fnn(self, xg, z, u):
        """ Compute the output of the feedforward network. """
        fnn_output = tf.concat((xg, z, u), axis=-1)
        for layer in self.fnn_layers:
            fnn_output = layer(fnn_output)
        # Return.
        return fnn_output

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ng + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
        """
        # Extract variables.
        [xGz] = states
        [xG, z] = tf.split(xGz, [self.Ng, self.Np*(self.Ny+self.Nu)],
                           axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
                                  axis=-1)
        u = inputs

        # Extract parameters.
        Delta = self.parameters['Delta']
        xscale, yscale = self.xuyscales['xscale'], self.xuyscales['yscale']
        xmean, xstd = xscale
        ymean, ystd = yscale

        # Get k1.
        k1 = self._fg(xG, u) + self._fnn(xG, z, u)

        # Interpolate for k2 and k3.
        hxG = self._hxg(xG)
        ypseq_interp = self.interp_layer(tf.concat((ypseq, hxG), axis=-1))
        z = tf.concat((ypseq_interp, upseq), axis=-1)
        
        # Get k2.
        k2 = self._fg(xG + Delta*(k1/2), u) + self._fnn(xG + Delta*(k1/2), z, u)

        # Get k3.
        k3 = self._fg(xG + Delta*(k2/2), u) + self._fnn(xG + Delta*(k2/2), z, u)

        # Get k4.
        ypseq_shifted = tf.concat((ypseq[..., self.Ny:], hxG), axis=-1)
        z = tf.concat((ypseq_shifted, upseq), axis=-1)
        k4 = self._fg(xG + Delta*k3, u) + self._fnn(xG + Delta*k3, z, u)
        
        # Get the current output/state and the next time step.
        y = (self._hxg(xG*xstd + xmean) - ymean)/ystd
        xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
        xplus = tf.concat((xGplus, zplus), axis=-1)

        # Measurement/Grey-box state.
        output = tf.concat((y, xG), axis=-1)
        
        # Return output and states at the next time-step.
        return (output, xplus)

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
        y is of dimension: (None, (Np+1)*p)
        Return y of dimension: (None, Np*p)
        """
        yseq_interp = []
        for t in range(self.Np):
            yseq_interp.append(0.5*(yseq[..., t*self.p:(t+1)*self.p] + 
                                    yseq[..., (t+1)*self.p:(t+2)*self.p]))
        return tf.concat(yseq_interp, axis=-1)

    def get_config(self):
        return super().get_config()

class CstrFlashModel(tf.keras.Model):
    """ Custom model for the CSTR Flash model. """
    def __init__(self, Np, fNDims, xuyscales, grey_box_pars):

        # Get the size and input layer, and initial state layer.
        Ny, Nu = grey_box_pars['Ny'], grey_box_pars['Nu']
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        if model_type == 'black-box':
            initial_state = tf.keras.Input(name='yz0',
                                           shape=(Ny + Np*(Ny+Nu), ))
        else:
            initial_state = tf.keras.Input(name='xGz0',
                                           shape=(Ng + Np*(Ny+Nu), ))
        
        # Dense layers for the NN.
        fnn_layers = []
        for dim in fnn_dims[1:-1]:
            fnn_layers.append(tf.keras.layers.Dense(dim, activation='tanh'))
        fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1]))

        # Build model depending on option.
        if model_type == 'black-box':
            cstr_flash_cell = BlackBoxCell(Np, Ny, Nu, fnn_layers)

        if model_type == 'hybrid':
            interp_layer = InterpolationLayer(p=Ny, Np=Np)
            cstr_flash_cell = CstrFlashHybridCell(Np, interp_layer, fnn_layers,
                                            xuyscales, cstr_flash_parameters)

        # Construct the RNN layer and the computation graph.
        cstr_flash_layer = tf.keras.layers.RNN(cstr_flash_cell,
                                               return_sequences=True)
        layer_output = cstr_flash_layer(inputs=layer_input,
                                        initial_state=[initial_state])
        if model_type == 'black-box':
            outputs = [layer_output]
        else:
            y, xG = tf.split(layer_output, [Ny, Ng], axis=-1)
            outputs = [y, xG]
        # Construct model.
        super().__init__(inputs=[layer_input, initial_state],
                         outputs=outputs)