"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf

class TwoReacHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_N(xG, y_{k-N_p:k-1}, u_{k-N_p:k-1}), y = xG
    """
    def __init__(self, Np, interp_layer,
                       fnn_layers, tworeac_parameters, **kwargs):
        super(TwoReacHybridCell, self).__init__(**kwargs)
        self.Np = Np
        self.interp_layer = interp_layer
        self.fnn_layers = fnn_layers
        self.tworeac_parameters = tworeac_parameters
        (self.Ng, self.Nu) = (tworeac_parameters['Ng'],
                              tworeac_parameters['Nu'])

    @property
    def state_size(self):
        return self.Ng + self.Np*(self.Ng + self.Nu)
    
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
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = (Caf - Ca)/tau - k1*Ca
        dCbbydt = k1*Ca - Cb/tau

        # Return the derivative.
        return tf.concat([dCabydt, dCbbydt], axis=-1)

    def _fnn(self, xg, z):
        """ Compute the output of the feedforward network. """
        fnn_output = tf.concat((xg, z), axis=-1)
        for layer in self.fnn_layers:
            fnn_output = layer(fnn_output)
        return fnn_output

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ng + Np*(Ng + Nu))
            Dimension of input: (None, Nu)
        """
        # Extract variables.
        [xGz] = states
        [xG, z] = tf.split(xGz, [self.Ng, self.Np*(self.Ng+self.Nu)],
                           axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ng, self.Np*self.Nu],
                                  axis=-1)
        u = inputs
        Delta = self.tworeac_parameters['sample_time']
        
        # Get k1.
        k1 = self._fg(xG, u) + self._fnn(xG, z)
        
        # Interpolate for k2 and k3.
        ypseq_interp = self.interp_layer(tf.concat((ypseq, xG), axis=-1))
        z = tf.concat((ypseq_interp, upseq), axis=-1)
        
        # Get k2.
        k2 = self._fg(xG + Delta*(k1/2), u) + self._fnn(xG + Delta*(k1/2), z)

        # Get k3.
        k3 = self._fg(xG + Delta*(k2/2), u) + self._fnn(xG + Delta*(k2/2), z)

        # Get k4.
        ypseq_shifted = tf.concat((ypseq[..., self.Ng:], xG), axis=-1)
        z = tf.concat((ypseq_shifted, upseq), axis=-1)
        k4 = self._fg(xG + Delta*k3, u) + self._fnn(xG + Delta*k3, z)
        
        # Get the current output/state and the next time step.
        y = xG
        xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
        xplus = tf.concat((xGplus, zplus), axis=-1)

        # Return output and states at the next time-step.
        return (y, xplus)

class CstrFlashHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_N(xG, y_{k-N_p:k-1}, u_{k-N_p:k-1}), y = xG
    """
    def __init__(self, Np, interp_layer, fnn_layers,
                       xuyscales, cstr_flash_parameters, **kwargs):
        super(CstrFlashHybridCell, self).__init__(**kwargs)
        self.Np = Np
        self.interp_layer = interp_layer
        self.fnn_layers = fnn_layers
        self.parameters = cstr_flash_parameters
        self.xuyscales = xuyscales
        (self.Ng, self.Ny, self.Nu) = (cstr_flash_parameters['Ng'],
                                       cstr_flash_parameters['Ny'],
                                       cstr_flash_parameters['Nu'])

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
        alphaA = self.parameters['alphaA']
        alphaB = self.parameters['alphaB']
        pho = self.parameters['pho']
        Cp = self.parameters['Cp']
        Ar = self.parameters['Ar']
        Ab = self.parameters['Ab']
        kr = self.parameters['kr']
        kb = self.parameters['kb']
        delH1 = self.parameters['delH1']
        EbyR = self.parameters['EbyR']
        k1star = self.parameters['k1star']
        Td = self.parameters['Td']
        ps = self.parameters['ps']
        (xscale, uscale) = (self.xuyscales['xscale'], 
                            self.xuyscales['uscale'])

        # Scale x and u back to physical values.
        x = x*xscale
        u = u*uscale

        # Extract the plant states into meaningful names.
        (Hr, CAr) = x[..., 0:1], x[..., 1:2]
        (CBr, Tr) = x[..., 2:3], x[..., 3:4] 
        (Hb, CAb) = x[..., 4:5], x[..., 5:6]
        (CBb, Tb) = x[..., 6:7], x[..., 7:8] 
        (F, Qr) = u[..., 0:1], u[..., 1:2]
        (D, Qb) = u[..., 2:3], u[..., 3:4]
        (CAf, Tf) = ps[0], ps[1]
        
        # The flash vapor phase mass fractions.
        den = alphaA*CAb + alphaB*CBb
        CAd = alphaA*CAb/den
        CBd = alphaB*CBb/den

        # The outlet mass flow rates.
        Fr = kr*tf.math.sqrt(Hr)
        Fb = kb*tf.math.sqrt(Hb)

        # Rate constant and reaction rate.
        k1 = k1star*tf.math.exp(-EbyR/Tr)
        r1 = k1*CAr

        # Write the CSTR odes.
        dHrbydt = (F + D - Fr)/Ar
        dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
        dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
        dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
        dTrbydt = dTrbydt + (r1*delH1)/(pho*Cp)
        dTrbydt = dTrbydt + Qr/(pho*Ar*Cp*Hr)

        # Write the flash odes.
        dHbbydt = (Fr - Fb - D)/Ab
        dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
        dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
        dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

        # Get the scaled derivative. 
        xdot = tf.concat([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
                          dHbbydt, dCAbbydt, dCBbbydt, dTbbydt], axis=-1)/xscale

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
        Delta = self.parameters['Delta']
        (xscale, yscale) = (self.xuyscales['xscale'], 
                            self.xuyscales['yscale'])

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
        y = self._hxg(xG*xscale)/yscale
        xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
        xplus = tf.concat((xGplus, zplus), axis=-1)
        
        # Return output and states at the next time-step.
        return (y, xplus)

class CstrFlashGreyBlackCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = y_{k-N_p:k-1}, u_{k-N_p:k-1}
    xG^+  = f_N(xG, z, u), y = xG
    z^+ = z
    """
    def __init__(self, Np, fnn_layers, cstr_flash_parameters, **kwargs):
        super(CstrFlashGreyBlackCell, self).__init__(**kwargs)
        self.Np = Np
        self.fnn_layers = fnn_layers
        self.yindices = cstr_flash_parameters['yindices']
        (self.Ng, self.Ny, self.Nu) = (cstr_flash_parameters['Ng'],
                                       cstr_flash_parameters['Ny'],
                                       cstr_flash_parameters['Nu'])

    @property
    def state_size(self):
        return self.Ng + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ng

    def _fnn(self, xg, z, u):
        """ Compute the output of the feedforward network. """
        fnn_output = tf.concat((xg, z, u), axis=-1)
        for layer in self.fnn_layers:
            fnn_output = layer(fnn_output)
        return fnn_output

    def _hxg(self, xg):
        """ Measurement function. """
        # Return the measured grey-box states.
        return tf.gather(xg, self.yindices, axis=-1)

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
                
        # Get shifted y.
        hxG = self._hxg(xG)
        ypseq_shifted = tf.concat((ypseq[..., self.Ny:], hxG), axis=-1)
        
        # Get the current output/state and the next time step.
        y = hxG
        xGplus = self._fnn(xG, z, u)
        zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
        xplus = tf.concat((xGplus, zplus), axis=-1)

        # Return output and states at the next time-step.
        return (y, xplus)

class BlackBoxCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    y^+  = f_N(y, y_{k-N_p:k-1}, u_{k-N_p:k-1}, u), y = y
    """
    def __init__(self, Np, Ny, Nu, fnn_layers, **kwargs):
        super(BlackBoxCell, self).__init__(**kwargs)
        self.Np = Np
        self.fnn_layers = fnn_layers
        (self.Ny, self.Nu) = (Ny, Nu)

    @property
    def state_size(self):
        return self.Ny + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny

    def _fnn(self, y, ypseq, upseq, u):
        """ Compute the output of the feedforward network. """
        fnn_output = tf.concat((y, ypseq, upseq, u), axis=-1)
        for layer in self.fnn_layers:
            fnn_output = layer(fnn_output)
        return fnn_output

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ny + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
        """
        # Extract important variables.
        [yz] = states
        [y, z] = tf.split(yz, [self.Ny, self.Np*(self.Ny + self.Nu)],
                          axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
                           axis=-1)
        u = inputs
        
        # Get the current output/state and the next time step.
        yplus = self._fnn(y, ypseq, upseq, u)
        zplus = tf.concat((ypseq[..., self.Ny:], y, upseq[..., self.Nu:], u),
                           axis=-1)
        xplus = tf.concat((yplus, zplus), axis=-1)

        # Return output and states at the next time-step.
        return (y, xplus)

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
        for t in range(self.Np):
            yseq_interp.append(0.5*(yseq[..., t*self.p:(t+1)*self.p] + 
                                    yseq[..., (t+1)*self.p:(t+2)*self.p]))
        return tf.concat(yseq_interp, axis=-1)

    def get_config(self):
        return super().get_config()

class TwoReacModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, Np, fnn_dims, tworeac_parameters, model_type):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        (Ng, Nu) = (tworeac_parameters['Ng'], tworeac_parameters['Nu'])
        layer_input = tf.keras.Input(name='u', shape=(None, Nu))
        initial_state = tf.keras.Input(name='xGz0',
                                       shape=(Ng + Np*(Ng+Nu), ))

        # Dense layers for the NN.
        fnn_layers = []
        for dim in fnn_dims[1:-1]:
            fnn_layers.append(tf.keras.layers.Dense(dim, activation='tanh'))
        fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1],
                                                kernel_initializer='zeros',
                                                use_bias=False))

        # Build model depending on option.
        if model_type == 'black-box':
            tworeac_cell = BlackBoxCell(Np, Ng, Nu, fnn_layers)
        if model_type == 'hybrid':
            interp_layer = InterpolationLayer(p=Ng, Np=Np)
            tworeac_cell = TwoReacHybridCell(Np, interp_layer, fnn_layers,
                                             tworeac_parameters)
        #elif model_type == 'residual':
        #    tworeac_cell = TwoReacResidualCell(Np, fnn_layers,
        #                                       tworeac_parameters)

        # Construct the RNN layer and the computation graph.
        tworeac_layer = tf.keras.layers.RNN(tworeac_cell, return_sequences=True)
        layer_output = tworeac_layer(inputs=layer_input, 
                                     initial_state=[initial_state])
        # Construct model.
        super().__init__(inputs=[layer_input, initial_state], 
                         outputs=layer_output)

class CstrFlashModel(tf.keras.Model):
    """ Custom model for the CSTR FLASH model. """
    def __init__(self, Np, fnn_dims, xuyscales, 
                       cstr_flash_parameters, model_type):

        # Get the size and input layer, and initial state layer.
        (Ng, Ny, Nu) = (cstr_flash_parameters['Ng'],
                        cstr_flash_parameters['Ny'],
                        cstr_flash_parameters['Nu'])
        layer_input = tf.keras.Input(name='u', shape=(None, Nu))
        if model_type == 'black-box':
            initial_state = tf.keras.Input(name='yz0',
                                           shape=(Ny + Np*(Ny+Nu), ))
        else:
            initial_state = tf.keras.Input(name='xGz0',
                                           shape=(Ng + Np*(Ny+Nu), ))
        
        # Dense layers for the NN.
        fnn_layers = []
        for dim in fnn_dims[1:-1]:
            fnn_layers.append(tf.keras.layers.Dense(dim,
                                            activation='tanh'))
        fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1], 
                                            kernel_initializer='zeros'))

        # Build model depending on option.
        if model_type == 'black-box':
            cstr_flash_cell = BlackBoxCell(Np, Ny, Nu, fnn_layers)
        if model_type == 'hybrid':
            interp_layer = InterpolationLayer(p=Ny, Np=Np)
            cstr_flash_cell = CstrFlashHybridCell(Np, interp_layer, fnn_layers,
                                            xuyscales, cstr_flash_parameters)
        if model_type == 'grey-black':
            cstr_flash_cell = CstrFlashGreyBlackCell(Np, fnn_layers,
                                                     cstr_flash_parameters)

        # Construct the RNN layer and the computation graph.
        cstr_flash_layer = tf.keras.layers.RNN(cstr_flash_cell,
                                               return_sequences=True)
        layer_output = cstr_flash_layer(inputs=layer_input,
                                        initial_state=[initial_state])
        # Construct model.
        super().__init__(inputs=[layer_input, initial_state],
                         outputs=layer_output)