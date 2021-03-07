"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf


def fnn(nn_input, nn_layers):
    """ Compute the output of the feedforward network. """
    nn_output = nn_input
    for layer in nn_layers:
        nn_output = layer(nn_output)
    return nn_output

# class TwoReacHybridCell(tf.keras.layers.AbstractRNNCell):
#     """
#     RNN Cell
#     dxG/dt  = fG(xG, u) + f_N(xG, y_{k-N_p:k-1}, u_{k-N_p:k-1}), y = xG
#     """
#     def __init__(self, Np, interp_layer, fnn_layers,
#                        xuscales, tworeac_parameters, **kwargs):
#         super(TwoReacHybridCell, self).__init__(**kwargs)
#         self.Np = Np
#         self.interp_layer = interp_layer
#         self.fnn_layers = fnn_layers
#         self.xuscales = xuscales
#         self.tworeac_parameters = tworeac_parameters
#         (self.Ng, self.Nu) = (tworeac_parameters['Ng'],
#                               tworeac_parameters['Nu'])

#     @property
#     def state_size(self):
#         return self.Ng + self.Np*(self.Ng + self.Nu)
    
#     @property
#     def output_size(self):
#         return self.Ng        

#     def _fg(self, x, u):
#         """ Function to compute the 
#             derivative (RHS of the ODE)
#             for the two reaction model. """

#         # Extract the parameters.
#         k1 = self.tworeac_parameters['k1']
#         tau = self.tworeac_parameters['ps'].squeeze()
        
#         # Scale back to physical states.
#         xmean, xstd = self.xuscales['xscale']
#         umean, ustd = self.xuscales['uscale']
#         x = x*xstd + xmean
#         u = u*ustd + umean

#         # Get the state and control.
#         (Ca, Cb) = (x[..., 0:1], x[..., 1:2])
#         Caf = u[..., 0:1]
        
#         # Write the ODEs.
#         dCabydt = (Caf - Ca)/tau - k1*Ca
#         dCbbydt = k1*Ca - Cb/tau

#         # Scaled derivate.
#         xdot = tf.concat([dCabydt, dCbbydt], axis=-1)/xstd

#         # Return the derivative.
#         return xdot

#     def _fnn(self, xg, z):
#         """ Compute the output of the feedforward network. """
#         fnn_output = tf.concat((xg, z), axis=-1)
#         for layer in self.fnn_layers:
#             fnn_output = layer(fnn_output)
#         return fnn_output

#     def call(self, inputs, states):
#         """ Call function of the hybrid RNN cell.
#             Dimension of states: (None, Ng + Np*(Ng + Nu))
#             Dimension of input: (None, Nu)
#         """
#         # Extract variables.
#         [xGz] = states
#         [xG, z] = tf.split(xGz, [self.Ng, self.Np*(self.Ng+self.Nu)],
#                            axis=-1)
#         (ypseq, upseq) = tf.split(z, [self.Np*self.Ng, self.Np*self.Nu],
#                                   axis=-1)
#         u = inputs
#         Delta = self.tworeac_parameters['Delta']
        
#         # Get k1.
#         k1 = self._fg(xG, u) + self._fnn(xG, z)
        
#         # Interpolate for k2 and k3.
#         ypseq_interp = self.interp_layer(tf.concat((ypseq, xG), axis=-1))
#         z = tf.concat((ypseq_interp, upseq), axis=-1)
        
#         # Get k2.
#         k2 = self._fg(xG + Delta*(k1/2), u) + self._fnn(xG + Delta*(k1/2), z)

#         # Get k3.
#         k3 = self._fg(xG + Delta*(k2/2), u) + self._fnn(xG + Delta*(k2/2), z)

#         # Get k4.
#         ypseq_shifted = tf.concat((ypseq[..., self.Ng:], xG), axis=-1)
#         z = tf.concat((ypseq_shifted, upseq), axis=-1)
#         k4 = self._fg(xG + Delta*k3, u) + self._fnn(xG + Delta*k3, z)
        
#         # Get the current output/state and the next time step.
#         y = xG
#         xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
#         zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
#         xplus = tf.concat((xGplus, zplus), axis=-1)

#         # Return output and states at the next time-step.
#         return (y, xplus)

# class CstrFlashHybridCell(tf.keras.layers.AbstractRNNCell):
#     """
#     RNN Cell
#     dxG/dt  = fG(xG, u) + f_N(xG, y_{k-N_p:k-1}, u_{k-N_p:k-1}), y = xG
#     """
#     def __init__(self, Np, interp_layer, fnn_layers,
#                        xuyscales, cstr_flash_parameters, **kwargs):
#         super(CstrFlashHybridCell, self).__init__(**kwargs)
#         self.Np = Np
#         self.interp_layer = interp_layer
#         self.fnn_layers = fnn_layers
#         self.parameters = cstr_flash_parameters
#         self.xuyscales = xuyscales
#         (self.Ng, self.Ny, self.Nu) = (cstr_flash_parameters['Ng'],
#                                        cstr_flash_parameters['Ny'],
#                                        cstr_flash_parameters['Nu'])
    
#     @property
#     def state_size(self):
#         return self.Ng + self.Np*(self.Ny + self.Nu)
    
#     @property
#     def output_size(self):
#         return self.Ny        

#     def _fg(self, x, u):
#         """ Function to compute the 
#             derivative (RHS of the ODE)
#             for the two reaction model. """
        
#         # Extract the parameters.
#         alphaA = self.parameters['alphaA']
#         alphaB = self.parameters['alphaB']
#         pho = self.parameters['pho']
#         Cp = self.parameters['Cp']
#         Ar = self.parameters['Ar']
#         Ab = self.parameters['Ab']
#         kr = self.parameters['kr']
#         kb = self.parameters['kb']
#         delH1 = self.parameters['delH1']
#         E1byR = self.parameters['E1byR']
#         k1star = self.parameters['k1star']
#         Td = self.parameters['Td']
#         ps = self.parameters['ps']

#         # Scale back to physical states.
#         xmean, xstd = self.xuyscales['xscale']
#         umean, ustd = self.xuyscales['uscale']
#         x = x*xstd + xmean
#         u = u*ustd + umean

#         # Extract the plant states into meaningful names.
#         (Hr, CAr) = x[..., 0:1], x[..., 1:2]
#         (CBr, Tr) = x[..., 2:3], x[..., 3:4] 
#         (Hb, CAb) = x[..., 4:5], x[..., 5:6]
#         (CBb, Tb) = x[..., 6:7], x[..., 7:8] 
#         (F, Qr) = u[..., 0:1], u[..., 1:2]
#         (D, Qb) = u[..., 2:3], u[..., 3:4]
#         (CAf, Tf) = ps[0], ps[1]
        
#         # The flash vapor phase mass fractions.
#         den = alphaA*CAb + alphaB*CBb
#         CAd = alphaA*CAb/den
#         CBd = alphaB*CBb/den

#         # The outlet mass flow rates.
#         Fr = kr*tf.math.sqrt(Hr)
#         Fb = kb*tf.math.sqrt(Hb)

#         # Rate constant and reaction rate.
#         k1 = k1star*tf.math.exp(-E1byR/Tr)
#         r1 = k1*CAr
        
#         # Write the CSTR odes.
#         dHrbydt = (F + D - Fr)/Ar
#         dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
#         dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
#         dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
#         dTrbydt = dTrbydt + (r1*delH1)/(pho*Cp)
#         dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

#         # Write the flash odes.
#         dHbbydt = (Fr - Fb - D)/Ab
#         dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
#         dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
#         dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

#         # Get the scaled derivative.
#         xdot = tf.concat([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
#                           dHbbydt, dCAbbydt, dCBbbydt, dTbbydt], axis=-1)/xstd

#         # Return the derivative.
#         return xdot

#     def _hxg(self, xg):
#         """ Measurement function. """
#         yindices = self.parameters['yindices']
#         # Return the measured grey-box states.
#         return tf.gather(xg, yindices, axis=-1)

#     def _fnn(self, xg, z, u):
#         """ Compute the output of the feedforward network. """
#         fnn_output = tf.concat((xg, z, u), axis=-1)
#         for layer in self.fnn_layers:
#             fnn_output = layer(fnn_output)
#         # Return.
#         return fnn_output

#     def call(self, inputs, states):
#         """ Call function of the hybrid RNN cell.
#             Dimension of states: (None, Ng + Np*(Ny + Nu))
#             Dimension of input: (None, Nu)
#         """
#         # Extract variables.
#         [xGz] = states
#         [xG, z] = tf.split(xGz, [self.Ng, self.Np*(self.Ny+self.Nu)],
#                            axis=-1)
#         (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
#                                   axis=-1)
#         u = inputs

#         # Extract parameters.
#         Delta = self.parameters['Delta']
#         xscale, yscale = self.xuyscales['xscale'], self.xuyscales['yscale']
#         xmean, xstd = xscale
#         ymean, ystd = yscale

#         # Get k1.
#         k1 = self._fg(xG, u) + self._fnn(xG, z, u)

#         # Interpolate for k2 and k3.
#         hxG = self._hxg(xG)
#         ypseq_interp = self.interp_layer(tf.concat((ypseq, hxG), axis=-1))
#         z = tf.concat((ypseq_interp, upseq), axis=-1)
        
#         # Get k2.
#         k2 = self._fg(xG + Delta*(k1/2), u) + self._fnn(xG + Delta*(k1/2), z, u)

#         # Get k3.
#         k3 = self._fg(xG + Delta*(k2/2), u) + self._fnn(xG + Delta*(k2/2), z, u)

#         # Get k4.
#         ypseq_shifted = tf.concat((ypseq[..., self.Ny:], hxG), axis=-1)
#         z = tf.concat((ypseq_shifted, upseq), axis=-1)
#         k4 = self._fg(xG + Delta*k3, u) + self._fnn(xG + Delta*k3, z, u)
        
#         # Get the current output/state and the next time step.
#         y = (self._hxg(xG*xstd + xmean) - ymean)/ystd
#         xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
#         zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
#         xplus = tf.concat((xGplus, zplus), axis=-1)

#         # Measurement/Grey-box state.
#         output = tf.concat((y, xG), axis=-1)
        
#         # Return output and states at the next time-step.
#         return (output, xplus)

# class CstrFlashGreyBlackCell(tf.keras.layers.AbstractRNNCell):
#     """
#     RNN Cell
#     z = y_{k-N_p:k-1}, u_{k-N_p:k-1}
#     xG^+  = f_N(xG, z, u), y = xG
#     z^+ = z
#     """
#     def __init__(self, Np, fnn_layers, cstr_flash_parameters, **kwargs):
#         super(CstrFlashGreyBlackCell, self).__init__(**kwargs)
#         self.Np = Np
#         self.fnn_layers = fnn_layers
#         self.yindices = cstr_flash_parameters['yindices']
#         (self.Ng, self.Ny, self.Nu) = (cstr_flash_parameters['Ng'],
#                                        cstr_flash_parameters['Ny'],
#                                        cstr_flash_parameters['Nu'])

#     @property
#     def state_size(self):
#         return self.Ng + self.Np*(self.Ny + self.Nu)
    
#     @property
#     def output_size(self):
#         return self.Ng

#     def _fnn(self, xg, z, u):
#         """ Compute the output of the feedforward network. """
#         fnn_output = tf.concat((xg, z, u), axis=-1)
#         for layer in self.fnn_layers:
#             fnn_output = layer(fnn_output)
#         return fnn_output

#     def _hxg(self, xg):
#         """ Measurement function. """
#         # Return the measured grey-box states.
#         return tf.gather(xg, self.yindices, axis=-1)

#     def call(self, inputs, states):
#         """ Call function of the hybrid RNN cell.
#             Dimension of states: (None, Ng + Np*(Ny + Nu))
#             Dimension of input: (None, Nu)
#         """
#         # Extract variables.
#         [xGz] = states
#         [xG, z] = tf.split(xGz, [self.Ng, self.Np*(self.Ny+self.Nu)],
#                            axis=-1)
#         (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
#                                   axis=-1)
#         u = inputs
        
#         # Get shifted y.
#         hxG = self._hxg(xG)
#         ypseq_shifted = tf.concat((ypseq[..., self.Ny:], hxG), axis=-1)
        
#         # Get the current output/state and the next time step.
#         y = hxG
#         xGplus = self._fnn(xG, z, u)
#         zplus = tf.concat((ypseq_shifted, upseq[..., self.Nu:], u), axis=-1)
#         xplus = tf.concat((xGplus, zplus), axis=-1)

#         # Return output and states at the next time-step.
#         return (y, xplus)

class BlackBoxCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']'
    z^+ = f_z(z, u)
    y  = h_N(z)
    """
    def __init__(self, Np, Ny, Nu, fnn_layers, **kwargs):
        super(BlackBoxCell, self).__init__(**kwargs)
        self.Np = Np
        self.Ny, self.Nu = Ny, Nu
        self.fnn_layers = fnn_layers

    @property
    def state_size(self):
        return self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny
    
    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Np*(Ny + Nu))
            Dimension of input: (None, Nu)
            Dimension of output: (None, Ny)
        """
        # Extract important variables.
        [z] = states
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
                                  axis=-1)
        u = inputs

        # Get the current output/state and the next time step.
        y = fnn(z, self.fnn_layers)
        zplus = tf.concat((ypseq[..., self.Ny:], y, upseq[..., self.Nu:], u),
                           axis=-1)

        # Return output and states at the next time-step.
        return (y, zplus)

class BlackBoxModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, Np, Ny, Nu, hN_dims):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        z0 = tf.keras.Input(name='z0', shape=(Np*(Ny+Nu), ))

        #def simplesig(x):
        #    return 1./(1. + tf.math.exp(-x))

        # Dense layers for the NN.
        hN_layers = []
        for dim in hN_dims[1:-1]:
            hN_layers += [tf.keras.layers.Dense(dim, activation='tanh')]
        hN_layers += [tf.keras.layers.Dense(hN_dims[-1])]

        # Build model.
        bbCell = BlackBoxCell(Np, Ny, Nu, hN_layers)

        # Construct the RNN layer and the computation graph.
        bbLayer = tf.keras.layers.RNN(bbCell, return_sequences=True)
        yseq = bbLayer(inputs=useq, initial_state=[z0])

        # Construct model.
        super().__init__(inputs=[useq, z0], outputs=yseq)

# class InterpolationLayer(tf.keras.layers.Layer):
#     """
#     The layer to perform interpolation for RK4 predictions.
#     """
#     def __init__(self, p, Np, trainable=False, name=None):
#         super(InterpolationLayer, self).__init__(trainable, name)
#         self.p = p
#         self.Np = Np

#     def call(self, yseq):
#         """ The main call function of the interpolation layer. 
#         y is of dimension: (None, (Np+1)*p)
#         Return y of dimension: (None, Np*p)
#         """
#         yseq_interp = []
#         for t in range(self.Np):
#             yseq_interp.append(0.5*(yseq[..., t*self.p:(t+1)*self.p] + 
#                                     yseq[..., (t+1)*self.p:(t+2)*self.p]))
#         return tf.concat(yseq_interp, axis=-1)

#     def get_config(self):
#         return super().get_config()

class TwoReacModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, Np, fnn_dims, xuyscales, tworeac_parameters, model_type):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        Ng, Ny, Nu = (tworeac_parameters['Ng'], 
                      tworeac_parameters['Ny'],
                      tworeac_parameters['Nu'])
        layer_input = tf.keras.Input(name='u', shape=(None, Nu))

        # Dense layers for the NN.
        fnn_layers = []
        for dim in fnn_dims[1:-1]:
            fnn_layers.append(tf.keras.layers.Dense(dim, activation='tanh'))
        fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1]))

        # Build model depending on option.
        if model_type == 'black-box':
            initial_state = tf.keras.Input(name='yz0',
                                       shape=(Ny + Np*(Ny+Nu), ))
            tworeac_cell = BlackBoxCell(Np, Ny, Nu, fnn_layers)
        if model_type == 'hybrid':
            initial_state = tf.keras.Input(name='xGz0',
                                       shape=(Ng + Np*(Ny+Nu), ))
            interp_layer = InterpolationLayer(p=Ng, Np=Np)
            tworeac_cell = TwoReacHybridCell(Np, interp_layer, fnn_layers,
                                             xuyscales, tworeac_parameters)

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

# class KoopmanCell(tf.keras.layers.AbstractRNNCell):
#     """
#     RNN Cell.
#     #z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}'];
#     #x_{kp} = NN(y, z)
#     x_{kp}^+  = Ax_{kp} + Bu
#     #[y';z']^+ = Hx_{kp}^+
#     [y;z] = H*xkp
#     #y = [I, 0]*[y;z]
#     """
#     def __init__(self, Nxkp, Ny, Nz, A, B, **kwargs):
#         super(KoopmanCell, self).__init__(**kwargs)
#         self.Nxkp, self.Ny, self.Nz = Nxkp, Ny, Nz
#         self.A, self.B = A, B

#     @property
#     def state_size(self):
#         return self.Nxkp
    
#     @property
#     def output_size(self):
#         return self.Ny + self.Nz

#     def call(self, inputs, states):
#         """ Call function of the hybrid RNN cell.
#             Dimension of states: (None, Nxkp)
#             Dimension of input: (None, Nu)
#             Dimension of output: (None, Nz)
#         """
#         # Extract important variables.
#         [xkp] = states
#         u = inputs

#         # Get the state prediction at the next time step.
#         xkplus = self.A(xkp) + self.B(u)
        
#         # Get the output at the current timestep.
#         [yz, _] = tf.split(xkp, [self.output_size, self.Nxkp-self.output_size], 
#                            axis=-1)

#         # Return output and states at the next time-step.
#         return (yz, xkplus)

# class KoopmanModel(tf.keras.Model):
#     """ Custom model for the Deep Koopman operator model. """
#     def __init__(self, Np, Ny, Nu, fnn_dims, kernel_reg=1e-4):
        
#         # Get a few sizes.
#         Nz = Np*(Ny+Nu)
#         Nxkp = Ny + Nz + fnn_dims[-1]

#         # Create inputs to the layers.
#         useq = tf.keras.Input(name='u', shape=(None, Nu))
#         yz0 = tf.keras.Input(name='yz0', shape=(Ny + Nz, ))
        
#         # Dense layers for the Koopman lifting NN.
#         fnn_layers = []
#         for dim in fnn_dims[1:-1]:
#             fnn_layers.append(tf.keras.layers.Dense(dim, 
#                             activation='tanh',
#                     kernel_regularizer=tf.keras.regularizers.l2(kernel_reg)))
#         fnn_layers.append(tf.keras.layers.Dense(fnn_dims[-1],
#                     kernel_regularizer=tf.keras.regularizers.l2(kernel_reg)))

#         # Use the created layers to create the initial state.
#         xkp0 = tf.concat((yz0, self._fnn(yz0, fnn_layers)), axis=-1)

#         # Custom weights for the linear dynamics in lifted space.
#         A = tf.keras.layers.Dense(Nxkp, input_shape=(Nxkp, ),
#                         kernel_initializer='zeros',
#                         kernel_regularizer=tf.keras.regularizers.l2(kernel_reg),
#                         use_bias=False)
#         B = tf.keras.layers.Dense(Nxkp, input_shape=(Nu, ),
#                     kernel_regularizer=tf.keras.regularizers.l2(kernel_reg), 
#                         use_bias=False)
#         #H = tf.keras.layers.Dense(Ny+Nz, input_shape=(Nxkp, ),
#         #            kernel_regularizer=tf.keras.regularizers.l2(kernel_reg), 
#         #                use_bias=False)
        
#         # Build model depending on option.
#         koopman_cell = KoopmanCell(Nxkp, Ny, Nz, A, B)

#         # Construct the RNN layer and the computation graph.
#         koopman_layer = tf.keras.layers.RNN(koopman_cell, return_sequences=True)
#         yzseq = koopman_layer(inputs=useq, initial_state=[xkp0])
        
#         # Get the list of outputs.
#         y, _ = tf.split(yzseq, [Ny, Nz], axis=-1)

#         # Get the list of inputs and outputs of the model.
#         inputs = [useq, yz0]
#         outputs = [yzseq, y]

#         # Construct model.
#         super().__init__(inputs=inputs, outputs=outputs)

#     def _fnn(self, yz, fnn_layers):
#         """ Compute the output of the feedforward network. """
#         fnn_output = yz
#         for layer in fnn_layers:
#             fnn_output = layer(fnn_output)
#         return fnn_output

# class KoopmanEncDecCell(tf.keras.layers.AbstractRNNCell):
#     """
#     RNN Cell.
#     z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}'];
#     x_{kp} = ENC_NN(y, z)
#     x_{kp}^+  = Ax_{kp} + Bu
#     #[y';z']^+ = Hx_{kp}^+
#     [y;z] = DEC_NN(xkp)
#     """
#     def __init__(self, Nxkp, Ny, Nz, A, B, dec_layers, **kwargs):
#         super(KoopmanEncDecCell, self).__init__(**kwargs)
#         self.Nxkp, self.Ny, self.Nz = Nxkp, Ny, Nz
#         self.A, self.B = A, B
#         self.dec_layers = dec_layers

#     @property
#     def state_size(self):
#         return self.Nxkp
    
#     @property
#     def output_size(self):
#         return self.Ny + self.Nz

#     def call(self, inputs, states):
#         """ Call function of the hybrid RNN cell.
#             Dimension of states: (None, Nxkp)
#             Dimension of input: (None, Nu)
#             Dimension of output: (None, Nz)
#         """
#         # Extract important variables.
#         [xkp] = states
#         u = inputs

#         # Get the state prediction at the next time step.
#         xkplus = self.A(xkp) + self.B(u)
        
#         # Get the output at the current timestep.
#         yz = fnn(xkp, self.dec_layers)

#         # Return output and states at the next time-step.
#         return (yz, xkplus)

# class KoopmanEncDecModel(tf.keras.Model):
#     """ Custom model for the Deep Koopman Encoder Decoder model. """
#     def __init__(self, Np, Ny, Nu, enc_dims, dec_dims):
        
#         # Get a few sizes.
#         Nz = Np*(Ny + Nu)
#         Nxkp = enc_dims[-1]

#         # Create the encoder and decoder layers.
#         enc_layers = self.get_layers(enc_dims)
#         dec_layers = self.get_layers(dec_dims)
        
#         # Create inputs to the layers.
#         useq = tf.keras.Input(name='u', shape=(None, Nu))
#         yz0 = tf.keras.Input(name='yz0', shape=(Ny + Nz, ))
#         xkp0 = fnn(yz0, enc_layers)

#         # Custom weights for the linear dynamics in lifted space.
#         A = tf.keras.layers.Dense(Nxkp, input_shape=(Nxkp, ),
#                                   kernel_initializer='zeros',
#                                   use_bias=False)
#         B = tf.keras.layers.Dense(Nxkp, input_shape=(Nu, ),
#                                   use_bias=False)
        
#         # Build model depending on option.
#         koopman_cell = KoopmanEncDecCell(Nxkp, Ny, Nz, A, B, dec_layers)

#         # Construct the RNN layer and the computation graph.
#         koopman_layer = tf.keras.layers.RNN(koopman_cell, return_sequences=True)
#         yzseq = koopman_layer(inputs=useq, initial_state=[xkp0])
        
#         # Get the list of outputs.
#         y, _ = tf.split(yzseq, [Ny, Nz], axis=-1)

#         # Get the list of inputs and outputs of the model.
#         inputs = [useq, yz0]
#         outputs = [yzseq, y]

#         # Construct model.
#         super().__init__(inputs=inputs, outputs=outputs)

#     def get_layers(self, layer_dims):
#         # Give the layer dimensions, get a list of dense layers.
#         fnn_layers = []
#         for dim in layer_dims[1:-1]:
#             fnn_layers.append(tf.keras.layers.Dense(dim, activation='tanh'))
#         fnn_layers.append(tf.keras.layers.Dense(layer_dims[-1]))
#         # Return the list of layers.
#         return fnn_layers