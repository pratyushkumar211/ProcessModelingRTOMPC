"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf
from BlackBoxFuncs import fnnTF, fnn
from hybridid import SimData

class KoopmanCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell.
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']
    x_{kp} = [y;z;NN(y;z)]
    x_{kp}^+  = Ax_{kp} + Bu
    H = [I, 0]
    yz = H*x_{kp}
    Cell output: yz
    """
    def __init__(self, Nxkp, Np, Ny, Nu, A, B, **kwargs):
        super(KoopmanCell, self).__init__(**kwargs)
        self.Nxkp, self.Np, self.Ny, self.Nu = Nxkp, Np, Ny, Nu
        self.Nz = Np*(Ny + Nu)
        self.A, self.B = A, B

    @property
    def state_size(self):
        return self.Nxkp
    
    @property
    def output_size(self):
        return self.Ny + self.Nz

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nxkp)
            Dimension of input: (None, Nu)
            Dimension of output: (None, Ny + Nz)
        """
        # Extract important variables.
        [xkp] = states
        u = inputs

        # Get the state prediction at the next time step.
        xkplus = self.A(xkp) + self.B(u)
        
        # Get the output at the current timestep.
        [yz, _] = tf.split(xkp, 
                           [self.output_size, self.Nxkp-self.output_size], 
                           axis=-1)

        # Return output and states at the next time-step.
        return (yz, xkplus)

class KoopmanModel(tf.keras.Model):
    """ Custom model for the Deep Koopman operator model. """
    def __init__(self, Np, Ny, Nu, fNDims):
        
        # Get a few sizes.
        Nz = Np*(Ny+Nu)
        Nxkp = Ny + Nz + fNDims[-1]

        # Create inputs to the layers.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        yz0 = tf.keras.Input(name='yz0', shape=(Ny + Nz, ))
        
        # Dense layers for the Koopman lifting NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

        # Use the created layers to create the initial state.
        xkp0 = tf.concat((yz0, fnnTF(yz0, fNLayers)), axis=-1)

        # Custom weights for the linear dynamics in lifted space.
        A = tf.keras.layers.Dense(Nxkp, input_shape=(Nxkp, ),
                                  kernel_initializer='zeros',
                                  use_bias=False)
        B = tf.keras.layers.Dense(Nxkp, input_shape=(Nu, ),
                                  use_bias=False)
        
        # Build model depending on option.
        koopCell = KoopmanCell(Nxkp, Np, Ny, Nu, A, B)

        # Construct the RNN layer and the computation graph.
        koopLayer = tf.keras.layers.RNN(koopCell, return_sequences=True)
        yzseq = koopLayer(inputs=useq, initial_state=[xkp0])

        # Split to get just the measurements as an additional output.
        [y, _] = tf.split(yzseq, [Ny, Nz], axis=-1)
        # Construct model.
        super().__init__(inputs=[useq, yz0], outputs=[yzseq, y])

def create_koopmodel(*, Np, Ny, Nu, fNDims):
    """ Create/compile the two reaction model for training. """
    model = KoopmanModel(Np, Ny, Nu, fNDims)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error', 
                  loss_weights=[0., 1.])
    # Return the compiled model.
    return model

def train_koopmodel(*, model, epochs, batch_size, train_data, trainval_data, 
                       stdout_filename, ckpt_path):
    """ Function to train the NN controller. """
    # Std out.
    sys.stdout = open(stdout_filename, 'w')
    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    # Call the fit method to train.
    model.fit(x=[train_data['inputs'], train_data['yz0']], 
              y=[train_data['yz'], train_data['outputs']], 
              epochs=epochs, batch_size=batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['yz0']], 
                           [trainval_data['yz'], trainval_data['outputs']]),
            callbacks = [checkpoint_callback])

def get_koopval_predictions(*, model, val_data, xuyscales, 
                               xinsert_indices, ckpt_path):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['yz0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    ypredictions = model_predictions[1].squeeze()*ystd + ymean
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean

    # Get xpredictions.
    xpredictions = np.insert(ypredictions, xinsert_indices, np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    val_predictions = SimData(t=np.arange(0, Nt, 1), x=xpredictions, 
                              u=uval, y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['yz0']], 
                                y=[val_data['yz'], val_data['outputs']])

    # Return predictions and metric.
    return (val_predictions, val_metric)

def get_KoopmanModel_pars(*, train, plant_pars):
    """ Get the black-box parameter dict and function handles. """

    # Weights.
    Np = train['Np']
    fNWeights = train['trained_weights'][-1][:-2]
    A = train['trained_weights'][-1][-2].T
    B = train['trained_weights'][-1][-1].T

    # Scaling.
    xuyscales = train['xuyscales']

    # Sizes.
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    Nx = Ny + Np*(Ny + Nu) + train['fNDims'][-1]

    # Constraints.
    ulb, uub = plant_pars['ulb'], plant_pars['uub']

    # Create the dict.
    koop_pars = dict(Nx=Nx, Ny=Ny, Nu=Nu, Np=Np, xuyscales=xuyscales,
                     fNWeights=fNWeights, ulb=ulb, uub=uub, A=A, B=B)
    
    # Return.
    return koop_pars

def koop_fxu(xkp, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        xkp^+ = A*xkp + Bu """
    
    # Get A, B matrices.
    A, B = parameters['A'], parameters['B']
    umean, ustd = parameters['xuyscales']['uscale']

    # Scale control input.
    u = (u - umean)/ustd

    # Add extra axis.
    xkp, u = xkp[:, np.newaxis], u[:, np.newaxis]

    # Get current output.
    xkplus = A @ xkp + B @ u

    # Remove an axis.
    xkplus = xkplus[:, 0]

    # Return the sum.
    return xkplus

def koop_hx(xkp, parameters):
    """ Measurement function. """
    
    # Extract a few parameters.
    Ny = parameters['Ny']
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    
    # Add extra axis.
    y = xkp[:Ny]*ystd + ymean

    # Return the sum.
    return y

def get_koopman_xguess(train, plant_pars):

    # Get initial state.
    Np = train['Np']
    us = plant_pars['us']
    yindices = plant_pars['yindices']
    ys = plant_pars['xs'][yindices]
    yz0 = np.concatenate((np.tile(ys, (Np+1, )), 
                          np.tile(us, (Np, ))))

    # Scale initial state and get the lifted state.
    fNWeights = train['trained_weights'][-1][:-2]
    xuyscales = train['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
                            np.tile(umean, (Np, ))))
    yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
                            np.tile(ustd, (Np, ))))
    yz0 = (yz0 - yzmean)/yzstd
    xguess = np.concatenate((yz0, fnn(yz0, fNWeights, 1.)))

    # Return.
    return xguess

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