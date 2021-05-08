"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf
from hybridid import SimData

def fnnTF(nnInput, nnLayers):
    """ Compute the output of the feedforward network. """
    nnOutput = nnInput
    for layer in nnLayers:
        nnOutput = layer(nnOutput)
    return nnOutput

def tanh(x, TF=True, a=1):
    """ Custom tanh function. """
    if TF:
        num = tf.math.exp(a*x) - tf.math.exp(-a*x)
        den = tf.math.exp(a*x) + tf.math.exp(-a*x)
    else:
        num = np.exp(a*x) - np.exp(-a*x)
        den = np.exp(a*x) + np.exp(-a*x)
    # Return.
    return num/den

class BlackBoxCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']'
    x = [y', z']'
    y^+ = f_N(x, u)
    y  = [I, 0]x
    """
    def __init__(self, Np, Ny, Nu, fNLayers, **kwargs):
        super(BlackBoxCell, self).__init__(**kwargs)
        self.Np = Np
        self.Ny, self.Nu = Ny, Nu
        self.fNLayers = fNLayers

    @property
    def state_size(self):
        return self.Ny + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny
    
    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ny + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
            Dimension of output: (None, Ny)
        """
        
        # Extract important variables.
        [yz] = states
        u = inputs
        Np, Ny, Nu = self.Np, self.Ny, self.Nu

        # Extract elements of the state.
        if Np > 0:
            (y, ypseq, upseq) = tf.split(yz, [Ny, Np*Ny, Np*Nu],
                                         axis=-1)
        else:
            y = yz

        # Get the current output/state and the next time step.
        nnInput = tf.concat((yz, u), axis=-1)
        yplus = fnnTF(nnInput, self.fNLayers)

        if Np > 0:
            yzplus = tf.concat((yplus, ypseq[..., Ny:], y, upseq[..., Nu:], u),
                               axis=-1)
        else:
            yzplus = yplus

        # Return output and states at the next time-step.
        return (y, yzplus)

class BlackBoxModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, Np, Ny, Nu, fNDims):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        yz0 = tf.keras.Input(name='yz0', shape=(Ny+Np*(Ny+Nu), ))

        # Dense layers for the NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation=tanh)]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1], 
                                           kernel_initializer='zeros')]

        # Build model.
        bbCell = BlackBoxCell(Np, Ny, Nu, fNLayers)

        # Construct the RNN layer and the computation graph.
        bbLayer = tf.keras.layers.RNN(bbCell, return_sequences=True)
        yseq = bbLayer(inputs=useq, initial_state=[yz0])

        # Construct model.
        super().__init__(inputs=[useq, yz0], outputs=yseq)

def create_model(*, Np, Ny, Nu, fNDims):
    """ Create/compile the two reaction model for training. """
    model = BlackBoxModel(Np, Ny, Nu, fNDims)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def train_model(*, model, epochs, batch_size, train_data, trainval_data, 
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
              y=train_data['outputs'], 
              epochs=epochs, batch_size=batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['yz0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

def get_val_predictions(*, model, val_data, xuyscales, 
                           xinsert_indices, ckpt_path):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['yz0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    ypredictions = model_predictions.squeeze()*ystd + ymean
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean

    # Get xpredictions.
    xpredictions = np.insert(ypredictions, xinsert_indices, np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    val_predictions = SimData(t=np.arange(0, Nt, 1), x=xpredictions, 
                              u=uval, y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['yz0']], 
                                y=val_data['outputs'])

    # Return predictions and metric.
    return (val_predictions, val_metric)

def fnn(nnInput, nnWeights):
    """ Compute the NN output. """

    # Add one extra dimension.
    nnOutput = nnInput[:, np.newaxis]
    nnOutput = nnInput

    # Loop over layers.
    for i in range(0, len(nnWeights)-2, 2):
        W, b = nnWeights[i:i+2]
        nnOutput = W.T @ nnOutput + b[:, np.newaxis]
        nnOutput = tanh(nnOutput, TF=False)
    Wf, bf = nnWeights[-2:]
    
    # Return output in the same number of dimensions as input.
    nnOutput = Wf.T @ nnOutput + bf[:, np.newaxis]
    nnOutput = nnOutput[:, 0]

    # Return.
    return nnOutput

def get_bbnn_pars(*, train, plant_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['Np'] = train['Np']
    parameters['fNWeights'] = train['trained_weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    parameters['Ny'], parameters['Nu'] = Ny, Nu
    parameters['Nx'] = Ny + parameters['Np']*(Ny + Nu)

    # Constraints.
    parameters['ulb'] = plant_pars['ulb']
    parameters['uub'] = plant_pars['uub']
    
    # Return.
    return parameters

def bbnn_fxu(yz, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        yz^+ = f_z(yz, u) """

    # Extract parameters.
    Np, Ny, Nu = parameters['Np'], parameters['Ny'], parameters['Nu']

    # Get NN weights.
    fNWeights = parameters['fNWeights']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    yzmean = np.concatenate((ymean, 
                             np.tile(ymean, (Np, )), 
                             np.tile(umean, (Np, ))))
    yzstd = np.concatenate((ystd, 
                            np.tile(ystd, (Np, )), 
                            np.tile(ustd, (Np, ))))
    
    # Scale.
    yz = (yz - yzmean)/yzstd
    u = (u - umean)/ustd

    # Get current output.
    nnInput = np.concatenate((yz, u))
    yplus = fnn(nnInput, fNWeights)
    
    # Concatenate.
    if Np > 0:
        yzplus = np.concatenate((yplus, yz[Ny:(Np+1)*Ny], yz[-(Np-1)*Nu:], u))
    else:
        yzplus = yplus

    # Scale back.
    yzplus = yzplus*yzstd + yzmean

    # Return the sum.
    return yzplus

def bbnn_hx(yz, parameters):
    """ Measurement function. """
    
    # Extract a few parameters.
    Np, Ny = parameters['Np'], parameters['Ny']
    
    # Exctact measurement.
    if Np > 0:
        y = yz[:Ny]
    else:
        y = yz

    # Return the measurement.
    return y