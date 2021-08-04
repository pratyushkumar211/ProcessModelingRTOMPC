# [depends] hybridId.py
import sys
import numpy as np
import tensorflow as tf
from hybridId import SimData

def tanh(x, tF=True, a=1):
    """ Custom tanh function. The input to tanh 
        can be scaled if required. """
    if tF:
        num = tf.math.exp(a*x) - tf.math.exp(-a*x)
        den = tf.math.exp(a*x) + tf.math.exp(-a*x)
    else:
        num = np.exp(a*x) - np.exp(-a*x)
        den = np.exp(a*x) + np.exp(-a*x)
    # Return.
    return num/den

def fnnTf(nnInput, nnLayers):
    """ Compute the output of a feedforward network, 
        with tensorflow inputs. """
    nnOutput = nnInput
    for layer in nnLayers:
        nnOutput = layer(nnOutput)
    # Return the final output.
    return nnOutput

def fnn(nnInput, nnWeights):
    """ Compute the output of a feedforward network, 
        with inputs and weights as numpy arrays. """

    # Check that the input has only one dimension, and add one extra.
    assert nnInput.ndim == 1 
    nnOutput = nnInput[:, np.newaxis]

    # Loop over layers.
    for i in range(0, len(nnWeights)-2, 2):
        W, b = nnWeights[i:i+2]
        nnOutput = W.T @ nnOutput + b[:, np.newaxis]
        nnOutput = tanh(nnOutput, tF=False)
    Wf, bf = nnWeights[-2:]
    
    # Return output in the same number of dimensions as input.
    nnOutput = Wf.T @ nnOutput + bf[:, np.newaxis]
    nnOutput = nnOutput[:, 0]

    # Return.
    return nnOutput

class BlackBoxCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']'
    yz = [y', z']'
    y^+ = f_N(yz, u)
    yz^+ = f_z(y, z, u) (Index shifting)
    y  = [I, 0, 0, ...]yz
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

        # Extract elements of the state.
        if self.Np > 0:
            (y, ypseq, upseq) = tf.split(yz, 
                                         [self.Ny, self.Np*self.Ny, 
                                          self.Np*self.Nu],
                                         axis=-1)
        else:
            y = yz

        # Get the current output/state at the next time step.
        nnInput = tf.concat((yz, u), axis=-1)
        yplus = fnnTf(nnInput, self.fNLayers)

        # Two cases, depending on if past data is included.
        if self.Np > 0:
            yzplus = tf.concat((yplus, ypseq[..., Ny:], y, upseq[..., Nu:], u),
                               axis=-1)
        else:
            yzplus = yplus

        # Return output and states at the next time step.
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
        fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

        # Build model.
        bbCell = BlackBoxCell(Np, Ny, Nu, fNLayers)

        # Construct the RNN layer and the computation graph.
        bbLayer = tf.keras.layers.RNN(bbCell, return_sequences=True)
        yseq = bbLayer(inputs=useq, initial_state=[yz0])

        # Construct model.
        super().__init__(inputs=[useq, yz0], outputs=yseq)

def create_model(*, Np, Ny, Nu, fNDims):
    """ Create and compile model for training. """
    
    # Create Model.
    model = BlackBoxModel(Np, Ny, Nu, fNDims)
    
    # Compile the NN model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Return.
    return model

def train_model(*, model, epochs, batch_size, train_data, trainval_data, 
                   stdout_filename, ckpt_path):
    """ Function to train model. """

    # Std out.
    sys.stdout = open(stdout_filename, 'w')

    # Create a checkpoint callback.
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
                           xinsert_indices, ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['yz0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    ypredictions = model_predictions*ystd + ymean
    uval = val_data['inputs']*ustd + umean

    # Get xpredictions.
    xpredictions = np.insert(ypredictions, xinsert_indices, np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    tval = np.arange(0, Nt, Delta)
    val_predictions = SimData(t=tval, x=xpredictions, 
                              u=uval, y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['yz0']], 
                                y=val_data['outputs'])

    # Return predictions and metric.
    return (val_predictions, val_metric)

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

    # Sample time.
    parameters['Delta'] = plant_pars['Delta']

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
    yzmean = np.concatenate((np.tile(ymean, (Np + 1, )), 
                             np.tile(umean, (Np, ))))
    yzstd = np.concatenate((np.tile(ystd, (Np + 1, )), 
                            np.tile(ustd, (Np, ))))
    
    # Scale.
    yz = (yz - yzmean)/yzstd
    u = (u - umean)/ustd

    # Get current output.
    nnInput = np.concatenate((yz, u))
    yplus = fnn(nnInput, fNWeights)
    
    # Concatenate.
    if Np > 0:
        yzplus = np.concatenate((yplus, yz[2*Ny:(Np+1)*Ny], yz[0:Ny], 
                                 yz[-(Np-1)*Nu:], u))
    else:
        yzplus = yplus

    # Scale back.
    yzplus = yzplus*yzstd + yzmean

    # Return.
    return yzplus

def bbnn_hx(yz, parameters):
    """ Measurement function. """
    
    # Exctact measurement.
    Ny = parameters['Ny']
    y = yz[:Ny]

    # Return.
    return y