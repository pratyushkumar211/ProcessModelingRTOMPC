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

def createDenseLayers(nnDims):
    """ Create dense layers based on the feed-forward NN layer dimensions.
        nnDims: List that contains the dimensions of the feed forward NN.
        nnLayers: Output of the feedforward NN.
    """
    nnLayers = []
    for dim in nnDims[1:-1]:
        nnLayers += [tf.keras.layers.Dense(dim, activation=tanh)]
    nnLayers += [tf.keras.layers.Dense(nnDims[-1])]
    # Return.
    return nnLayers

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
    
    # Final layer.
    nnOutput = Wf.T @ nnOutput + bf[:, np.newaxis]

    # Output with same number of dimensions as the input.
    nnOutput = nnOutput[:, 0]

    # Return.
    return nnOutput

def get_weights_from_tflayers(layers):
    """ Function to get the weights from a list of layers. """
    
    # Get weights from layer lists.
    weights = []
    for layer in layers:
        weights += layer.get_weights()
    
    # Return.
    return weights

class BlackBoxCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']' (x := z)

    z^+ = f(z, u) (State evolution, 
                   includes index shifting + measurement NN)
    y = h(z) (Measurement equation, this is a NN).

    """
    def __init__(self, Np, Ny, Nu, fNLayers, **kwargs):
        super(BlackBoxCell, self).__init__(**kwargs)

        # Get sizes and NN layers.
        self.Np = Np
        self.Ny, self.Nu = Ny, Nu
        self.fNLayers = fNLayers

    @property
    def state_size(self):
        return self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny
    
    def call(self, inputs, states):
        """ Call function of the Black-Box RNN cell.
            Dimension of states: (None, Np*(Ny + Nu))
            Dimension of input: (None, Nu)
            Dimension of output: (None, Ny)
        """
        
        # Extract states and input.
        [z] = states
        u = inputs

        # Extract past measurements and inputs.
        ypseq, upseq = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
                                axis=-1)

        # Get the current output.
        y = fnnTf(z, self.fNLayers)

        # Get the state at the next time-step.
        zplus = tf.concat((ypseq[..., self.Ny:], y, upseq[..., self.Nu:], u),
                           axis=-1)

        # Return y and xplus.
        return (y, zplus)

class BlackBoxModel(tf.keras.Model):
    """ Black-box model. """

    def __init__(self, Np, Ny, Nu, fNDims):
        """ Create dense layers for the NN and construct the overall model. """

        # Check for number of inputs/outputs.
        assert Np > 0, "Zero past inputs and outputs provided to the model. "

        # Get z0 and useq.
        z0 = tf.keras.Input(name='z0', shape=(Np*(Ny+Nu), ))
        useq = tf.keras.Input(name='u', shape=(None, Nu))

        # Dense layers for the NN.
        fNLayers = createDenseLayers(fNDims)

        # Create a black-box model.
        bbCell = BlackBoxCell(Np, Ny, Nu, fNLayers)

        # Construct the RNN layer and the computation graph.
        bbLayer = tf.keras.layers.RNN(bbCell, return_sequences=True)
        yseq = bbLayer(inputs=useq, initial_state=[z0])

        # Construct model.
        super().__init__(inputs=[useq, z0], outputs=yseq)

def create_model(*, Np, Ny, Nu, fNDims):
    """ Create and compile a model for training. """
    
    # Create.
    model = BlackBoxModel(Np, Ny, Nu, fNDims)
    
    # Compile.
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Return.
    return model

def train_model(*, model, epochs, batch_size, 
                   train_data, trainval_data, 
                   stdout_filename, ckpt_path):
    """ Train model. """

    # Std out.
    sys.stdout = open(stdout_filename, 'w')

    # Create a checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=1)

    # Fit model.
    model.fit(x=[train_data['useq'], train_data['z0']], 
              y=train_data['yseq'], 
              epochs=epochs, batch_size=batch_size,
              validation_data=([trainval_data['useq'], trainval_data['z0']], 
                                trainval_data['yseq']),
              callbacks=[checkpoint_callback])

def get_val_predictions(*, model, val_data, xuyscales,
                           unmeasXIndices, ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    modelPredictions = model.predict(x=[val_data['useq'], val_data['z0']])

    # Get scaling.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']

    # Get validation predictions.
    uval = val_data['useq'].squeeze(axis=0)*ustd + umean
    ypred = modelPredictions.squeeze(axis=0)*ystd + ymean
    xpred = np.insert(ypred, unmeasXIndices, np.nan, axis=1)
    Nt = uval.shape[0]
    tval = np.arange(0, Nt*Delta, Delta)
    pval = np.tile(np.nan, (Nt, 1))

    # Collect the predictions in a simdata format.
    valPredData = SimData(t=tval, x=xpred, u=uval, y=ypred, p=pval)

    # Return.
    return valPredData

def get_bbnn_pars(*, train, plant_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['Np'] = train['Np']
    parameters['fNWeights'] = train['fNWeights']
    parameters['xuyscales'] = train['xuyscales']

    # Get sizes.
    parameters['Ny'], parameters['Nu'] = plant_pars['Ny'], plant_pars['Nu']
    parameters['Nx'] = train['Np']*(plant_pars['Ny'] + plant_pars['Nu'])

    # Constraints.
    parameters['ulb'] = plant_pars['ulb']
    parameters['uub'] = plant_pars['uub']

    # Sample time.
    parameters['Delta'] = plant_pars['Delta']

    # Return.
    return parameters

def bbnn_fxu(z, u, parameters):
    """ Function describing the state-space dynamics 
        of the black-box neural network. 
        z^+ = f(z, u) """

    # Sizes.
    Np, Ny, Nu = parameters['Np'], parameters['Ny'], parameters['Nu']

    # Get NN weights.
    fNWeights = parameters['fNWeights']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                            np.tile(umean, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                           np.tile(ustd, (Np, ))))
    
    # Scale.
    z = (z - zmean)/zstd
    u = (u - umean)/ustd

    # Get current output.
    y = fnn(z, fNWeights)
    
    # Concatenate.
    zplus = np.concatenate((z[Ny:Np*Ny], y, 
                            z[Np*Ny+Nu:], u))

    # Scale back.
    zplus = zplus*zstd + zmean

    # Return.
    return zplus

def bbnn_hx(z, parameters):
    """ Measurement function. """
    
    # Number of past measurements/inputs.
    Np = parameters['Np']

    # Get NN weights.
    fNWeights = parameters['fNWeights']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                            np.tile(umean, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                           np.tile(ustd, (Np, ))))
    
    # Scale.
    z = (z - zmean)/zstd

    # Get current output.
    y = fnn(z, fNWeights)

    # Scale back.
    y = y*ystd + ymean

    # Return.
    return y