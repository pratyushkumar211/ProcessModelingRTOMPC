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
    def __init__(self, Np, Ny, Nu, hN_dims, tanhScale):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        z0 = tf.keras.Input(name='z0', shape=(Np*(Ny+Nu), ))

        def scaledtanh(x, a=tanhScale):
            num = tf.math.exp(a*x) - tf.math.exp(-a*x)
            den = tf.math.exp(a*x) + tf.math.exp(-a*x)
            return num/den

        # Dense layers for the NN.
        hN_layers = []
        for dim in hN_dims[1:-1]:
            hN_layers += [tf.keras.layers.Dense(dim, activation=scaledtanh)]
        hN_layers += [tf.keras.layers.Dense(hN_dims[-1])]

        # Build model.
        bbCell = BlackBoxCell(Np, Ny, Nu, hN_layers)

        # Construct the RNN layer and the computation graph.
        bbLayer = tf.keras.layers.RNN(bbCell, return_sequences=True)
        yseq = bbLayer(inputs=useq, initial_state=[z0])

        # Construct model.
        super().__init__(inputs=[useq, z0], outputs=yseq)

def create_bbmodel(*, Np, Ny, Nu, hN_dims, tanhScale):
    """ Create/compile the two reaction model for training. """
    model = BlackBoxModel(Np, Ny, Nu, hN_dims, tanhScale)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def train_bbmodel(*, model, epochs, batch_size, train_data, trainval_data, 
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
    model.fit(x=[train_data['inputs'], train_data['z0']], 
              y=train_data['outputs'], 
              epochs=epochs, batch_size=batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['z0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

def get_bbval_predictions(*, model, val_data, xuyscales, 
                             xinsert_indices, ckpt_path):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['z0']])

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
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['z0']], 
                                y=val_data['outputs'])

    # Return predictions and metric.
    return (val_predictions, val_metric)

def fnn(nn_input, nn_weights, tanh_scale):
    """ Compute the NN output. 
        Assume that the input is a vector with shape size 
        1, and return output with shape size 1.
        """
    def scaledtanh(x, a):
        num = np.exp(a*x) - np.exp(-a*x)
        den = np.exp(a*x) + np.exp(-a*x)
        return num/den

    nn_output = nn_input[:, np.newaxis]
    for i in range(0, len(nn_weights)-2, 2):
        (W, b) = nn_weights[i:i+2]
        nn_output = W.T @ nn_output + b[:, np.newaxis]
        nn_output = scaledtanh(nn_output, tanh_scale)
    (Wf, bf) = nn_weights[-2:]
    nn_output = (Wf.T @ nn_output + bf[:, np.newaxis])[:, 0]
    # Return.
    return nn_output

def get_bbpars_fxu_hx(*, train, parameters):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    Np = train['Np']
    hN_weights = train['trained_weights'][-1]
    xuyscales = train['xuyscales']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    Nx = Np*(Ny + Nu)
    ulb, uub = parameters['ulb'], parameters['uub']
    tanhScale = train['tanhScale']
    bb_pars = dict(Nx=Nx, Ny=Ny, Nu=Nu, Np=Np, xuyscales=xuyscales,
                   hN_weights=hN_weights, ulb=ulb, uub=uub, 
                   tanhScale=tanhScale)
    
    # Get function handles.
    fxu = lambda x, u: bb_fxu(x, u, bb_pars)
    hx = lambda x: bb_hx(x, bb_pars)

    # Return.
    return bb_pars, fxu, hx

def bb_fxu(z, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        z^+ = f_z(z, u) """

    # Extract a few parameters.
    Np = parameters['Np']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    hN_weights = parameters['hN_weights']
    tanhScale = parameters['tanhScale']
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
    y = fnn(z, hN_weights, tanhScale)
    
    # Concatenate.
    zplus = np.concatenate((z[Ny:Np*Ny], y, z[-(Np-1)*Nu:], u))

    # Scale back.
    zplus = zplus*zstd + zmean

    # Return the sum.
    return zplus

def bb_hx(z, parameters):
    """ Measurement function. """
    
    # Extract a few parameters.
    Np = parameters['Np']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    tanhScale = parameters['tanhScale']
    hN_weights = parameters['hN_weights']
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
    y = fnn(z, hN_weights, tanhScale)

    # Scale measurement back.
    y = y*ystd + ymean

    # Return the measurement.
    return y