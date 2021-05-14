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
from hybridid import SimData

class TwoReacHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']';
    x = [xG', z'];
    dxG/dt  = fG(xG, u) + f_N(xG, z, u)
    y = xG
    """
    def __init__(self, fNLayers, xuyscales, greybox_pars, **kwargs):
        super(TwoReacHybridCell, self).__init__(**kwargs)
        self.fNLayers = fNLayers
        self.xuyscales = xuyscales
        self.greybox_pars = greybox_pars

    @property
    def state_size(self):
        return self.greybox_pars['Nx']
    
    @property
    def output_size(self):
        return self.greybox_pars['Nx']

    def _fgreybox(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. 
            
            dCa/dt = (Caf-Ca)/tau 
            dCb/dt = -Cb/tau
            dCc/dt = -Cc/tau
            """
        
        # Extract the parameters.
        tau = self.greybox_pars['ps'].squeeze()
        
        # Scale back to physical states.
        xmean, xstd = self.xuyscales['xscale']
        umean, ustd = self.xuyscales['uscale']
        x = x*xstd + xmean
        u = u*ustd + umean

        # Get the state and control.
        Ca, Cb, Cc = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = (Caf - Ca)/tau
        dCbbydt = -(Cb/tau)
        dCcbydt = -(Cc/tau)

        # Scaled derivate.
        xdot = tf.concat([dCabydt, dCbbydt, dCcbydt], axis=-1)/xstd

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """
        # Extract states/inputs.
        [x] = states
        u = inputs

        # Sample time.
        Delta = self.greybox_pars['Delta']        

        # Get k1.
        nnInput = tf.concat((x, u), axis=-1)
        k1 = self._fgreybox(x, u) + fnnTF(nnInput, self.fNLayers)
                
        # Get k2.
        nnInput = tf.concat((x + Delta*(k1/2), u), axis=-1)
        k2 = self._fgreybox(x + Delta*(k1/2), u) + fnnTF(nnInput, self.fNLayers)

        # Get k3.
        nnInput = tf.concat((x + Delta*(k2/2), u), axis=-1)
        k3 = self._fgreybox(x + Delta*(k2/2), u) + fnnTF(nnInput, self.fNLayers)

        # Get k4.
        nnInput = tf.concat((x + Delta*k3, u), axis=-1)
        k4 = self._fgreybox(x + Delta*k3, u) + fnnTF(nnInput, self.fNLayers)
        
        # Get the current output/state and the next time step.
        y = x
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Return output and states at the next time-step.
        return (y, xplus)

class TwoReacModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, fNDims, xuyscales, greybox_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        Nx, Nu = greybox_pars['Nx'], greybox_pars['Nu']
        layer_input = tf.keras.Input(name='u', shape=(None, Nu))
        initial_state = tf.keras.Input(name='x0', shape=(Nx, ))

        # Dense layers for the NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

        # Get the tworeac cell object.
        tworeac_cell = TwoReacHybridCell(fNLayers, xuyscales, greybox_pars)

        # Construct the RNN layer and the computation graph.
        tworeac_layer = tf.keras.layers.RNN(tworeac_cell, return_sequences=True)
        layer_output = tworeac_layer(inputs=layer_input, 
                                     initial_state=[initial_state])
        # Construct model.
        super().__init__(inputs=[layer_input, initial_state], 
                         outputs=layer_output)

def create_tworeac_model(*, fNDims, xuyscales, greybox_pars):
    """ Create/compile the two reaction model for training. """
    model = TwoReacModel(fNDims, xuyscales, greybox_pars)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def train_hybrid_model(*, model, epochs, batch_size, train_data, 
                          trainval_data, stdout_filename, ckpt_path):
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
    model.fit(x=[train_data['inputs'], train_data['x0']], 
              y=train_data['outputs'], 
              epochs=epochs, batch_size=batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['x0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

def get_hybrid_predictions(*, model, val_data, xuyscales, 
                             xinsert_indices, ckpt_path):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['x0']])

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
    val_metric = model.evaluate(x=[val_data['inputs'], val_data['x0']], 
                                y=val_data['outputs'])

    # Return predictions and metric.
    return (val_predictions, val_metric)

def fnn(nnInput, nnWeights):
    """ Compute the NN output. """

    # Check input dimensions. 
    if nnInput.ndim == 1:
        nnOutput = nnInput[:, np.newaxis]
    else:
        nnOutput = nnInput

    # Loop over layers.
    for i in range(0, len(nnWeights)-2, 2):
        (W, b) = nnWeights[i:i+2]
        nnOutput = W.T @ nnOutput + b[:, np.newaxis]
        nnOutput = np.tanh(nnOutput)
    (Wf, bf) = nnWeights[-2:]
    
    # Return output in the same number of dimensions as input.
    nnOutput = Wf.T @ nnOutput + bf[:, np.newaxis]
    if nnInput.ndim == 1:
        nnOutput = nnOutput[:, 0]

    # Return.
    return nnOutput

def get_tworeacHybrid_pars(*, train, greybox_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['fNWeights'] = train['trained_weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    Nx, Ny, Nu = greybox_pars['Nx'], greybox_pars['Ny'], greybox_pars['Nu']
    parameters['Nx'], parameters['Ny'], parameters['Nu'] = Nx, Ny, Nu
    parameters['Np'] = train['Np']

    # Constraints.
    parameters['ulb'] = greybox_pars['ulb']
    parameters['uub'] = greybox_pars['uub']
    
    # Time constant/sample time.
    parameters['tau'] = greybox_pars['ps'].squeeze()
    parameters['Delta'] = greybox_pars['Delta']

    # Return.
    return parameters

def fgreybox(x, u, tau):
    """ Partial grey-box ODE function. """

    # Extract the plant states into meaningful names.
    (Ca, Cb, Cc) = x[0:3]
    Caf = u[0]

    # Write the ODEs.
    dCabydt = (Caf-Ca)/tau
    dCbbydt = -(Cb/tau)
    dCcbydt = -(Cc/tau)

    # Return. 
    return np.array([dCabydt, dCbbydt, dCcbydt])

def tworeacHybrid_fxu(x, u, parameters):
    """ Function describing the dynamics 
        of the Two reac hybrid model. """

    # Extract parameters.
    Nx, Nu = parameters['Nx'], parameters['Nu']

    # Get NN weights.
    fNWeights = parameters['fNWeights']
    tau = parameters['tau']
    Delta = parameters['Delta']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    
    # Scale state and inputs.
    x = (x - xmean)/xstd
    u = (u - umean)/ustd
    
    # Get k1.
    nnInput = np.concatenate((x, u))
    k1 = fgreybox(x*xstd + xmean, u*ustd + umean, tau)/xstd
    k1 += fnn(nnInput, fNWeights)
    
    # Get k2.
    nnInput = np.concatenate((x + Delta*(k1/2), u))
    k2 = fgreybox((x + Delta*(k1/2))*xstd + xmean, u*ustd + umean, tau)/xstd
    k2 += fnn(nnInput, fNWeights)

    # Get k3.
    nnInput = np.concatenate((x + Delta*(k2/2), u))
    k3 = fgreybox((x + Delta*(k2/2))*xstd + xmean, u*ustd + umean, tau)/xstd
    k3 += fnn(nnInput, fNWeights)

    # Get k4.
    nnInput = np.concatenate((x + Delta*k3, u))
    k4 = fgreybox((x + Delta*k3)*xstd + xmean, u*ustd + umean, tau)/xstd
    k4 += fnn(nnInput, fNWeights)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Scale back.
    xplus = xplus*xstd + xmean

    # Return the sum.
    return xplus

def tworeacHybrid_hx(x):
    """ Measurement function. """
    return x