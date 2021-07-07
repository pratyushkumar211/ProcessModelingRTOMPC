# [depends] BlackBoxFuncs.py hybridid.py
"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTF, fnn
from hybridid import SimData

class TwoReacHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']';
    x = [xG', z'];
    dxG/dt  = fG(xG, u) + f_N(xG, z, u)
    y = xG
    """
    def __init__(self, fNLayers, xuyscales, hyb_greybox_pars, **kwargs):
        super(TwoReacHybridCell, self).__init__(**kwargs)
        self.fNLayers = fNLayers
        self.xuyscales = xuyscales
        self.hyb_greybox_pars = hyb_greybox_pars

    @property
    def state_size(self):
        return self.hyb_greybox_pars['Nx']
    
    @property
    def output_size(self):
        return self.hyb_greybox_pars['Nx']

    def _fxu(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. 
            
            dCa/dt = (Caf-Ca)/tau - r1
            dCb/dt = -Cb/tau + r1 - 3*r2
            dCc/dt = -Cc/tau + r2
            """
        
        # Extract the parameters.
        tau = self.hyb_greybox_pars['ps'].squeeze()
        
        # Get the output of the neural network.
        nnOutput = fnnTF(x, self.fNLayers)
        r1, r2 = nnOutput[..., 0:1], nnOutput[..., 1:2]

        # Scale back to physical states.
        xmean, xstd = self.xuyscales['yscale']
        Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
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

        # NN contributions.
        fN = tf.concat([-r1, (r1*Castd - 3*r2*Ccstd)/Cbstd, r2], axis=-1)
        
        # Final derivative.
        xdot += fN

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
        Delta = self.hyb_greybox_pars['Delta']        

        # Get k1, k2, k3, and k4.
        k1 = self._fxu(x, u)
        k2 = self._fxu(x + Delta*(k1/2), u)
        k3 = self._fxu(x + Delta*(k2/2), u)
        k4 = self._fxu(x + Delta*k3, u)
        
        # Get the current output/state and the next time step.
        y = x
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Return output and states at the next time-step.
        return (y, xplus)

class TwoReacModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, fNDims, xuyscales, hyb_greybox_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        Nx, Nu = hyb_greybox_pars['Nx'], hyb_greybox_pars['Nu']
        layer_input = tf.keras.Input(name='u', shape=(None, Nu))
        initial_state = tf.keras.Input(name='x0', shape=(Nx, ))

        # Dense layers for the NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

        # Get the tworeac cell object.
        tworeac_cell = TwoReacHybridCell(fNLayers, xuyscales, hyb_greybox_pars)

        # Construct the RNN layer and the computation graph.
        tworeac_layer = tf.keras.layers.RNN(tworeac_cell, return_sequences=True)
        layer_output = tworeac_layer(inputs=layer_input, 
                                     initial_state=[initial_state])
        # Construct model.
        super().__init__(inputs=[layer_input, initial_state], 
                         outputs=layer_output)

def create_model(*, fNDims, xuyscales, hyb_greybox_pars):
    """ Create/compile the two reaction model for training. """
    model = TwoReacModel(fNDims, xuyscales, hyb_greybox_pars)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def train_model(*, model, epochs, batch_size, train_data, 
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

def get_val_predictions(*, model, val_data, xuyscales, 
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

def get_hybrid_pars(*, train, hyb_greybox_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['fNWeights'] = train['trained_weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    Nx, Ny = hyb_greybox_pars['Nx'], hyb_greybox_pars['Ny']
    Nu = hyb_greybox_pars['Nu']
    parameters['Nx'], parameters['Ny'], parameters['Nu'] = Nx, Ny, Nu
    parameters['Np'] = train['Np']

    # Constraints.
    parameters['ulb'] = hyb_greybox_pars['ulb']
    parameters['uub'] = hyb_greybox_pars['uub']
    
    # Time constant/sample time.
    parameters['ps'] = hyb_greybox_pars['ps']
    parameters['Delta'] = hyb_greybox_pars['Delta']

    # Return.
    return parameters

def fxup(x, u, p, xuyscales, fNWeights):
    """ Partial grey-box ODE function. """

    # Extract the plant states into meaningful names.
    (Ca, Cb, Cc) = x[0:1], x[1:2], x[2:3]
    Caf = u[0:1]
    tau = p.squeeze()

    # Get the scales.
    xmean, xstd = xuyscales['yscale']
    Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
    umean, ustd = xuyscales['uscale']
    
    # Scale state, inputs, for the NN.
    x = (x - xmean)/xstd
    u = (u - umean)/ustd
    nnOutput = fnn(x, fNWeights)
    r1, r2 = nnOutput[0:1], nnOutput[1:2]

    # Write the ODEs.
    dCabydt = (Caf-Ca)/tau
    dCbbydt = -(Cb/tau)
    dCcbydt = -(Cc/tau)

    # Scale.
    xdot = mpc.vcat([dCabydt, dCbbydt, dCcbydt])
    fN = np.concatenate((-r1, (r1*Castd - 3*r2*Ccstd)/Cbstd, r2))*xstd
    xdot += fN

    # Return.
    return xdot

def hybrid_fxup(x, u, p, parameters):
    """ Function describing the dynamics 
        of the Two reac hybrid model. """

    # Get NN weights.
    fNWeights = parameters['fNWeights']
    Delta = parameters['Delta']

    # Get scaling.
    xuyscales = parameters['xuyscales']

    # Get k1, k2, k3, and k4.
    k1 = fxup(x, u, p, xuyscales, fNWeights)
    k2 = fxup(x + Delta*(k1/2), u, p, xuyscales, fNWeights)
    k3 = fxup(x + Delta*(k2/2), u, p, xuyscales, fNWeights)
    k4 = fxup(x + Delta*k3, u, p, xuyscales, fNWeights)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Return the sum.
    return xplus

def hybrid_hx(x):
    """ Measurement function. """
    return x