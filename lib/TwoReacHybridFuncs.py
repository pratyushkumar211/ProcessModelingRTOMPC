# [depends] BlackBoxFuncs.py hybridId.py
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTf, fnn
from hybridId import SimData

class TwoReacHybridFullGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dx_g/dt  = f_g(x_g, u) + f_N(x_g, u)
    y = x_g
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
        return self.hyb_greybox_pars['Ny']

    def _fxu(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. 
            
            dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2 + r3
            dCc/dt = -F*Cc/V + r2 - r3
            """
        
        # Extract the parameters (nominal value of unmeasured disturbance).
        F = self.hyb_greybox_pars['ps'].squeeze()
        V = self.hyb_greybox_pars['V']

        # Get the output of the neural network.
        nnOutput = fnnTf(x, self.fNLayers)
        r1NN, r2NN = nnOutput[..., 0:1], nnOutput[..., 1:2]

        # Get scaling factors.
        # Such that scalings based on noisy measurements are used.
        xmean, xstd = self.xuyscales['yscale']
        Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
        umean, ustd = self.xuyscales['uscale']

        # Scale back to physical states and control inputs.
        x = x*xstd + xmean
        u = u*ustd + umean

        # Get the state and control.
        Ca, Cb, Cc = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = F*(Caf - Ca)/V - r1NN*Castd
        dCbbydt = -F*Cb/V + r1NN*Castd - 3*r2NN*Ccstd
        dCcbydt = -F*Cc/V + r2NN*Ccstd

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

class TwoReacHybridPartialGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dx_g/dt  = f_g(x_g, u) + f_N(x_g, u)
    y = x_g
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
            
            dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2 + r3
            dCc/dt = -F*Cc/V + r2 - r3
            """
        
        # Extract the parameters (nominal value of unmeasured disturbance).
        F = self.hyb_greybox_pars['ps'].squeeze()
        V = self.hyb_greybox_pars['V']

        # Get the output of the neural network.
        nnOutput = fnnTf(x, self.fNLayers)
        r1NN, r2NN = nnOutput[..., 0:1], nnOutput[..., 1:2]

        # Get scaling factors.
        # Such that scalings based on noisy measurements are used.
        xmean, xstd = self.xuyscales['yscale']
        Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
        umean, ustd = self.xuyscales['uscale']

        # Scale back to physical states and control inputs.
        x = x*xstd + xmean
        u = u*ustd + umean

        # Get the state and control.
        Ca, Cb, Cc = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = F*(Caf - Ca)/V - r1NN*Castd
        dCbbydt = -F*Cb/V + r1NN*Castd - 3*r2NN*Ccstd
        dCcbydt = -F*Cc/V + r2NN*Ccstd

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
    """ Custom model for the Two reaction system. """
    
    def __init__(self, fNDims, xuyscales, hyb_greybox_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Sizes.
        Nx, Nu = hyb_greybox_pars['Nx'], hyb_greybox_pars['Nu']
        
        # Input layers to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        x0 = tf.keras.Input(name='x0', shape=(Nx, ))

        # Dense layers for the NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

        # Get the tworeac cell object.
        tworeacCell = TwoReacHybridCell(fNLayers, xuyscales, hyb_greybox_pars)

        # Construct the RNN layer and get the predicted xseq.
        tworeacLayer = tf.keras.layers.RNN(tworeacCell, return_sequences = True)
        xseq = tworeacLayer(inputs = useq, initial_state = [x0])

        # Construct model.
        super().__init__(inputs = [useq, x0], outputs = xseq)

def create_model(*, fNDims, xuyscales, hyb_greybox_pars):
    """ Create and compile the two reaction model for training. """

    # Create a model.
    model = TwoReacModel(fNDims, xuyscales, hyb_greybox_pars)

    # Compile the model.
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return.
    return model

def train_model(*, model, epochs, batch_size, train_data, 
                          trainval_data, stdout_filename, ckpt_path):
    """ Function to train the hybrid model. """

    # Std out.
    sys.stdout = open(stdout_filename, 'w')

    # Create the checkpoint callback.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    # Call the fit method to train.
    model.fit(x = [train_data['inputs'], train_data['x0']], 
              y = train_data['outputs'], 
              epochs = epochs, batch_size = batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['x0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

def get_val_predictions(*, model, val_data, xuyscales, 
                           xinsert_indices, ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['x0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    ypredictions = model_predictions.squeeze(axis=0)*ystd + ymean
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean

    # Get xpredictions.
    xpredictions = np.insert(ypredictions, xinsert_indices, np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    tval = np.arange(0, Nt*Delta, Delta)
    val_predictions = SimData(t=tval, x=xpredictions, 
                              u=uval, y=ypredictions)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data['x0']], 
                                y = val_data['outputs'])

    # Return.
    return (val_predictions, val_metric)

def get_hybrid_pars(*, train, hyb_greybox_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['fNWeights'] = train['trained_weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    parameters['Nx'] = hyb_greybox_pars['Nx']
    parameters['Ny'] = hyb_greybox_pars['Ny']
    parameters['Nu'] = hyb_greybox_pars['Nu']
    parameters['Np'] = train['Np']

    # Constraints.
    parameters['ulb'] = hyb_greybox_pars['ulb']
    parameters['uub'] = hyb_greybox_pars['uub']
    
    # Greybox model parameters.
    parameters['ps'] = hyb_greybox_pars['ps']
    parameters['V'] = hyb_greybox_pars['V']

    # Sample time.
    parameters['Delta'] = hyb_greybox_pars['Delta']

    # Return.
    return parameters

def fxup(x, u, p, parameters, xuyscales, fNWeights):
    """ Partial grey-box ODE function. """

    # Extract the plant states into meaningful names.
    Ca, Cb, Cc = x[0:1], x[1:2], x[2:3]
    Caf = u[0:1]
    F = p.squeeze()

    # Parameters.
    V = parameters['V']

    # Get the scales.
    xmean, xstd = xuyscales['yscale']
    Castd, Cbstd, Ccstd = xstd[0:1], xstd[1:2], xstd[2:3]
    umean, ustd = xuyscales['uscale']
    
    # Scale state, inputs, for the NN.
    x = (x - xmean)/xstd
    u = (u - umean)/ustd
    nnOutput = fnn(x, fNWeights)
    r1NN, r2NN = nnOutput[0:1], nnOutput[1:2]

    # Write the ODEs.
    dCabydt = F*(Caf-Ca)/V - r1NN*Castd
    dCbbydt = -F*Cb/V + r1NN*Castd - 3*r2NN*Ccstd
    dCcbydt = -F*Cc/V + r2NN*Ccstd

    # Scale.
    xdot = mpc.vcat([dCabydt, dCbbydt, dCcbydt])

    # Return.
    return xdot

def hybrid_fxup(x, u, p, parameters):
    """ Hybrid model. """

    # Get NN weights.
    fNWeights = parameters['fNWeights']
    Delta = parameters['Delta']

    # Get scaling.
    xuyscales = parameters['xuyscales']

    # Get k1, k2, k3, and k4.
    k1 = fxup(x, u, p, parameters, xuyscales, fNWeights)
    k2 = fxup(x + Delta*(k1/2), u, p, parameters, xuyscales, fNWeights)
    k3 = fxup(x + Delta*(k2/2), u, p, parameters, xuyscales, fNWeights)
    k4 = fxup(x + Delta*k3, u, p, parameters, xuyscales, fNWeights)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Return the sum.
    return xplus

def hybrid_hx(x):
    """ Measurement function. """
    return x