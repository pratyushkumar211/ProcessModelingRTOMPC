# [depends] BlackBoxFuncs.py hybridId.py
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTf, fnn, createDenseLayers
from hybridId import SimData

class InterpolationLayer(tf.keras.layers.Layer):
    """
    The layer to perform interpolation for RK4 predictions.
    Nvar: Number of variables.
    Np + 1: Number of variables.
    """
    def __init__(self, Nvar, Np, trainable=False, name=None):
        super(InterpolationLayer, self).__init__(trainable, name)
        self.Nvar = Nvar
        self.Np = Np

    def call(self, yseq):
        """ The main call function of the interpolation layer.
            yseq is of dimension: (None, (Np+1)*Nvar)
            Return y of dimension: (None, Np*Nvar)
        """
        yseq_interp = []
        for t in range(self.Np):
            yseq_interp += [0.5*(yseq[..., t*self.Nvar:(t+1)*self.Nvar] + 
                                 yseq[..., (t+1)*self.Nvar:(t+2)*self.Nvar])]
        # Return.
        return tf.concat(yseq_interp, axis=-1)

def getInterpolatedVals(yseq, Nvar, Np):
    """ y is of dimension: (None, (Np+1)*Nvar)
        Return y of dimension: (None, Np*Nvar). """
    yseq_interp = []
    for t in range(Np):
        yseq_interp += [0.5*(yseq[t*Nvar:(t+1)*Nvar] + 
                             yseq[(t+1)*Nvar:(t+2)*Nvar])]
    # Return.
    return np.concatenate(yseq_interp)

class ReacPartialGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell:
    dx_g/dt  = f_g(x_g, u) + (chosen functions).
    y = x_g
    r1 = NN1(Ca)
    r2 = NN2(Cb)
    r3 = NN3(z)
    """
    def __init__(self, r1Layers, r2Layers, r3Layers, Np, interpLayer,
                       xuyscales, hyb_partialgb_pars, **kwargs):
        super(ReacPartialGbCell, self).__init__(**kwargs)

        # r1, r2, and r3 layers.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.r3Layers = r3Layers
        self.interpLayer = interpLayer

        # Sizes.
        assert Np > 0
        self.Np = Np
        self.Nx = hyb_partialgb_pars['Nx']
        self.Nu = hyb_partialgb_pars['Nu']
        self.Ny = hyb_partialgb_pars['Ny']

        # xuyscales and hybrid parameters.
        self.xuyscales = xuyscales
        self.hyb_partialgb_pars = hyb_partialgb_pars

    @property
    def state_size(self):
        """ State size of the model. """
        return self.Nx + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        """ Number of outputs of the model. """
        return self.Ny

    def _fxzu(self, x, z, u):
        """ dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2 + f_N(z)
        """
        
        # Extract the parameters.
        F = self.hyb_greybox_pars['ps'].squeeze()
        V = self.hyb_greybox_pars['V']

        # Get scaling factors.
        ymean, ystd = self.xuyscales['yscale']
        Castd, Cbstd = ystd[0:1], ystd[1:2]
        umean, ustd = self.xuyscales['uscale']

        # Get the output of the neural network.
        Ca, Cb = x[..., 0:1], x[..., 1:2]
        r1 = fnnTf(Ca, self.r1Layers)*Castd
        r2 = fnnTf(Cb, self.r2Layers)*Cbstd
        r3 = fnnTf(z, self.r3Layers)*Cbstd

        # Scale back to physical states and control inputs.
        x = x*ystd + ymean
        u = u*ustd + umean

        # Get the state and control (after scaling to physical variables).
        Ca, Cb = x[..., 0:1], x[..., 1:2]
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = F*(Caf - Ca)/V - r1
        dCbbydt = -F*Cb/V + r1 - 3*r2 + r3

        # Scaled derivate.
        xdot = tf.concat([dCabydt, dCbbydt], axis=-1)/ystd

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
        """

        # Extract states/inputs.
        [xz] = states
        u = inputs

        # Extract y, ypast, and upast.
        (x, z) = tf.split(xz, [self.Nx, self.Np*(self.Nx + self.Nu)], 
                          axis=-1)
        (xpseq, upseq) = tf.split(z, [self.Np*self.Nx, self.Np*self.Nu],
                                  axis=-1)

        # Sample time.
        Delta = self.hyb_greybox_pars['Delta']
        
        # Get k1.
        k1 = self._fxzu(x, z, u)
        
        # Get k2.
        xpseq_k2k3 = self.interpLayer(tf.concat((xpseq, x), axis=-1))
        z = tf.concat((xpseq_k2k3, upseq), axis=-1)
        k2 = self._fxzu(x + Delta*(k1/2), z, u)

        # Get k3.
        k3 = self._fxzu(x + Delta*(k2/2), z, u)

        # Get k4.
        xpseq_k4 = tf.concat((xpseq[..., self.Nx:], x), axis=-1)
        z = tf.concat((xpseq_k4, upseq), axis=-1)
        k4 = self._fxzu(x + Delta*k3, z, u)
        
        # Get the xzplus at the next time step.
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((xpseq[..., self.Nx:], x, upseq[..., self.Nu:], u))
        xzplus = tf.concat((xplus, zplus), axis=-1)

        # Return current output and states at the next time point.
        return (x, xzplus)

class ReacPartialGbModel(tf.keras.Model):
    """
    Partial model for the reaction system.
    """
    
    def __init__(self, r1Dims, r2Dims, r3Dims,
                       Np, xuyscales, hyb_partialgb_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Sizes.
        Nx, Nu = hyb_partialgb_pars['Nx'], hyb_partialgb_pars['Nu']
        Nz = Np*(Nx + Nu)

        # Input layers to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        xz0 = tf.keras.Input(name='xz0', shape=(Nx + Nz, ))

        # Layers for the reactions.
        r1Layers = createDenseLayers(r1Dims)
        r2Layers = createDenseLayers(r2Dims)
        r3Layers = createDenseLayers(r3Dims)

        # Interpolation layer for RK4 predictions.
        interpLayer = InterpolationLayer(Nx, Np)

        # Get the reac cell object.
        reacCell = ReacPartialGbCell(r1Layers, r2Layers, r3Layers, Np,
                                     interpLayer, xuyscales, hyb_partialgb_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        yseq = reacLayer(inputs = useq, initial_state = [x0])

        # Construct model.
        super().__init__(inputs = [useq, x0], outputs = yseq)

        # Store the layers (to extract weights for use in numpy).
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.r3Layers = r3Layers

def create_model(*, r1Dims, r2Dims, r3Dims, Np,
                    xuyscales, hyb_partialgb_pars):
    """ Create and compile the two reaction model for training. """

    # Create a model.
    model = ReacPartialGbModel(r1Dims, r2Dims, r3Dims, Np, xuyscales,
                               hyb_partialgb_pars)

    # Create a loss function and compile the model.
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
    model.fit(x = [train_data['inputs'], train_data['xz0']], 
              y = train_data['outputs'], 
              epochs = epochs, batch_size = batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['xz0']], 
                            trainval_data['outputs']),
            callbacks = [checkpoint_callback])

def get_val_predictions(*, model, val_data, xuyscales,
                           ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['xz0']])

    # Compute val metric.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data['xz0']], 
                                y = val_data['outputs'])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']

    # Validation input sequence.
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean
    Nt = uval.shape[0]

    # Get y and x predictions.
    if model.estCLayers is not None:

        # Sizes. 
        Nx = model.reacCell.hyb_fullgb_pars['Nx']
        Ny = model.reacCell.hyb_fullgb_pars['Ny']

        # Get the y predictions.
        ymean = np.concatenate((ymean, ymean[-1:], ymean[-1:]))
        ystd = np.concatenate((ystd, ystd[-1:], ystd[-1:]))
        ypredictions = model_predictions.squeeze(axis=0)*ystd + ymean
        xpredictions = ypredictions[:, :Nx]
        xpredictions_Cc = np.concatenate((np.tile(np.nan, (Nt, Ny)), 
                                          ypredictions[:, -1:]), axis=1)
        ypredictions = ypredictions[:, :Ny]
        
    else:
        ypredictions = model_predictions.squeeze(axis=0)*ystd + ymean
        xpredictions = np.insert(ypredictions, [2], np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    tval = np.arange(0, Nt*Delta, Delta)
    pval = np.tile(np.nan, (Nt, 1))
    val_prediction_list = [SimData(t=tval, x=xpredictions, 
                                   u=uval, y=ypredictions, p=pval)]

    # Create one more Simdata object to plot the Cc predictions 
    # using the estimator NN.
    if model.estCLayers is not None:

        # Sizes.
        Nu = model.reacCell.hyb_fullgb_pars['Nu']
        
        # Model predictions.
        uval = np.tile(np.nan, (Nt, Nu))
        ypredictions = np.tile(np.nan, (Nt, Ny))
        val_prediction_list += [SimData(t=tval, x=xpredictions_Cc,
                                        u=uval, y=ypredictions, p=pval)]

    # Return.
    return (val_prediction_list, val_metric)

def get_hybrid_pars(*, train, hyb_partialgb_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['r1Weights'] = train['trained_r1Weights'][-1]
    parameters['r2Weights'] = train['trained_r2Weights'][-1]
    parameters['r3Weights'] = train['trained_r3Weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    parameters['Nx'] = hyb_partialgb_pars['Nx']
    parameters['Ny'] = hyb_partialgb_pars['Ny']
    parameters['Nu'] = hyb_partialgb_pars['Nu']
    parameters['Np'] = train['Np']

    # Constraints.
    parameters['ulb'] = hyb_partialgb_pars['ulb']
    parameters['uub'] = hyb_partialgb_pars['uub']
    
    # Greybox model parameters.
    parameters['ps'] = hyb_partialgb_pars['ps']
    parameters['V'] = hyb_partialgb_pars['V']

    # Sample time.
    parameters['Delta'] = hyb_partialgb_pars['Delta']

    # Return.
    return parameters

def fxup(x, z, u, p, parameters):
    """ Partial grey-box ODE function. """

    # Extract disturbance.
    F = p.squeeze()

    # Parameters.
    V = parameters['V']
    r1Weights = parameters['r1Weights']
    r2Weights = parameters['r2Weights']
    r3Weights = parameters['r3Weights']
    
    # Get the scales.
    ymean, ystd = xuyscales['yscale']
    Castd, Cbstd = ystd[0:1], ystd[1:2]
    umean, ustd = xuyscales['uscale']
    
    # Scale states and z for NNs.
    x = (x - ymean)/ystd
    Np = parameters['Np']
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                            np.tile(ustd, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                           np.tile(ustd, (Np, ))))
    z = (z - zmean)/zstd

    # Get NN reaction rates.
    Ca, Cb = x[0:1], x[1:2]
    r1 = fnn(Ca, r1Weights)*Castd
    r2 = fnn(Cb, r2Weights)*Cbstd
    r3 = fnn(z, r3Weights)*Cbstd

    # Scale states back to physical.
    x = x*ystd + ymean
    Ca, Cb = x[0:1], x[1:2]
    Caf = u[0:1]

    # Write the ODEs.
    dCabydt = F*(Caf-Ca)/V - r1
    dCbbydt = -F*Cb/V + r1 - 3*r2 + r3

    # xdot.
    xdot = mpc.vcat([dCabydt, dCbbydt])

    # Return.
    return xdot

def hybrid_fxup(xz, u, p, parameters):
    """ Hybrid model. """

    # Sizes.
    Nx = parameters['Nx']
    Nu = parameters['Nu']
    Np = parameters['Np']

    # x, z, xpseq, and upseq.
    x, z = xz[:Nx], xz[Nx:]
    xpseq, upseq = z[:Nx*Np], z[Nx*Np:]

    # Get NN weights.
    Delta = parameters['Delta']

    # Get k1.
    k1 = fxup(x, z, u, p, parameters)

    # Get k2.
    xpseq_k2k3 = getInterpolatedVals(np.concatenate((xpseq, x)), Nx, Np)
    k2 = fxup(x + Delta*(k1/2), z, u, p, parameters)

    # Get k3.
    k3 = fxup(x + Delta*(k2/2), z, u, p, parameters)

    # Get k4.
    xpseq_k4 = np.concatenate((xpseq[Nx:], x))
    k4 = fxup(x + Delta*k3, z, u, p, parameters)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Get zplus and state at the next time step.
    zplus = np.concatenate((z[Nx:], x, z[Nx*Np+Nu:], u))
    xzplus = np.concatenate((xplus, zplus))

    # Return.
    return xzplus

def hybrid_hx(xz):
    """ Measurement function. """
    Ny = parameters['Ny']
    y = xz[:Ny]
    # Return measurement.
    return y