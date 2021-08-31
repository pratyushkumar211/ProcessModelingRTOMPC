# [depends] BlackBoxFuncs.py hybridId.py
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTf, fnn
from hybridId import SimData

def createDenseLayers(nnDims):
    """ Create dense layers based on the feed-forward NN layer dimensions.
        nnDims: List that contains the dimensions of the forward NN.
    """
    nnLayers = []
    for dim in nnDims[1:-1]:
        nnLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
    nnLayers += [tf.keras.layers.Dense(nnDims[-1])]
    # Return.
    return nnLayers

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

class FullGbLoss(tf.keras.losses.Loss):

    """ 
        Loss function for the full grey-box model. 

        lam: scaling factor for the extra cost term.

        All the indices are lists.
        yi: indices relevant to predict the measurements.

        unmeasGbPredi: indices relevant to predict the unmeasured grey-box
        states by the predictor model.

        unmeasGbEsti: indices relevant to predict the unmeasured grey-box states
        by the estimator model.

    """

    def __init__(self, lam, yi, unmeasGbPredi, unmeasGbEsti):

        # Lambda, yi, unmeasured grey-box, and estimator indices.
        self.lam = lam
        self.yi = yi
        self.unmeasGbPredi = unmeasGbPredi
        self.unmeasGbEsti = unmeasGbEsti

        # Initialize.
        super(FullGbLoss, self).__init__()

    def call(self, y_true, y_pred):
        """ Write the call function. """
        
        # Custom MSE.
        ei = self.yi[-1]
        y_pred = y_pred[..., :ei+1]

        # Prediction of unmeasured grey-box states 
        # by the predictor model.
        si, ei = self.unmeasGbPredi[0], self.unmeasGbPredi[-1]
        y_unmeasGbPred = y_pred[..., si:ei+1]

        # Prediction of unmeasured grey-box states 
        # by the estimator model.
        si, ei = self.unmeasGbEsti[0], self.unmeasGbEsti[-1]
        y_unmeasGbEst = y_pred[..., si:ei+1]

        # Cost terms.
        cost_prederror = tf.math.reduce_mean(tf.square((y_true - y_pred)))
        cost_unmeasgberror = tf.square((y_unmeasGbEst - y_unmeasGbPred))
        cost_unmeasgberror = lam*tf.math.reduce_mean(cost_unmeasgberror)
        cost = cost_prederror + cost_unmeasgberror

        # Return.
        return cost

class ReacFullGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dx_g/dt  = f_g(x_g, u) + f_N(x_g, u)
    y = x_g
    """
    def __init__(self, r1Layers, r2Layers, r3Layers, estCLayers, 
                       Np, xuyscales, hyb_greybox_pars, **kwargs):
        super(ReacFullGbCell, self).__init__(**kwargs)
        
        # Attributes.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.r3Layers = r3Layers
        self.estCLayers = estCLayers
        self.Np = Np
        self.xuyscales = xuyscales
        self.hyb_greybox_pars = hyb_greybox_pars

        # Number of unmeasured grey-box states.
        self.numUnmeasGb = self.hyb_greybox_pars['Nx']
        self.numUnmeasGb += -self.hyb_greybox_pars['Ny']

    @property
    def state_size(self):
        """ Number of states in the model. """
        Nx = self.hyb_greybox_pars['Nx']
        Nx += self.Np*self.hyb_greybox_pars['Ny']
        Nx += self.Np*self.hyb_greybox_pars['Nu']
        # Return.
        return Nx
    
    @property
    def output_size(self):
        """ Number of outputs of the model. """
        Ny = self.hyb_greybox_pars['Nx'] + self.numUnmeasGb
        # Return.
        return Ny

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

        # Get the states before scaling. Compute the NN reaction rates.
        Ca, Cb, Cc = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        r1NN = fnnTf(Ca, self.r1Layers)
        r2NN = fnnTf(Cb, self.r2Layers)
        r3NN = fnnTf(Cc, self.r3Layers)

        # Get scaling factors.
        # Such that scalings based on only the measurements are used.
        ymean, ystd = self.xuyscales['yscale']
        Camean, Cbmean = ymean[0:1], ymean[1:2]
        Castd, Cbstd = ystd[0:1], ystd[1:2]
        umean, ustd = self.xuyscales['uscale']

        # Scale back to physical states and controls.
        Ca = Ca*Castd + Camean
        Cb = Cb*Castd + Cbmean
        Cc = Cc*Cbstd + Cbmean
        u = u*ustd + umean

        # Get the control input after scaling.
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = F*(Caf - Ca)/V - r1NN*Castd
        dCbbydt = -F*Cb/V + r1NN*Castd - 3*r2NN*Cbstd + r3NN*Cbstd
        dCcbydt = -F*Cc/V + r2NN*Cbstd - r3NN*Cbstd

        # Scaled derivate.
        xdot = tf.concat([dCabydt/Castd, dCbbydt/Cbstd, dCcbydt/Cbstd], axis=-1)

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx + Nz)
            Dimension of input: (None, Nu)
        """

        # Extract states.
        [xz] = states
        u = inputs

        # Extract the grey-box state, past measurements, and controls (z).
        (x, z) = tf.split(xz, [self.Nx, self.Np*(self.Ny + self.Nu)], axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu], 
                                  axis=-1)
        Ca, Cb, Cc = x[..., 0:1], x[..., 1:2], x[..., 2:3]

        # NN predictions for the third state based on z.
        CcNN = fnnTf(z, self.estCLayers)
        x = tf.concat((Ca, Cb, CcNN), axis=-1)

        # Sample time.
        Delta = self.hyb_greybox_pars['Delta']

        # Get k1, k2, k3, and k4.
        k1 = self._fxu(x, u)
        k2 = self._fxu(x + Delta*(k1/2), u)
        k3 = self._fxu(x + Delta*(k2/2), u)
        k4 = self._fxu(x + Delta*k3, u)
        
        # Get the state at the next time step.
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((ypseq[..., self.Ny:], x[..., :self.Ny], 
                           upseq[..., self.Nu:], u), axis=-1)
        xzplus = tf.concat((xplus, zplus), axis=-1)

        # Current output.
        y = tf.concat((Ca, Cb, Cc, CcNN), axis=-1)

        # Return output and states at the next time-step.
        return (y, xzplus)

class ReacFullGbModel(tf.keras.Model):
    """ Custom model for the Two reaction system. """
    
    def __init__(self, r1Dims, r2Dims, r3Dims, estCDims, 
                       Np, xuyscales, hyb_greybox_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Sizes.
        Nx = hyb_greybox_pars['Nx']
        Nu = hyb_greybox_pars['Nu']
        Ny = hyb_greybox_pars['Ny']
        Nz = Np*(Nu + Ny)
        
        # Input layers to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        x0 = tf.keras.Input(name='x0', shape=(Nx + Nz, ))

        # Dense layers for the NN.
        r1Layers = createDenseLayers(r1Dims)
        r2Layers = createDenseLayers(r2Dims)
        r3Layers = createDenseLayers(r3Dims)
        estCLayers = createDenseLayers(estCDims)

        # Get the reac cell object.
        reacCell = ReacFullGbCell(r1Layers, r2Layers, r3Layers, 
                                  estCLayers, Np, xuyscales, 
                                  hyb_greybox_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        xseq = reacLayer(inputs = useq, initial_state = [x0])

        # Construct model.
        super().__init__(inputs = [useq, x0], outputs = xseq)

class ReacPartialGbCell(tf.keras.layers.AbstractRNNCell):
    """
    TODO: Review This class.
    RNN Cell:
    dx_g/dt  = f_g(x_g, u) + (chosen functions).
    y = x_g
    r1 = NN1(Ca)
    r2 = NN2(Cb)
    r3 = NN3(z)
    """
    def __init__(self, r1Layers, r2Layers, r3Layers, Np, interpLayer,
                       xuyscales, hyb_greybox_pars, **kwargs):
        super(ReacPartialGbCell, self).__init__(**kwargs)

        # r1, r2, and r3 layers.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.r3Layers = r3Layers
        self.interpLayer = interpLayer

        # Number of past measurements and controls.
        assert Np > 0
        self.Np = Np

        # xuyscales and hybrid parameters.
        self.xuyscales = xuyscales
        self.hyb_greybox_pars = hyb_greybox_pars

    @property
    def state_size(self):
        return self.hyb_greybox_pars['Nx']
    
    @property
    def output_size(self):
        return self.hyb_greybox_pars['Ny']

    def _fyzu(self, y, z, u):
        """ dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2 + f_N(z)
        """
        
        # Extract the parameters.
        F = self.hyb_greybox_pars['ps'].squeeze()
        V = self.hyb_greybox_pars['V']

        # Get the states (before scaling to physical variables).
        Ca, Cb = y[..., 0:1], y[..., 1:2]

        # Get the output of the neural network.
        r1NN = fnnTf(Ca, self.r1Layers)
        r2NN = fnnTf(Cb, self.r2Layers)
        r3NN = fnnTf(z, self.r3Layers)

        # Get scaling factors.
        ymean, ystd = self.xuyscales['yscale']
        Castd, Cbstd = ystd[0:1], ystd[1:2]
        umean, ustd = self.xuyscales['uscale']

        # Scale back to physical states and control inputs.
        y = y*ystd + ymean
        u = u*ustd + umean

        # Get the state and control (after scaling to physical variables).
        Ca, Cb = y[..., 0:1], y[..., 1:2]
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = F*(Caf - Ca)/V - r1NN*Castd
        dCbbydt = -F*Cb/V + r1NN*Castd - 3*r2NN*Cbstd + r3NN*Cbstd

        # Scaled derivate.
        xdot = tf.concat([dCabydt, dCbbydt], axis=-1)/ystd

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ny + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
        """

        # Extract states/inputs.
        [yz] = states
        u = inputs

        # Extract y, ypast, and upast.
        (y, z) = tf.split(yz, [self.Ny, self.Np*(self.Ny + self.Nu)], 
                          axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
                                  axis=-1)

        # Sample time.
        Delta = self.hyb_greybox_pars['Delta']

        # Get k1.
        k1 = self._fyzu(y, z, u)
        
        # Get k2.
        ypseqInterp = self.interpLayer(tf.concat((ypseq, y), axis=-1))
        z = tf.concat((ypseqInterp, upseq), axis=-1)
        k2 = self._fyzu(y + Delta*(k1/2), z, u)

        # Get k3.
        k3 = self._fyzu(y + Delta*(k2/2), z, u)

        # Get k4.
        ypseqInterp = tf.concat((ypseq[..., self.Ny:], y), axis=-1)
        z = tf.concat((ypseqInterp, upseq), axis=-1)
        k4 = self._fyzu(y + Delta*k3, z, u)
        
        # Get the yzplus at the next time step.
        yplus = y + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((ypseq[..., self.Ny:], y, upseq[..., self.Nu:], u))
        yzplus = tf.concat((yplus, zplus), axis=-1)

        # Return current output and states at the next time point.
        return (y, yzplus)

class ReacPartialGbModel(tf.keras.Model):
    """ 
    TODO: Review this class.
    Custom model for the Two reaction system. """
    
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

        # Get the reac cell object.
        reacCell = ReacHybridCell(fNLayers, xuyscales, hyb_greybox_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        xseq = reacLayer(inputs = useq, initial_state = [x0])

        # Construct model.
        super().__init__(inputs = [useq, x0], outputs = xseq)

def create_fullgb_model(*, r1Dims, r2Dims, r3Dims, estCDims, Np,  
                           xuyscales, hyb_greybox_pars, 
                           lam, yi, unmeasGbPredi, unmeasGbEsti):
    """ Create and compile the two reaction model for training. """

    # Create a model.
    model = ReacFullGbModel(r1Dims, r2Dims, r3Dims, estCDims, 
                            Np, xuyscales, hyb_greybox_pars)

    # Create a loss.
    loss = FullGbLoss(lam, yi, unmeasGbPredi, unmeasGbEsti)

    # Compile the model.
    model.compile(optimizer='adam', loss=loss)

    # Return.
    return model

def create_partialgb_model(*, r1Layers, r2Layers, r3Layers, estCLayers, 
                    xuyscales, hyb_greybox_pars):
    """ 
    TODO: Review this function.
    Create and compile the two reaction model for training. """

    # Create a model.
    model = ReacPartialGbCell(r1Layers, r2Layers, r3Layers, Np, interpLayer, 
                                 xuyscales, hyb_greybox_pars)

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