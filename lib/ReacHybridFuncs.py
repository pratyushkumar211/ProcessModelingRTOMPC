# [depends] BlackBoxFuncs.py hybridId.py
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTf, fnn
from hybridId import SimData

def createDenseLayers(nnDims):
    """ Create dense layers based on the feed-forward NN layer dimensions.
        nnDims: List that contains the dimensions of the feed forward NN.
    """
    nnLayers = []
    for dim in nnDims[1:-1]:
        nnLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
    nnLayers += [tf.keras.layers.Dense(nnDims[-1])]
    # Return.
    return nnLayers

# class InterpolationLayer(tf.keras.layers.Layer):
#     """
#     The layer to perform interpolation for RK4 predictions.
#     Nvar: Number of variables.
#     Np + 1: Number of variables.
#     """
#     def __init__(self, Nvar, Np, trainable=False, name=None):
#         super(InterpolationLayer, self).__init__(trainable, name)
#         self.Nvar = Nvar
#         self.Np = Np

#     def call(self, yseq):
#         """ The main call function of the interpolation layer.
#             yseq is of dimension: (None, (Np+1)*Nvar)
#             Return y of dimension: (None, Np*Nvar)
#         """
#         yseq_interp = []
#         for t in range(self.Np):
#             yseq_interp += [0.5*(yseq[..., t*self.Nvar:(t+1)*self.Nvar] + 
#                                  yseq[..., (t+1)*self.Nvar:(t+2)*self.Nvar])]
#         # Return.
#         return tf.concat(yseq_interp, axis=-1)

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

        # Lambda, yi, indices relevant to the unmeasured grey-box states.
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
        # by the forward predictor model.
        si, ei = self.unmeasGbPredi[0], self.unmeasGbPredi[-1]
        y_unmeasGbPred = y_pred[..., si:ei+1]

        # Prediction of unmeasured grey-box states 
        # by the neural network.
        si, ei = self.unmeasGbEsti[0], self.unmeasGbEsti[-1]
        y_unmeasGbEst = y_pred[..., si:ei+1]

        # Cost terms.
        cost_prederror = tf.math.reduce_mean(tf.square((y_true - y_pred)))
        cost_unmeasgberror = tf.square((y_unmeasGbEst - y_unmeasGbPred))
        cost_unmeasgberror = self.lam*tf.math.reduce_mean(cost_unmeasgberror)
        cost = cost_prederror + cost_unmeasgberror

        # Return.
        return cost

class ReacFullGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    x_g^+ = f_g(x_g, u; NN(x_g))
    z = [y(k-N_p); y(k-N_p+1); ..., y(k-1); 
         u(k-N_p); u(k-N_p+1); ... u(k-1)];
    """
    def __init__(self, r1Layers, r2Layers, r3Layers, estCLayers, 
                       Np, xuyscales, hyb_fullgb_pars, **kwargs):
        super(ReacFullGbCell, self).__init__(**kwargs)
        
        # Attributes.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.r3Layers = r3Layers
        self.estCLayers = estCLayers
        self.xuyscales = xuyscales
        self.hyb_fullgb_pars = hyb_fullgb_pars

        # Sizes.
        self.Nx = hyb_fullgb_pars['Nx']
        self.Nu = hyb_fullgb_pars['Nu']
        self.Ny = hyb_fullgb_pars['Ny']
        self.Np = Np
        self.numUnmeasGb = self.hyb_fullgb_pars['Nx']
        self.numUnmeasGb += -self.hyb_fullgb_pars['Ny']

    @property
    def state_size(self):
        """ Number of states in the model. """
        # Return.
        return self.Nx + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        """ Number of outputs of the model. """
        if self.estCLayers is not None:
            return self.Ny + 2*self.numUnmeasGb
        else:
            return self.Ny

    def _fxu(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. 
            
            dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2 + r3
            dCc/dt = -F*Cc/V + r2 - r3
        """
        
        # Extract the parameters (nominal value of unmeasured disturbance).
        F = self.hyb_fullgb_pars['ps'].squeeze()
        V = self.hyb_fullgb_pars['V']

        # Get the states before scaling.
        Ca, Cb, Cc = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        Caf = u[..., 0:1]

        # Get scaling factors.
        ymean, ystd = self.xuyscales['yscale']
        Camean, Cbmean = ymean[0:1], ymean[1:2]
        Castd, Cbstd = ystd[0:1], ystd[1:2]
        Cafmean, Cafstd = self.xuyscales['uscale']

        # Compute NN reaction rates.
        r1 = fnnTf(Ca, self.r1Layers)*Castd
        r2 = fnnTf(Cb, self.r2Layers)*Cbstd
        r3 = fnnTf(Cc, self.r3Layers)*Cbstd

        # Scale back to physical states and controls.
        Ca = Ca*Castd + Camean
        Cb = Cb*Cbstd + Cbmean
        Cc = Cc*Cbstd + Cbmean
        Caf = Caf*Cafstd + Cafmean
        
        # ODEs.
        dCabydt = F*(Caf - Ca)/V - r1
        dCbbydt = -F*Cb/V + r1 - 3*r2 + r3
        dCcbydt = -F*Cc/V + r2 - r3

        # Get scaled derivate.
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
        
        # Extract the grey-box state (x), and past measurements/controls (z).
        (x, z) = tf.split(xz, [self.Nx, self.Np*(self.Ny + self.Nu)], axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu], 
                                  axis=-1)

        # Sample time.
        Delta = self.hyb_fullgb_pars['Delta']

        # Get k1, k2, k3, and k4.
        k1 = self._fxu(x, u)
        k2 = self._fxu(x + Delta*(k1/2), u)
        k3 = self._fxu(x + Delta*(k2/2), u)
        k4 = self._fxu(x + Delta*k3, u)
        
        # Get the state at the next time step.
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        if self.Np > 0:
            zplus = tf.concat((ypseq[..., self.Ny:], x[..., :self.Ny], 
                               upseq[..., self.Nu:], u), axis=-1)
        else:
            zplus = z
        xzplus = tf.concat((xplus, zplus), axis=-1)
        
        # Get the current output.
        if self.estCLayers is not None:
            CcNN = fnnTf(z, self.estCLayers)
            y = tf.concat((x, CcNN), axis=-1)
        else:
            y = x[..., :self.Ny]

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
        if estCDims is not None:
            estCLayers = createDenseLayers(estCDims)
        else:
            estCLayers = None

        # Get the reac cell object.
        reacCell = ReacFullGbCell(r1Layers, r2Layers, r3Layers,
                                  estCLayers, Np, xuyscales,
                                  hyb_greybox_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        yseq = reacLayer(inputs = useq, initial_state = [x0])

        # Construct model.
        super().__init__(inputs = [useq, x0], outputs = yseq)

        # Store the layers (to extract weights for use in numpy).
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.r3Layers = r3Layers
        self.estCLayers = estCLayers

# class ReacPartialGbCell(tf.keras.layers.AbstractRNNCell):
#     """
#     TODO: Review This class.
#     RNN Cell:
#     dx_g/dt  = f_g(x_g, u) + (chosen functions).
#     y = x_g
#     r1 = NN1(Ca)
#     r2 = NN2(Cb)
#     r3 = NN3(z)
#     """
#     def __init__(self, r1Layers, r2Layers, r3Layers, Np, interpLayer,
#                        xuyscales, hyb_greybox_pars, **kwargs):
#         super(ReacPartialGbCell, self).__init__(**kwargs)

#         # r1, r2, and r3 layers.
#         self.r1Layers = r1Layers
#         self.r2Layers = r2Layers
#         self.r3Layers = r3Layers
#         self.interpLayer = interpLayer

#         # Number of past measurements and controls.
#         assert Np > 0
#         self.Np = Np

#         # xuyscales and hybrid parameters.
#         self.xuyscales = xuyscales
#         self.hyb_greybox_pars = hyb_greybox_pars

#     @property
#     def state_size(self):
#         return self.hyb_greybox_pars['Nx']
    
#     @property
#     def output_size(self):
#         return self.hyb_greybox_pars['Ny']

#     def _fyzu(self, y, z, u):
#         """ dCa/dt = F*(Caf - Ca)/V - r1
#             dCb/dt = -F*Cb/V + r1 - 3*r2 + f_N(z)
#         """
        
#         # Extract the parameters.
#         F = self.hyb_greybox_pars['ps'].squeeze()
#         V = self.hyb_greybox_pars['V']

#         # Get the states (before scaling to physical variables).
#         Ca, Cb = y[..., 0:1], y[..., 1:2]

#         # Get the output of the neural network.
#         r1NN = fnnTf(Ca, self.r1Layers)
#         r2NN = fnnTf(Cb, self.r2Layers)
#         r3NN = fnnTf(z, self.r3Layers)

#         # Get scaling factors.
#         ymean, ystd = self.xuyscales['yscale']
#         Castd, Cbstd = ystd[0:1], ystd[1:2]
#         umean, ustd = self.xuyscales['uscale']

#         # Scale back to physical states and control inputs.
#         y = y*ystd + ymean
#         u = u*ustd + umean

#         # Get the state and control (after scaling to physical variables).
#         Ca, Cb = y[..., 0:1], y[..., 1:2]
#         Caf = u[..., 0:1]
        
#         # Write the ODEs.
#         dCabydt = F*(Caf - Ca)/V - r1NN*Castd
#         dCbbydt = -F*Cb/V + r1NN*Castd - 3*r2NN*Cbstd + r3NN*Cbstd

#         # Scaled derivate.
#         xdot = tf.concat([dCabydt, dCbbydt], axis=-1)/ystd

#         # Return the derivative.
#         return xdot

#     def call(self, inputs, states):
#         """ Call function of the hybrid RNN cell.
#             Dimension of states: (None, Ny + Np*(Ny + Nu))
#             Dimension of input: (None, Nu)
#         """

#         # Extract states/inputs.
#         [yz] = states
#         u = inputs

#         # Extract y, ypast, and upast.
#         (y, z) = tf.split(yz, [self.Ny, self.Np*(self.Ny + self.Nu)], 
#                           axis=-1)
#         (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
#                                   axis=-1)

#         # Sample time.
#         Delta = self.hyb_greybox_pars['Delta']

#         # Get k1.
#         k1 = self._fyzu(y, z, u)
        
#         # Get k2.
#         ypseqInterp = self.interpLayer(tf.concat((ypseq, y), axis=-1))
#         z = tf.concat((ypseqInterp, upseq), axis=-1)
#         k2 = self._fyzu(y + Delta*(k1/2), z, u)

#         # Get k3.
#         k3 = self._fyzu(y + Delta*(k2/2), z, u)

#         # Get k4.
#         ypseqInterp = tf.concat((ypseq[..., self.Ny:], y), axis=-1)
#         z = tf.concat((ypseqInterp, upseq), axis=-1)
#         k4 = self._fyzu(y + Delta*k3, z, u)
        
#         # Get the yzplus at the next time step.
#         yplus = y + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
#         zplus = tf.concat((ypseq[..., self.Ny:], y, upseq[..., self.Nu:], u))
#         yzplus = tf.concat((yplus, zplus), axis=-1)

#         # Return current output and states at the next time point.
#         return (y, yzplus)

# class ReacPartialGbModel(tf.keras.Model):
#     """ 
#     TODO: Review this class.
#     Custom model for the Two reaction system. """
    
#     def __init__(self, fNDims, xuyscales, hyb_greybox_pars):
#         """ Create the dense layers for the NN, and 
#             construct the overall model. """

#         # Sizes.
#         Nx, Nu = hyb_greybox_pars['Nx'], hyb_greybox_pars['Nu']
        
#         # Input layers to the model.
#         useq = tf.keras.Input(name='u', shape=(None, Nu))
#         x0 = tf.keras.Input(name='x0', shape=(Nx, ))

#         # Dense layers for the NN.
#         fNLayers = []
#         for dim in fNDims[1:-1]:
#             fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
#         fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

#         # Get the reac cell object.
#         reacCell = ReacHybridCell(fNLayers, xuyscales, hyb_greybox_pars)

#         # Construct the RNN layer and get the predicted xseq.
#         reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
#         xseq = reacLayer(inputs = useq, initial_state = [x0])

#         # Construct model.
#         super().__init__(inputs = [useq, x0], outputs = xseq)

def create_fullgb_model(*, r1Dims, r2Dims, r3Dims, estCDims, Np,
                           xuyscales, hyb_fullgb_pars,
                           lamGbError, yi, unmeasGbPredi,
                           unmeasGbEsti):
    """ Create and compile the two reaction model for training. """

    # Create a model.
    model = ReacFullGbModel(r1Dims, r2Dims, r3Dims, estCDims,
                            Np, xuyscales, hyb_fullgb_pars)

    # Create a loss.
    loss = FullGbLoss(lamGbError, yi, unmeasGbPredi, unmeasGbEsti)

    # Compile the model.
    model.compile(optimizer='adam', loss=loss)

    # Return.
    return model

# def create_partialgb_model(*, r1Layers, r2Layers, r3Layers, estCLayers, 
#                               xuyscales, hyb_greybox_pars):
#     """ 
#     TODO: Review this function.
#     Create and compile the two reaction model for training. """

#     # Create a model.
#     model = ReacPartialGbCell(r1Layers, r2Layers, r3Layers, Np, interpLayer, 
#                               xuyscales, hyb_greybox_pars)

#     # Compile the model.
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     # Return.
#     return model

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

def get_fullgbval_predictions(*, model, val_data, xuyscales, 
                                 xinsert_indices, ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['inputs'], val_data['x0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']

    # Add extra terms to scaling factor.
    ymean = np.concatenate((ymean, ymean[-1:], ymean[-1:]))
    ystd = np.concatenate((ystd, ystd[-1:], ystd[-1:]))

    # Validation predictions.
    ypredictions = model_predictions.squeeze(axis=0)*ystd + ymean
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean

    # Get xpredictions.
    xpredictions = np.insert(ypredictions, xinsert_indices, np.nan, axis=1)

    # Collect data in a Simdata format.
    Nt = uval.shape[0]
    tval = np.arange(0, Nt*Delta, Delta)
    pval = np.zeros((Nt, ))
    val_predictions = SimData(t=tval, x=xpredictions, 
                              u=uval, y=ypredictions, p=pval)

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data['x0']], 
                                y = val_data['outputs'])

    # Return.
    return (val_predictions, val_metric)

def get_weights(layers):
    """ Function to get the weights from a list 
        of layers. """
    Weights = []
    for layer in layers:
        Weights += layer.get_weights()
    # Return weights.
    return Weights

def get_fullgb_pars(*, train, hyb_greybox_pars):
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

def fxup(x, u, p, parameters):
    """ Partial grey-box ODE function. """

    # Extract the plant states into meaningful names.
    Ca, Cb, Cc = x[0:1], x[1:2], x[2:3]
    Caf = u[0:1]
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
    
    # Get NN reaction rates.
    xmean = np.concatenate((ymean, ymean[1:]))
    xstd = np.concatenate((ystd, ystd[1:]))
    x = (x - xmean)/xstd
    Ca, Cb, Cc = x[0:1], x[1:2], x[2:3]
    r1 = fnn(Ca, r1Weights)*Castd
    r2 = fnn(Cb, r2Weights)*Cbstd
    r3 = fnn(Cc, r3Weights)*Cbstd

    # Write the ODEs.
    dCabydt = F*(Caf-Ca)/V - r1
    dCbbydt = -F*Cb/V + r1 - 3*r2 + r3
    dCcbydt = -F*Cc/V + r2 - r3

    # Scale.
    xdot = mpc.vcat([dCabydt, dCbbydt, dCcbydt])

    # Return.
    return xdot

def hybrid_fxup(xz, u, p, parameters):
    """ Hybrid model. """

    # Split into states and past measurements/controls.
    Nx = parameters['Nx']
    x, z = xz[:Nx], xz[Nx:]

    # Get NN weights.
    Delta = parameters['Delta']

    # Get k1, k2, k3, and k4.
    k1 = fxup(x, u, p, parameters)
    k2 = fxup(x + Delta*(k1/2), u, p, parameters)
    k3 = fxup(x + Delta*(k2/2), u, p, parameters)
    k4 = fxup(x + Delta*k3, u, p, parameters)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Get zplus and state at the next time step.
    Np = parameters['Np']
    if Np > 0:
        zplus = np.concatenate((z[Ny:], x[:Ny], z[Ny*Np+Nu:], u))
    else:
        zplus = z
    xzplus = np.concatenate((xplus, zplus))

    # Return.
    return xzplus

def hybrid_hx(x, parameters):
    """ Measurement function. """
    Ny = parameters['Ny']
    y = x[:Ny]
    # Return measurement.
    return y