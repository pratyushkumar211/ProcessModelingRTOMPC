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
    Nt: Number of time steps in the signal. 
    """
    def __init__(self, Nvar, Nt, trainable=False, name=None):
        super(InterpolationLayer, self).__init__(trainable, name)
        self.Nvar = Nvar
        self.Nt = Nt

    def call(self, yseq):
        """ The main call function of the interpolation layer.
            yseq is of dimension: (None, Nt*Nvar)
            Return y of dimension: (None, (Nt-1)*Nvar)
        """
        yseq_interp = []
        for t in range(self.Nt-1):
            yseq_interp += [0.5*(yseq[..., t*self.Nvar:(t+1)*self.Nvar] + 
                                 yseq[..., (t+1)*self.Nvar:(t+2)*self.Nvar])]
        # Return.
        return tf.concat(yseq_interp, axis=-1)

def getInterpolatedVals(yseq, Nvar, Nt):
    """ y is of dimension: (None, Nt*Nvar)
        Return y of dimension: (None, (Nt-1)*Nvar). """
    yseq_interp = []
    for t in range(Nt-1):
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
    r2 = NN2(Cb, z)
    """
    def __init__(self, r1Layers, r2Layers, Np, interpLayer,
                       xuyscales, hyb_partialgb_pars, **kwargs):
        super(ReacPartialGbCell, self).__init__(**kwargs)

        # r1, r2, and r3 layers.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.interpLayer = interpLayer

        # Sizes.
        assert Np > 0
        self.Np = Np
        self.Ng = hyb_partialgb_pars['Ng']
        self.Nu = hyb_partialgb_pars['Nu']
        self.Ny = hyb_partialgb_pars['Ny']
        self.Nx = self.Ng + self.Np*(self.Nu + self.Ny)

        # xuyscales and hybrid parameters.
        self.xuyscales = xuyscales
        self.hyb_partialgb_pars = hyb_partialgb_pars

    @property
    def state_size(self):
        """ State size of the model. """
        return self.Nx
    
    @property
    def output_size(self):
        """ Number of outputs of the model. """
        return self.Ny

    def _ode(self, x, z, u):
        """ dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2
        """
        
        # Extract the parameters.
        F = self.hyb_partialgb_pars['ps'].squeeze()
        V = self.hyb_partialgb_pars['V']

        # Get scaling factors.
        ymean, ystd = self.xuyscales['yscale']
        Castd, Cbstd = ystd[0:1], ystd[1:2]
        umean, ustd = self.xuyscales['uscale']

        # Get the output of the neural network.
        Ca, Cb = x[..., 0:1], x[..., 1:2]
        r1 = fnnTf(Ca, self.r1Layers)*Castd
        r2Input = tf.concat((Cb, z), axis=-1)
        r2 = fnnTf(r2Input, self.r2Layers)*Cbstd

        # Scale back to physical states and control inputs.
        x = x*ystd + ymean
        u = u*ustd + umean

        # Get the state and control (after scaling to physical variables).
        Ca, Cb = x[..., 0:1], x[..., 1:2]
        Caf = u[..., 0:1]
        
        # Write the ODEs.
        dCabydt = F*(Caf - Ca)/V - r1
        dCbbydt = -F*Cb/V + r1 - 3*r2

        # Scaled derivative.
        xdot = tf.concat([dCabydt, dCbbydt], axis=-1)/ystd

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ng + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
        """

        # Extract states and inputs.
        [xz] = states
        u = inputs

        # Extract x, xpast, and upast.
        (x, z) = tf.split(xz, [self.Ng, self.Np*(self.Ng + self.Nu)], 
                          axis=-1)
        (xpseq, upseq) = tf.split(z, [self.Np*self.Ng, self.Np*self.Nu],
                                  axis=-1)

        # Sample time.
        Delta = self.hyb_partialgb_pars['Delta']
        
        # Get k1.
        k1 = self._ode(x, z, u)
        
        # Get k2.
        xpseq_k2k3 = self.interpLayer(tf.concat((xpseq, x), axis=-1))
        z = tf.concat((xpseq_k2k3, upseq), axis=-1)
        k2 = self._ode(x + Delta*(k1/2), z, u)

        # Get k3.
        k3 = self._ode(x + Delta*(k2/2), z, u)

        # Get k4.
        xpseq_k4 = tf.concat((xpseq[..., self.Ng:], x), axis=-1)
        z = tf.concat((xpseq_k4, upseq), axis=-1)
        k4 = self._ode(x + Delta*k3, z, u)
        
        # Get the xzplus at the next time step.
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((xpseq[..., self.Ng:], x, 
                           upseq[..., self.Nu:], u), axis=-1)
        xzplus = tf.concat((xplus, zplus), axis=-1)

        # Return current output and states at the next time point.
        return (x, xzplus)

class ReacPartialGbModel(tf.keras.Model):
    """
    Partial model for the reaction system.
    """
    def __init__(self, r1Dims, r2Dims, Np, xuyscales, hyb_partialgb_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Sizes.
        Ng, Nu = hyb_partialgb_pars['Ng'], hyb_partialgb_pars['Nu']
        Nz = Np*(Ng + Nu)

        # Input layers to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        xz0 = tf.keras.Input(name='xz0', shape=(Ng + Nz, ))

        # Layers for the reactions.
        r1Layers = createDenseLayers(r1Dims)
        r2Layers = createDenseLayers(r2Dims)

        # Interpolation layer for RK4 predictions.
        interpLayer = InterpolationLayer(Ng, Np + 1)

        # Get the reac cell object.
        reacCell = ReacPartialGbCell(r1Layers, r2Layers, Np,
                                     interpLayer, xuyscales, hyb_partialgb_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        yseq = reacLayer(inputs = useq, initial_state = [xz0])

        # Construct model.
        super().__init__(inputs = [useq, xz0], outputs = yseq)

        # Store the layers (to extract weights for use in numpy).
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers

def create_model(*, r1Dims, r2Dims, Np, xuyscales, hyb_partialgb_pars):
    """ Create and compile the two reaction model for training. """

    # Create a model.
    model = ReacPartialGbModel(r1Dims, r2Dims, Np, xuyscales,
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
    model.fit(x = [train_data['useq'], train_data['yz0']], 
              y = train_data['yseq'], 
              epochs = epochs, batch_size = batch_size,
        validation_data = ([trainval_data['useq'], trainval_data['yz0']], 
                            trainval_data['yseq']),
            callbacks = [checkpoint_callback])

def get_val_predictions(*, model, val_data, xuyscales, 
                           unmeasXIndices, ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    model_predictions = model.predict(x=[val_data['useq'], val_data['yz0']])

    # Scale.
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']

    # Get validation predictions.
    uval = val_data['useq'].squeeze(axis=0)*ustd + umean
    ypred = model_predictions.squeeze(axis=0)*ystd + ymean
    xpred = np.insert(ypred, unmeasXIndices, np.nan, axis=1)
    Nt = uval.shape[0]
    tval = np.arange(0, Nt*Delta, Delta)
    pval = np.tile(np.nan, (Nt, 1))

    # Collect the predictions in a simdata format.
    valPredData = SimData(t=tval, x=xpred, u=uval, y=ypred, p=pval)

    # Return.
    return valPredData

def get_hybrid_pars(*, train, hyb_partialgb_pars, plant_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['r1Weights'] = train['r1Weights']
    parameters['r2Weights'] = train['r2Weights']
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    parameters['Ny'] = hyb_partialgb_pars['Ny']
    parameters['Nu'] = hyb_partialgb_pars['Nu']
    parameters['Np'] = train['Np']
    parameters['Nx'] = hyb_partialgb_pars['Ng']
    parameters['Nx'] += parameters['Np']*(parameters['Ny'] + parameters['Nu'])

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
    """ Partial grey-box ODE function. 
        x, z, and, u are scaled quantities.
    """

    # Extract disturbance.
    F = p.squeeze()

    # Parameters.
    V = parameters['V']
    r1Weights = parameters['r1Weights']
    r2Weights = parameters['r2Weights']
    
    # Get the scales.
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    Castd, Cbstd = ystd[0:1], ystd[1:2]
    umean, ustd = xuyscales['uscale']
    
    # Get NN reaction rates.
    Ca, Cb = x[0:1], x[1:2]
    r1 = fnn(Ca, r1Weights)*Castd
    r2Input = np.concatenate((Cb, z))
    r2 = fnn(r2Input, r2Weights)*Cbstd

    # Scale states back to their physical values.
    x = x*ystd + ymean
    u = u*ustd + umean
    Ca, Cb = x[0:1], x[1:2]
    Caf = u[0:1]

    # Write the ODEs.
    dCabydt = F*(Caf-Ca)/V - r1
    dCbbydt = -F*Cb/V + r1 - 3*r2

    # xdot.
    xdot = mpc.vcat([dCabydt, dCbbydt])/ystd

    # Return.
    return xdot

def hybrid_fxup(xz, u, p, parameters):
    """ Hybrid model. """

    # Sizes.
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    Np = parameters['Np']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']
    xzmean = np.concatenate((np.tile(ymean, (Np + 1, )), 
                             np.tile(umean, (Np, ))))
    xzstd = np.concatenate((np.tile(ystd, (Np + 1, )), 
                            np.tile(ustd, (Np, ))))
    xz = (xz - xzmean)/xzstd
    u = (u - umean)/ustd

    # x, z, xpseq and upseq.
    x, z = xz[:Ny], xz[Ny:]
    xpseq, upseq = z[:Ny*Np], z[Ny*Np:]

    # Get NN weights.
    Delta = parameters['Delta']

    # Get k1.
    k1 = fxup(x, z, u, p, parameters)

    # Get k2.
    xpseq_k2k3 = getInterpolatedVals(np.concatenate((xpseq, x)), Ny, Np)
    z = np.concatenate((xpseq_k2k3, upseq))
    k2 = fxup(x + Delta*(k1/2), z, u, p, parameters)

    # Get k3.
    k3 = fxup(x + Delta*(k2/2), z, u, p, parameters)

    # Get k4.
    xpseq_k4 = np.concatenate((xpseq[Ny:], x))
    z = np.concatenate((xpseq_k4, upseq))
    k4 = fxup(x + Delta*k3, z, u, p, parameters)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Get zplus and state at the next time step.
    zplus = np.concatenate((xpseq[Ny:], x, upseq[Nu:], u))
    xzplus = np.concatenate((xplus, zplus))

    # Scale back to physical quantity.
    xzplus = xzplus*xzstd + xzmean

    # Return.
    return xzplus

def hybrid_hx(xz, parameters):
    """ Measurement function. """
    Ny = parameters['Ny']
    y = xz[:Ny]
    # Return measurement.
    return y