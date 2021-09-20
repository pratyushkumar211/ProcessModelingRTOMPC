# [depends] BlackBoxFuncs.py hybridId.py
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTf, fnn, createDenseLayers
from hybridId import SimData

class ReacFullGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    x_g^+ = f_g(x_g, u; NN(x_g))
    """
    def __init__(self, r1Layers, r2Layers,
                       xuyscales, hyb_fullgb_pars, **kwargs):
        super(ReacFullGbCell, self).__init__(**kwargs)
        
        # Attributes.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.xuyscales = xuyscales
        self.hyb_fullgb_pars = hyb_fullgb_pars

    @property
    def state_size(self):
        """ Number of states in the model. """
        # Return.
        return self.hyb_fullgb_pars['Nx']
    
    @property
    def output_size(self):
        """ Number of outputs of the model. """
        # Return.
        return self.hyb_fullgb_pars['Nx']

    def _fxu(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model.
            
            dCa/dt = F*(Caf - Ca)/V - r1
            dCb/dt = -F*Cb/V + r1 - 3*r2
            dCc/dt = -F*Cc/V + r2
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
        r2Input = tf.concat((Cb, Cc), axis=-1)
        r2 = fnnTf(r2Input, self.r2Layers)*Cbstd

        # Scale back to physical states and controls.
        Ca = Ca*Castd + Camean
        Cb = Cb*Cbstd + Cbmean
        Cc = Cc*Cbstd + Cbmean
        Caf = Caf*Cafstd + Cafmean
        
        # ODEs.
        dCabydt = F*(Caf - Ca)/V - r1
        dCbbydt = -F*Cb/V + r1 - 3*r2
        dCcbydt = -F*Cc/V + r2

        # Get scaled derivate.
        xdot = tf.concat([dCabydt/Castd, dCbbydt/Cbstd, dCcbydt/Cbstd], axis=-1)

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """
        
        # Extract states.
        [x] = states
        u = inputs
        
        # Sample time.
        Delta = self.hyb_fullgb_pars['Delta']

        # Get k1, k2, k3, and k4.
        k1 = self._fxu(x, u)
        k2 = self._fxu(x + Delta*(k1/2), u)
        k3 = self._fxu(x + Delta*(k2/2), u)
        k4 = self._fxu(x + Delta*k3, u)
        
        # Get the state at the next time step.
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Return output and states at the next time-step.
        return (x, xplus)

class ReacFullGbModel(tf.keras.Model):
    """ Custom model for the reaction system. """
    
    def __init__(self, r1Dims, r2Dims, estCDims, 
                       Np, xuyscales, hyb_greybox_pars):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Sizes.
        Nx = hyb_greybox_pars['Nx']
        Nu = hyb_greybox_pars['Nu']
        Ny = hyb_greybox_pars['Ny']

        # If Np == 0, estCDims should be None.
        # If Np > 0, estCDims should be a list of estimator NN dimensions.
        assert Np == 0 and estCDims is None or Np > 0 and estCDims is not None
        Nz = Np*(Nu + Ny)
        self.Np = Np

        # Input layers to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        xz0 = tf.keras.Input(name='xz0', shape=(Nx + Nz, ))
        
        # Dense layers for the NN.
        r1Layers = createDenseLayers(r1Dims)
        r2Layers = createDenseLayers(r2Dims)

        # Check if dimensions of a NN to estimate the initial 
        # condition of Cc is provided.
        if estCDims is not None:
            estCLayers = createDenseLayers(estCDims)
            y0, _, z0 = tf.split(xz0, [Ny, Nx-Ny, Nz], axis=1)
            Cc0 = fnnTf(z0, estCLayers)
            x0 = tf.concat((y0, Cc0), axis=-1)
        else:
            x0 = xz0
            estCLayers = None

        # Get the reac cell object.
        reacCell = ReacFullGbCell(r1Layers, r2Layers,
                                  xuyscales, hyb_greybox_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        xseq = reacLayer(inputs = useq, initial_state = [x0])

        # Just get the yseq.
        yseq, _ = tf.split(xseq, [Ny, Nx-Ny], axis=-1)

        # Construct model.
        super().__init__(inputs = [useq, xz0], outputs = [yseq, xseq])

        # Store the layers (to extract weights for use in numpy).
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.estCLayers = estCLayers

def create_model(*, r1Dims, r2Dims, estCDims, Np,
                    xuyscales, hyb_fullgb_pars):
    """ Create and compile the two reaction model for training. """

    # Create a model.
    model = ReacFullGbModel(r1Dims, r2Dims, estCDims,
                            Np, xuyscales, hyb_fullgb_pars)

    # Compile.
    model.compile(optimizer='adam', loss='mean_squared_error', 
                  loss_weights=[1., 0.])

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
              y = [train_data['outputs'], train_data['xseq']],
              epochs = epochs, batch_size = batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['xz0']], 
                            [trainval_data['outputs'], trainval_data['xseq']]),
            callbacks = [checkpoint_callback])

def get_val_predictions(*, model, val_data, xuyscales, ckpt_path, Delta):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    ypredictions, xpredictions = model.predict(x=[val_data['inputs'], 
                                                  val_data['xz0']])

    # Compute val metric.
    val_metric = model.evaluate(x = [val_data['inputs'], val_data['xz0']], 
                                y = [val_data['outputs'], val_data['xseq']])[0]

    # Scale.
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']

    # Validation input sequence.
    uval = val_data['inputs'].squeeze(axis=0)*ustd + umean
    Nt = uval.shape[0]

    # Get predictions. 
    ypredictions = ypredictions.squeeze(axis=0)*ystd + ymean
    xpredictions = xpredictions.squeeze(axis=0)*xstd + xmean
    Nt = uval.shape[0]
    tval = np.arange(0, Nt*Delta, Delta)
    pval = np.tile(np.nan, (Nt, 1))
    val_predictions = [SimData(t=tval, x=xpredictions, 
                              u=uval, y=ypredictions, p=pval)]

    # Get the predictions of C concentrations by the estimator NN model.
    if model.estCLayers is not None:

        # Get NN weights and create a list.
        Np = model.Np
        estC_predictions = [np.tile(np.nan, (1,)) for _ in range(Np)]
        estCWeights = get_weights(model.estCLayers)

        # Re-scale back the ypredictions and uval.
        ypredictions = (ypredictions - ymean)/ystd
        uval = (uval - umean)/ustd
        Ny, Nu = len(ymean), len(umean)

        # For loop over time to get predictions.
        for t in range(Np, Nt):

            ypseq = ypredictions[t-Np:t, :].reshape(Np*Ny, )
            upseq = uval[t-Np:t, :].reshape(Np*Nu, )
            z = np.concatenate((ypseq, upseq))
            estC = fnn(z, estCWeights)
            estC_predictions += [estC]

        # Get scaled predictions.
        Cbmean, Cbstd = ymean[-1:], ystd[-1:]
        estC = np.array(estC_predictions)*Cbstd + Cbmean
        uval = uval*ustd + umean

        # Collect predictions.
        Cxpredictions = np.concatenate((np.tile(np.nan, (Nt, Ny)), 
                                        estC_predictions), axis=1)
        Cypredictions = np.tile(np.nan, (Nt, Ny))
        val_Cpredictions = SimData(t=tval, x=Cxpredictions, u=uval, 
                                   y=Cypredictions, p=pval)
        val_predictions += [val_Cpredictions]

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

def get_hybrid_pars(*, train, hyb_fullgb_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['r1Weights'] = train['trained_r1Weights'][-1]
    parameters['r2Weights'] = train['trained_r2Weights'][-1]
    parameters['estCWeights'] = train['trained_estCWeights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    parameters['Nx'] = hyb_fullgb_pars['Nx']
    parameters['Ny'] = hyb_fullgb_pars['Ny']
    parameters['Nu'] = hyb_fullgb_pars['Nu']
    parameters['Np'] = train['Np']

    # Constraints.
    parameters['ulb'] = hyb_fullgb_pars['ulb']
    parameters['uub'] = hyb_fullgb_pars['uub']
    
    # Greybox model parameters.
    parameters['ps'] = hyb_fullgb_pars['ps']
    parameters['V'] = hyb_fullgb_pars['V']

    # Sample time.
    parameters['Delta'] = hyb_fullgb_pars['Delta']

    # Return.
    return parameters

def fxup(x, u, p, parameters):
    """ Partial grey-box ODE function. """

    # Extract the disturbance.
    F = p.squeeze()

    # Parameters.
    V = parameters['V']
    r1Weights = parameters['r1Weights']
    r2Weights = parameters['r2Weights']
    
    # Get the scales.
    xuyscales = parameters['xuyscales']
    xmean, xstd = xuyscales['xscale']
    ymean, ystd = xuyscales['yscale']
    Castd, Cbstd = ystd[0:1], ystd[1:2]
    umean, ustd = xuyscales['uscale']
    
    # Get NN reaction rates.
    x = (x - xmean)/xstd
    Ca, CbCc = x[0:1], x[1:3]
    r1 = fnn(Ca, r1Weights)*Castd
    r2 = fnn(CbCc, r2Weights)*Cbstd

    # Scale the states back to physical variables 
    # and extract control input.
    x = x*xstd + xmean
    Ca, Cb, Cc = x[0:1], x[1:2], x[2:3]
    Caf = u[0:1]

    # Write the ODEs.
    dCabydt = F*(Caf-Ca)/V - r1
    dCbbydt = -F*Cb/V + r1 - 3*r2
    dCcbydt = -F*Cc/V + r2

    # Scale.
    xdot = mpc.vcat([dCabydt, dCbbydt, dCcbydt])

    # Return.
    return xdot

def hybrid_fxup(x, u, p, parameters):
    """ Hybrid model. """

    # Split into states and past measurements/controls.
    Nx = parameters['Nx']

    # Get NN weights.
    Delta = parameters['Delta']

    # Get k1, k2, k3, and k4.
    k1 = fxup(x, u, p, parameters)
    k2 = fxup(x + Delta*(k1/2), u, p, parameters)
    k3 = fxup(x + Delta*(k2/2), u, p, parameters)
    k4 = fxup(x + Delta*k3, u, p, parameters)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Return.
    return xplus

def hybrid_hx(x, parameters):
    """ Measurement function. """
    Ny = parameters['Ny']
    y = x[:Ny]
    # Return measurement.
    return y