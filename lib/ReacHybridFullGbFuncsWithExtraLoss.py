# [depends] BlackBoxFuncs.py hybridId.py
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTf, fnn, createDenseLayers
from hybridId import SimData

class CustomFullGbLoss(tf.keras.losses.Loss):
    """ 
        Custom loss function for the full grey-box model. 
        Parameters: lam, yi_list, unmeasGbi_list, unmeasGbi_NN_list.

    """
    def __init__(self, lam, yi_list, unmeasGbi_list, unmeasGbi_NN_list):

        # Lambda, yi, indices relevant to the unmeasured grey-box states.
        self.lam = lam
        self.yi_list = yi_list
        self.unmeasGbi_list = unmeasGbi_list
        self.unmeasGbi_NN_list = unmeasGbi_NN_list

        # Initialize.
        super(CustomFullGbLoss, self).__init__()

    def call(self, y_true, y_pred):
        """ Call function for the custom MSE. """
        
        # Measurement predictions.
        ei = self.yi_list[-1]
        y_pred = y_pred[..., :ei+1]

        # Unmeasured grey-box state predictions.
        si, ei = self.unmeasGbi_list[0], self.unmeasGbi_list[-1]
        y_unmeasGb = y_pred[..., si:ei+1]

        # Unmeasured grey-box state predictions by the NN.
        si, ei = self.unmeasGbi_NN_list[0], self.unmeasGbi_NN_list[-1]
        y_unmeasGb_NN = y_pred[..., si:ei+1]

        # Cost terms.
        predError = tf.math.reduce_mean(tf.square((y_true - y_pred)))
        unmeasGbError = tf.square((y_unmeasGb - y_unmeasGb_NN))
        unmeasGbError = self.lam*tf.math.reduce_mean(unmeasGbError)
        cost = predError + unmeasGbError

        # Return.
        return cost

class ReacFullGbCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    x_g^+ = f_g(x_g, u; NN(x_g))
    """
    def __init__(self, r1Layers, r2Layers, estCLayers,
                       Np, xuyscales, hyb_fullgb_pars, **kwargs):
        super(ReacFullGbCell, self).__init__(**kwargs)
        
        # Attributes.
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.estCLayers = estCLayers
        self.xuyscales = xuyscales
        self.hyb_fullgb_pars = hyb_fullgb_pars

        # Sizes.
        self.Np = Np
        self.Nx = hyb_fullgb_pars['Nx']
        self.Nu = hyb_fullgb_pars['Nu']
        self.Ny = hyb_fullgb_pars['Ny']       

    @property
    def state_size(self):
        """ Number of states in the model. """
        # Return.
        return self.Nx + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        """ Number of outputs of the model.
            (one extra output for the extra predictions of the third state). """
        # Return.
        return self.Nx + 1

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
        [xz] = states
        u = inputs

        # Extract the states and z.
        (x, z) = tf.split(xz, [self.Nx, self.Np*(self.Ny + self.Nu)], 
                          axis=-1)
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
        zplus = tf.concat((ypseq[..., self.Ny:], x[..., :self.Ny], 
                           upseq[..., self.Nu:], u), axis=-1)
        xzplus = tf.concat((xplus, zplus), axis=-1)

        # Get extra prediction for the unmeasured state.
        CcNN = fnnTf(z, self.estCLayers)
        y = tf.concat((x, CcNN), axis=-1)

        # Return.
        return (y, xzplus)

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

        # Np > 0 and estCDims should be a list of estimator NN dimensions.
        assert Np > 0 and estCDims is not None
        Nz = Np*(Nu + Ny)
        
        # Input layers to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        xz0 = tf.keras.Input(name='xz0', shape=(Nx + Nz, ))
        
        # Dense layers for the NN.
        r1Layers = createDenseLayers(r1Dims)
        r2Layers = createDenseLayers(r2Dims)
        estCLayers = createDenseLayers(estCDims)

        # Initial state for forecasting.
        y0, _, z0 = tf.split(xz0, [Ny, Nx-Ny, Nz], axis=1)
        Cc0 = fnnTf(z0, estCLayers)
        xz0_NN = tf.concat((y0, Cc0, z0), axis=-1)

        # Get the reac cell object.
        reacCell = ReacFullGbCell(r1Layers, r2Layers, estCLayers,
                                  Np, xuyscales, hyb_greybox_pars)

        # Construct the RNN layer and get the predicted xseq.
        reacLayer = tf.keras.layers.RNN(reacCell, return_sequences = True)
        yseq_full = reacLayer(inputs = useq, initial_state = [xz0_NN])

        # Get yseq, xseq.
        yseq, _ = tf.split(yseq_full, [Ny, Nx+1-Ny], axis=-1)
        xseq, _ = tf.split(yseq_full, [Nx, 1], axis=-1)

        # Construct model.
        super().__init__(inputs = [useq, xz0], 
                         outputs = [yseq_full, yseq, xseq])

        # Store the layers (to extract weights for use in numpy).
        self.r1Layers = r1Layers
        self.r2Layers = r2Layers
        self.estCLayers = estCLayers

def create_model(*, r1Dims, r2Dims, estCDims, Np,
                    xuyscales, hyb_fullgb_pars, lam, yi_list,
                    unmeasGbi_list, unmeasGbi_NN_list):
    """ Create and compile the two reaction model for training. """

    # Create model.
    model = ReacFullGbModel(r1Dims, r2Dims, estCDims,
                            Np, xuyscales, hyb_fullgb_pars)

    # Create loss/compile model.
    loss = CustomFullGbLoss(lam, yi_list, unmeasGbi_list, unmeasGbi_NN_list)
    model.compile(optimizer='adam', loss=[loss, 'mean_squared_error', 
                                          'mean_squared_error'], 
                                    loss_weights=[1., 0., 0])

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
              y = [train_data['outputs'], 
                   train_data['outputs'], train_data['xseq']],
              epochs = epochs, batch_size = batch_size,
        validation_data = ([trainval_data['inputs'], trainval_data['xz0']], 
                           [trainval_data['outputs'], trainval_data['outputs'], 
                           trainval_data['xseq']]),
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
                                y = [val_data['outputs'], val_data['outputs'], 
                                     val_data['xseq']])

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