"""
Custom neural network layers for black-box modeling 
using input convex neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf

def scaledExp(x, expScale):
    """ Scaled exponential to use as activation function. """
    return tf.math.exp(expScale*x)

class InputConvexLayer(tf.keras.layers.Layer):
    """
    Input convex layer.
    z_{i+1} = g(W_i^(z)*z_i + W_i^y*y + b_i)
    W_0^(z) = 0, W_{1:k}^(z) >= 0
    """
    def __init__(self, zPlusDim, zDim, yDim, Wz=False, activation=False, 
                 expScale=1, **kwargs):
        super(InputConvexLayer, self).__init__(**kwargs)

        # Save activation function information.
        self.activation = activation
        self.expScale = expScale

        # Create Wz.
        if Wz:
            WzInit =  tf.random_normal_initializer()
            self.Wz = tf.Variable(initial_value = 
                                  WzInit(shape=(zDim, zPlusDim)),
                                  trainable=True, dtype='float32',
                                  constraint=tf.keras.constraints.NonNeg())
        else:
            self.Wz = None
        
        # Create Wy.
        WyInit =  tf.random_normal_initializer()
        self.Wy = tf.Variable(initial_value = WyInit(shape=(yDim, zPlusDim)),
                              trainable=True, dtype='float32')

        # Create bias.
        biasInit =  tf.random_normal_initializer()
        self.bias = tf.Variable(initial_value = biasInit(shape=(zPlusDim, )),
                                trainable=True, dtype='float32')
    
    def call(self, z, y):
        """ Call function of the input convex NN layer. """
        
        if self.Wz is None:
            zplus = tf.linalg.matmul(y, self.Wy) + self.bias
        else:
            zplus = tf.linalg.matmul(z, self.Wz) + tf.linalg.matmul(y, self.Wy)
            zplus = zplus + self.bias

        if self.activation:
            zplus = tf.math.exp(self.expScale*zplus)

        # Return output.
        return zplus

def iCNNTF(nnInput, nnLayers):
    """ Compute the output of the feedforward network. """
    nnOutput = nnInput
    for layer in nnLayers:
        nnOutput = layer(nnOutput, nnInput)
    return nnOutput

class InputConvexCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    z = [y_{k-N_p:k-1}', u_{k-N_p:k-1}']'
    x = [y', z']'
    y^+ = f_N(x, u)
    y  = [I, 0]x
    """
    def __init__(self, Np, Ny, Nu, fNLayers, **kwargs):
        super(InputConvexCell, self).__init__(**kwargs)
        self.Np = Np
        self.Ny, self.Nu = Ny, Nu
        self.fNLayers = fNLayers

    @property
    def state_size(self):
        return self.Ny + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny
    
    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ny + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
            Dimension of output: (None, Ny)
        """
        
        # Extract important variables.
        [yz] = states
        u = inputs
        Np, Ny, Nu = self.Np, self.Ny, self.Nu

        # Extract elements of the state.
        if Np > 0:
            (y, ypseq, upseq) = tf.split(yz, [Ny, Np*Ny, Np*Nu],
                                         axis=-1)
        else:
            y = yz

        # Get the current output/state and the next time step.
        nnInput = tf.concat((yz, u), axis=-1)
        yplus = iCNNTF(nnInput, self.fNLayers)

        if Np > 0:
            yzplus = tf.concat((yplus, ypseq[..., Ny:], y, upseq[..., Nu:], u),
                               axis=-1)
        else:
            yzplus = yplus

        # Return output and states at the next time-step.
        return (y, yzplus)

class InputConvexModel(tf.keras.Model):
    """ Input convex neural network model. """
    def __init__(self, Nu, fNDims, expScale, Np=0, Nd=0):
        
        # Get the input layers (from which convexity is required).
        modelInputList = []
        u = tf.keras.Input(name='u', shape=(Nu, ))
        modelInputList += [u]

        # Get the input layers (from which convexity is not required).
        # p contains cost parameters.
        if Np > 0:
            p = tf.keras.Input(name='p', shape=(Np, ))
            modelInputList += [p]

        # d contains disturbances.
        if Nd > 0:
            d = tf.keras.Input(name='d', shape=(Nd, ))
            modelInputList += [d]
        
        # Get Input Convex NN layers.
        assert len(fNDims) > 2
        fNLayers = [InputConvexLayer(fNDims[1], fNDims[0], Nx + Nu, 
                                     activation=True, expScale=expScale)]
        for (zDim, zPlusDim) in zip(fNDims[1:-2], fNDims[2:-1]):
            fNLayers += [InputConvexLayer(zPlusDim, zDim, Nx + Nu, Wz=True, 
                                          activation=True, expScale=expScale)]
        fNLayers += [InputConvexLayer(fNDims[-1], fNDims[-2], Nx + Nu, Wz=True)]

        # Build model.
        iCCell = InputConvexCell(Np, Ny, Nu, fNLayers)

        # Construct the RNN layer and the computation graph.
        iCRNNLayer = tf.keras.layers.RNN(iCCell, return_sequences=True)
        yseq = iCRNNLayer(inputs=useq, initial_state=[yz0])

        # Construct model.
        super().__init__(inputs=modelInputList, outputs=yseq)

def create_iCNNmodel(*, Np, Ny, Nu, fNDims, expScale):
    """ Create/compile the two reaction model for training. """
    model = InputConvexModel(Np, Ny, Nu, fNDims, expScale)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def iCNN(nnInput, zWeights, yWeights, bias, expScale):
    """ Compute the NN output. """

    # Check input dimensions. 
    if nnInput.ndim == 1:
        numOutDim = 1
        nnInput = nnInput[:, np.newaxis]
    nnOutput = nnInput

    # Out of First layer.
    Wy, b = yWeights[0], bias[0]
    nnOutput = Wy.T @ nnOutput + b[:, np.newaxis]
    nnOutput = np.exp(expScale*nnOutput)

    # Loop over layers.
    for Wz, Wy, b in zip(zWeights[:-1], yWeights[1:-1], bias[1:-1]):
        nnOutput = Wz.T @ nnOutput + Wy.T @ nnInput + b[:, np.newaxis]
        nnOutput = np.exp(expScale*nnOutput)

    # Last layer.
    (Wzf, Wyf, bf) = zWeights[-1], yWeights[-1], bias[-1]
    nnOutput = Wzf.T @ nnOutput + Wyf.T @ nnInput + bf[:, np.newaxis]

    # Return output in same number of dimensions.
    if numOutDim == 1:
        nnOutput = nnOutput[:, 0]

    # Return.
    return nnOutput

def get_iCNN_pars(*, train, plant_pars):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    parameters = {}
    parameters['Np'] = train['Np']
    parameters['xuyscales'] = train['xuyscales']

    # Get weights.
    numLayers = len(train['fNDims']) - 1
    trained_weights = train['trained_weights'][-1]
    parameters['yWeights'] = trained_weights[slice(0, 3*numLayers, 3)]
    parameters['bias'] = trained_weights[slice(1, 3*numLayers, 3)]
    parameters['zWeights'] = trained_weights[slice(2, 3*numLayers, 3)]

    # Sizes.
    Ny, Nu = plant_pars['Ny'], plant_pars['Nu']
    parameters['Ny'], parameters['Nu'] = Ny, Nu
    parameters['Nx'] = Ny + parameters['Np']*(Ny + Nu)

    # Constraints.
    parameters['ulb'] = plant_pars['ulb']
    parameters['uub'] = plant_pars['uub']

    # Scaling for activation function.
    parameters['expScale'] = train['expScale']
    
    # Return.
    return parameters

def iCNN_fxu(yz, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        yz^+ = f_z(yz, u) """

    # Extract parameters.
    Np, Ny, Nu = parameters['Np'], parameters['Ny'], parameters['Nu']

    # Get NN weights.
    yWeights = parameters['yWeights']
    zWeights = parameters['zWeights']
    bias = parameters['bias']
    expScale = parameters['expScale']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    yzmean = np.concatenate((ymean, 
                             np.tile(ymean, (Np, )), 
                             np.tile(umean, (Np, ))))
    yzstd = np.concatenate((ystd, 
                            np.tile(ystd, (Np, )),
                            np.tile(ustd, (Np, ))))
    
    # Scale.
    yz = (yz - yzmean)/yzstd
    u = (u - umean)/ustd

    # Get current output.
    nnInput = np.concatenate((yz, u))
    yplus = iCNN(nnInput, zWeights, yWeights, bias, expScale)
    
    # Concatenate.
    if Np > 0:
        yzplus = np.concatenate((yplus, yz[Ny:(Np+1)*Ny], yz[-(Np-1)*Nu:], u))
    else:
        yzplus = yplus

    # Scale back.
    yzplus = yzplus*yzstd + yzmean

    # Return the sum.
    return yzplus
