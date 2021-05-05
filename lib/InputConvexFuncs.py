"""
Custom neural network layers for black-box modeling 
using input convex neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf

def approxReluTF(x, expScale=0.1):
    """ Scaled exponential to use as activation function. """
    return tf.math.exp(expScale*x)

def iCNNTF(nnInput, nnLayers):
    """ Compute the output of the feedforward network. """
    nnOutput = nnInput
    for layer in nnLayers:
        nnOutput = layer(nnOutput, nnInput)
    return nnOutput

class InputConvexLayer(tf.keras.layers.Layer):
    """
    Input convex layer.
    u_{i+1} = g1(Wut @ u + but)
    z_{i+1} = Wz @ (z*g2(Wzu @ u + bzu)) + Wy @ (y*(Wyu @ u + byu)) 
    z_{i+1} += Wu @ u + bz
    z_{i+1} = g2(z_{i+1})
    Wz = 0 or Wz >= 0
    g1 is tanh, g2 is approx smooth Relu.
    """
    def __init__(self, zPlusDim, zDim, yDim, 
                       uPlusDim, udim, layerPos, **kwargs):
        super(InputConvexLayer, self).__init__(**kwargs)

        # Check for layerPos string.
        if layerPos not in ["First", "Mid", "Last"]:
            raise ValueError("Layer position not found.")
        else:
            self.layerPos = layerPos

        # Random initializer.
        initializer =  tf.random_normal_initializer()

        # Create Wz, Wzu, and bzu.
        if layerPos == "Mid" or layerPos == "Last":
            
            self.Wz = tf.Variable(initial_value = 
                                  initializer(shape=(zDim, zPlusDim)),
                                  trainable=True,
                                  constraint=tf.keras.constraints.NonNeg())

            self.Wzu = tf.Variable(initial_value = 
                                  initializer(shape=(uDim, zDim)),
                                  trainable=True)

            self.bzu = tf.Variable(initial_value = 
                                    biasInit(shape=(zDim, )),
                                    trainable=True)

        # Create Wut and but.
        if layerPos == "First" or layerPos == "Mid":

            self.Wut = tf.Variable(initial_value = 
                                  initializer(shape=(uDim, uPlusDim)),
                                  trainable=True)

            self.but = tf.Variable(initial_value = 
                                    biasInit(shape=(uPlusDim, )),
                                    trainable=True)

        # Create Wy, Wyu, byu, Wu, and bz.
        # These 5 weights are used regardless of the layer position.
        self.Wy = tf.Variable(initial_value = 
                                initializer(shape=(yDim, zPlusDim)),
                                trainable=True)
        self.Wyu = tf.Variable(initial_value = 
                                initializer(shape=(uDim, yDim)),
                                trainable=True)
        self.byu = tf.Variable(initial_value = 
                                biasInit(shape=(yDim, )),
                                trainable=True)
        self.Wu = tf.Variable(initial_value = 
                                initializer(shape=(uDim, zPlusDim)),
                                trainable=True)
        self.bz = tf.Variable(initial_value = 
                                biasInit(shape=(zPlusDim, )),
                                trainable=True)

    def call(self, z, u, y):
        """ Call function of the input convex NN layer. """
        
        # Get uplus.
        if self.layerPos == "First" or self.layerPos == "Mid":
            uplus = tf.math.tanh(tf.linalg.matmul(u, self.Wut) + self.but)
        else:
            uplus = None

        # Get zplus.
        zplus = tf.linalg.matmul(u, self.Wyu) + self.byu
        zplus = tf.math.multiply(zplus, y)
        zplus = tf.linalg.matmul(zplus, self.Wy) 
        zplus += tf.linalg.matmul(u, self.Wu) + self.bz
        # Get the driving term related to z.
        if self.layerPos == "Mid" or self.layerPos == "Last":
            zplusz = approxReluTF(tf.linalg.matmul(u, self.Wzu) + self.bzu)
            zplusz = tf.math.multiply(zplusz, z)
            zplusz = tf.linalg.matmul(zplusz, self.Wz)
            zplus += zplusz
        if self.layerPos == "First" or self.layerPos == "Mid":
            zplus = approxReluTF(zplus)

        # Return output.
        return zplus, uplus

class PartialInputConvexLayer(tf.keras.layers.Layer):
    """
    Input convex layer.
    u_{i+1} = g1(Wut @ u + but)
    z_{i+1} = Wz @ (z*g2(Wzu @ u + bzu)) + Wy @ (y*(Wyu @ u + byu)) 
    z_{i+1} += Wu @ u + bz
    z_{i+1} = g2(z_{i+1})
    Wz = 0 or Wz >= 0
    g1 is tanh, g2 is approx smooth Relu.
    """
    def __init__(self, zPlusDim, zDim, yDim, 
                       uPlusDim, udim, layerPos, **kwargs):
        super(PartialInputConvexLayer, self).__init__(**kwargs)

        # Check for layerPos string.
        if layerPos not in ["First", "Mid", "Last"]:
            raise ValueError("Layer position not found.")
        else:
            self.layerPos = layerPos

        # Random initializer.
        initializer =  tf.random_normal_initializer()

        # Create Wz, Wzu, and bzu.
        if layerPos == "Mid" or layerPos == "Last":
            
            self.Wz = tf.Variable(initial_value = 
                                  initializer(shape=(zDim, zPlusDim)),
                                  trainable=True,
                                  constraint=tf.keras.constraints.NonNeg())

            self.Wzu = tf.Variable(initial_value = 
                                  initializer(shape=(uDim, zDim)),
                                  trainable=True)

            self.bzu = tf.Variable(initial_value = 
                                    biasInit(shape=(zDim, )),
                                    trainable=True)

        # Create Wut and but.
        if layerPos == "First" or layerPos == "Mid":

            self.Wut = tf.Variable(initial_value = 
                                  initializer(shape=(uDim, uPlusDim)),
                                  trainable=True)

            self.but = tf.Variable(initial_value = 
                                    biasInit(shape=(uPlusDim, )),
                                    trainable=True)

        # Create Wy, Wyu, byu, Wu, and bz.
        # These 5 weights are used regardless of the layer position.
        self.Wy = tf.Variable(initial_value = 
                                initializer(shape=(yDim, zPlusDim)),
                                trainable=True)
        self.Wyu = tf.Variable(initial_value = 
                                initializer(shape=(uDim, yDim)),
                                trainable=True)
        self.byu = tf.Variable(initial_value = 
                                biasInit(shape=(yDim, )),
                                trainable=True)
        self.Wu = tf.Variable(initial_value = 
                                initializer(shape=(uDim, zPlusDim)),
                                trainable=True)
        self.bz = tf.Variable(initial_value = 
                                biasInit(shape=(zPlusDim, )),
                                trainable=True)

    def call(self, z, u, y):
        """ Call function of the input convex NN layer. """
        
        # Get uplus.
        if self.layerPos == "First" or self.layerPos == "Mid":
            uplus = tf.math.tanh(tf.linalg.matmul(u, self.Wut) + self.but)
        else:
            uplus = None

        # Get zplus.
        zplus = tf.linalg.matmul(u, self.Wyu) + self.byu
        zplus = tf.math.multiply(zplus, y)
        zplus = tf.linalg.matmul(zplus, self.Wy) 
        zplus += tf.linalg.matmul(u, self.Wu) + self.bz
        # Get the driving term related to z.
        if self.layerPos == "Mid" or self.layerPos == "Last":
            zplusz = approxReluTF(tf.linalg.matmul(u, self.Wzu) + self.bzu)
            zplusz = tf.math.multiply(zplusz, z)
            zplusz = tf.linalg.matmul(zplusz, self.Wz)
            zplus += zplusz
        if self.layerPos == "First" or self.layerPos == "Mid":
            zplus = approxReluTF(zplus)

        # Return output.
        return zplus, uplus

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
