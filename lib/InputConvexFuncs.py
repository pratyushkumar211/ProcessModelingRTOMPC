"""
Custom neural network layers for black-box modeling 
using input convex neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf

def approxRelu(x, TF=True, k=4):
    """ Scaled exponential to use as activation function. """
    if TF:
        return tf.math.log(1. + tf.math.exp(k*x))/k
    else:
        return np.log(1. + np.exp(k*x))/k

def icnnTF(y, nnLayers):
    """ Compute the output of the feedforward network. """
    z = None
    for layer in nnLayers:
        z = layer(z, y)
    return z

def picnnTF(y, x, nnLayers):
    """ Compute the output of the feedforward network. """
    z = None
    u = x
    for layer in nnLayers:
        z, u = layer(z, u, y)
    return z

class InputConvexLayer(tf.keras.layers.Layer):
    """
    Input convex layer.
    z_{i+1} = g(Wz @ z + Wy @ y + b)
    Wz = 0 or Wz >= 0
    g is approx smooth Relu.
    """
    def __init__(self, zPlusDim, zDim, yDim, layerPos, **kwargs):
        super(InputConvexLayer, self).__init__(**kwargs)

        # Check for layerPos string.
        if layerPos not in ["First", "Mid", "Last"]:
            raise ValueError("Layer position not found.")
        else:
            self.layerPos = layerPos

        # Random initializer.
        initializer =  tf.random_normal_initializer()

        # Create Wz.
        if layerPos == "Mid" or layerPos == "Last":
            
            self.Wz = tf.Variable(initial_value = 
                                  initializer(shape=(zDim, zPlusDim)),
                                  trainable=True,
                                  constraint=tf.keras.constraints.NonNeg())

        # Create Wy and bz.
        # These 2 weights are used regardless of the layer position.
        self.Wy = tf.Variable(initial_value = 
                                initializer(shape=(yDim, zPlusDim)),
                                trainable=True)
        self.b = tf.Variable(initial_value = 
                                biasInit(shape=(zPlusDim, )),
                                trainable=True)

    def call(self, z, y):
        """ Call function of the input convex NN layer. """
        
        # Get zplus.
        zplus = tf.linalg.matmul(y, self.Wy) + self.b

        # Get the driving term related to z.
        if self.layerPos == "Mid" or self.layerPos == "Last":
            zplus += tf.linalg.matmul(z, self.Wz)

        # Apply activation only if first or middle layer.
        if self.layerPos == "First" or self.layerPos == "Mid":
            zplus = approxRelu(zplus)

        # Return output.
        return zplus

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
                       uPlusDim, uDim, layerPos, **kwargs):
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
            zplusz = approxRelu(tf.linalg.matmul(u, self.Wzu) + self.bzu)
            zplusz = tf.math.multiply(zplusz, z)
            zplusz = tf.linalg.matmul(zplusz, self.Wz)
            zplus += zplusz
        # Apply activation only if first or middle layer.
        if self.layerPos == "First" or self.layerPos == "Mid":
            zplus = approxRelu(zplus)

        # Return output.
        return zplus, uplus

class InputConvexModel(tf.keras.Model):

    """ Input convex neural network model. """
    def __init__(self, Ny, zDims, uDims=None):
        
        # Get the input layers (from which convexity is required).
        InputList = []
        y = tf.keras.Input(name='y', shape=(Ny, ))
        InputList += [y]

        # Get the input layer (from which convexity is not required).
        if uDims is not None:
            x = tf.keras.Input(name='x', shape=(uDims[0], ))
            InputList += [x]
        
        # Check for no initial size in zDims.
        assert zDims[0] is None, "Remove initial size in zDims."

        # Get the layers.
        if uDims is not None:

            # Check for at least three layer values. 
            assert len(zDims) > 2, "Check zDims size."
            assert len(uDims) > 2, "Check uDims size."
            assert len(uDims) == len(zDims), """ Dimensions of zDims 
                                                 and uDims not same. """
            # Check for no last size in uDims.
            assert uDims[-1] is None, "Remove last size in uDims."

            # Create layers.
            fNLayers = [PartialInputConvexLayer(zDims[1], zDims[0], Ny, 
                                                uDims[1], uDims[0], "First")]
            for (zDim, zPlusDim, 
                 uDim, uPlusDim) in zip(zDims[1:-2], zDims[2:-1], 
                                        uDims[1:-2], uDims[2:-1]):
                fNLayers += [PartialInputConvexLayer(zPlusDim, zDim, Ny, 
                                                     uPlusDim, uDim, "Mid")]
            fNLayers += [PartialInputConvexLayer(zDims[-1], zDims[-2], Ny, 
                                                 uDims[-1], uDim[-2], "Last")]
            
            # Get symbolic output.
            f = picnnTF(y, x, fNLayers)
        else:
            
            # Check for at least three layer values. 
            assert len(zDims) > 2, "Check zDims size."

            # Create layers.
            fNLayers = [InputConvexLayer(zDims[1], zDims[0], Ny, "First")]
            for (zDim, zPlusDim) in zip(zDims[1:-2], zDims[2:-1]):
                fNLayers += [InputConvexLayer(zPlusDim, zDim, Ny, "Mid")]
            fNLayers += [InputConvexLayer(zDims[-1], zDims[-2], Ny, "Last")]

            # Get symbolic output.
            f = icnnTF(y, fNLayers)

        # Construct model.
        super().__init__(inputs=InputList, outputs=f)

def icnn(y, Wz_list, Wy_list, b_list):
    """ Compute the NN output. """

    # Check input dimensions. 
    assert y.ndim == 1
    y = y[:, np.newaxis]

    # Out of First layer.
    Wy, b = Wy_list[0], b_list[0]
    z = Wy.T @ y + b[:, np.newaxis]
    z = approxRelu(z, TF=False)

    # Loop over middle layers.
    for Wz, Wy, b in zip(Wz_list[:-1], Wy_list[1:-1], b_list[1:-1]):
        z = Wz.T @ z + Wy.T @ y + b[:, np.newaxis]
        z = approxRelu(z, TF=False)
    
    # Last layer.
    Wz, Wy, b = Wz_list[-1], Wy_list[-1], b_list[-1]
    z = Wz.T @ z + Wy.T @ y + b[:, np.newaxis]

    # Return output in same number of dimensions.
    z = z[:, 0]

    # Return.
    return z

def picnn(y, x, Wut_list, but_list, Wz_list, 
                Wzu_list, bzu_list, Wy_list, 
                Wyu_list, byu_list, Wu_list, bz_list):
    """ Compute the NN output. """

    # Check input dimensions.
    u = x[:, np.newaxis] 
    y = y[:, np.newaxis]

    # First layer.
    # z propagation.
    Wy, Wyu, byu = Wy_list[0], Wyu_list[0], byu_list[0]
    Wu, bz = Wu_list[0], bz_list[0]
    z = Wy.T @ (y*(Wyu.T @ u + byu[:, np.newaxis]))
    z += Wu.T @ u + bz[:, np.newaxis]
    z = approxRelu(z, TF=False)
    # u propagation.
    Wut, but = Wut_list[0], but_list[0]
    u = np.tanh(Wut.T @ u + but[:, np.newaxis])

    # Loop over middle layers.
    for (Wz, Wzu, bzu, Wy, 
         Wyu, byu, Wu, bz, 
         Wut, but) in zip(Wz_list[:-1], Wzu_list[:-1], bzu_list[:-1], 
                          Wy_list[1:-1], Wyu_list[1:-1], byu_list[1:-1], 
                    Wu_list[1:-1], bz_list[1:-1], Wut_list[1:], but_list[1:]):
        z = Wz.T @ (z*(Wzu.T @ u + bzu[:, np.newaxis]))
        z += Wy.T @ (y*(Wyu.T @ u + byu[:, np.newaxis]))
        z += Wu.T @ u + bz[:, np.newaxis]
        z = approxRelu(z, TF=False)
        u = np.tanh(Wut.T @ u + but[:, np.newaxis])

    # Last layer.
    Wz, Wzu, bzu = Wz_list[-1], Wzu_list[-1], bzu_list[-1]
    Wy, Wyu, byu = Wy_list[-1], Wyu_list[-1], byu_list[-1]
    Wu, bz = Wu_list[-1], bz_list[-1]
    z = Wz.T @ (z*(Wzu.T @ u + bzu[:, np.newaxis]))
    z += Wy.T @ (y*(Wyu.T @ u + byu[:, np.newaxis]))
    z += Wu.T @ u + bz[:, np.newaxis]

    # Return output in same number of dimensions.
    z = z[:, 0]

    # Return.
    return z

def icnn_lyu(u, parameters):
    """ Function describing the cost function of 
        the input convex neural network. """

    # Get NN weights.
    Wz_list = parameters['Wz_list']
    Wy_list = parameters['Wy_list']
    b_list = parameters['b_list']

    # Get scaling.
    ulpscales = parameters['ulpscales']
    umean, ustd = xuyscales['uscale']
    lyupmean, lyupstd = xuyscales['lyupscale']
    
    # Scale.
    u = (u - umean)/ustd

    # Get the ICNN cost.
    lyu = icnn(u, Wz_list, Wy_list, b_list)
    
    # Scale back.
    lyu = lyu*lyupstd + lyupmean

    # Return the cost.
    return lyu

def picnn_lyup(u, p, parameters):
    """ Function describing the cost function of 
        the partial input convex neural network. """

    # Get NN weights.
    Wz_list = parameters['Wz_list']
    Wzu_list = parameters['Wzu_list']
    bzu_list = parameters['bzu_list']
    Wy_list = parameters['Wy_list']
    Wyu_list = parameters['Wyu_list']
    byu_list = parameters['byu_list']
    Wu_list = parameters['Wu_list']
    bz_list = parameters['bz_list']
    Wut_list = parameters['Wut_list']
    but_list = parameters['but_list']

    # Get scaling.
    ulpscales = parameters['ulpscales']
    umean, ustd = xuyscales['uscale']
    pmean, pstd = xuyscales['pscale']
    lyupmean, lyupstd = xuyscales['lyupscale']
    
    # Scale.
    u = (u - umean)/ustd
    p = (p - pmean)/pstd

    # Get the ICNN cost.
    lyup = picnn(u, p, Wut_list, but_list, Wz_list, 
                 Wzu_list, bzu_list, Wy_list, 
                 Wyu_list, byu_list, Wu_list, bz_list)
    
    # Scale back.
    lyup = lyup*lyupstd + lyupmean

    # Return the cost.
    return lyup

def get_icnn_pars(*, train):
    """ Get the black-box parameter dict and function handles. """

    # Get model parameters.
    parameters = {}
    parameters['ulpscales'] = train['ulpscales']

    # Get weights.
    numLayers = len(train['fNDims']) - 1
    trained_weights = train['trained_weights'][-1]
    parameters['yWeights'] = trained_weights[slice(0, 3*numLayers, 3)]
    parameters['bias'] = trained_weights[slice(1, 3*numLayers, 3)]
    parameters['zWeights'] = trained_weights[slice(2, 3*numLayers, 3)]
    
    # Return.
    return parameters

def get_picnn_pars(*, train):
    """ Get the black-box parameter dict and function handles. """

    # Get model parameters.
    parameters = {}
    parameters['ulpscales'] = train['ulpscales']

    # Get weights.
    numLayers = len(train['fNDims']) - 1
    trained_weights = train['trained_weights'][-1]
    parameters['yWeights'] = trained_weights[slice(0, 3*numLayers, 3)]
    parameters['bias'] = trained_weights[slice(1, 3*numLayers, 3)]
    parameters['zWeights'] = trained_weights[slice(2, 3*numLayers, 3)]
    
    # Return.
    return parameters

def create_model(*, Nu, zDims, uDims):
    """ Create/compile the two reaction model for training. """
    model = InputConvexModel(Nu, zDims, uDims)
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
    model.fit(x=train_data['inputs'], 
              y=train_data['output'], 
              epochs=epochs, batch_size=batch_size,
        validation_data = (trainval_data['inputs'], trainval_data['output']),
            callbacks = [checkpoint_callback])

def get_val_predictions_metric(*, model, val_data, ulpscales, ckpt_path):
    """ Get the validation predictions. """

    # Load best weights.
    model.load_weights(ckpt_path)

    # Predict.
    val_predictions = model.predict(x=val_data['inputs'])

    # Scale.
    lyupmean, lyupstd = ulpscales['lyupscale']
    val_predictions = val_predictions.squeeze()*lyupstd + lyupmean

    # Get prediction error on the validation data.
    val_metric = model.evaluate(x=val_data['inputs'], y=val_data['output'])

    # Return predictions and metric.
    return val_predictions, val_metric