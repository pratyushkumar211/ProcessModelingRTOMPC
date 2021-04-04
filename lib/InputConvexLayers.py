"""
Custom neural network layers for black-box modeling 
using input convex neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf

class InputConvexLayer(tf.keras.layers.Layer):
    """
    Input convex layer.
    z_{i+1} = g(W^(z)*z_i + W^y*y + b)
    """
    def __init__(self, zPlusDim, zDim, yDim, Wz=False, expActivation=False, 
                 **kwargs):
        super(InputConvexLayer, self).__init__(**kwargs)

        # SAve activation function information.
        self.expActivation = expActivation

        # Create Wz.
        if Wz:
            WzInit =  tf.random_normal_initializer()
            self.Wz = tf.Variable(initial_value = 
                                  WzInit(shape=(zDim, zPlusDim)),
                                  trainable=True,
                                  constraints=tf.keras.constraints.NonNeg())
        else:
            self.Wz = None
        
        # Create Wy.
        WyInit =  tf.random_normal_initializer()
        self.Wy = tf.Variable(initial_value = WyInit(shape=(yDim, zPlusDim)),
                              trainable=True)

        # Create bias.
        bInit =  tf.zeros_initializer()
        self.b = tf.Variable(initial_value = bInit(shape=(zPlusDim, )),
                             trainable=True)
    
    def call(self, z, y):
        """ Call function of the input convex NN layer. """
        
        if self.Wz is None:
            a = tf.matmul(y,self.Wy) + self.b
        else:
            a = tf.matmul(z,self.Wz) + tf.matmul(y,self.Wy) + self.b         

        if self.expActivation:
            zplus = tf.math.exp(a)
        else:
            zplus = a

        # Return output.
        return zplus

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
        self.Np, self.Ny, self.Nu = Np, Ny, Nu
        self.fNLayers = fNLayers

    @property
    def state_size(self):
        return self.Ny + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny
    
    def iCNN(self, nnInput, fNLayers):
        """ Forward propagation for the input
            convex neural network. """
        nnOutput = nnInput
        for layer in fNLayers:
            nnOutput = layer(nnOutput, nnInput)
        # Return output.
        return nnOutput

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

        if Np > 0:
            (y, ypseq, upseq) = tf.split(yz, [Ny, Np*Ny, Np*Nu],
                                         axis=-1)
        else:
            y = yz

        # Get the current output/state and the next time step.
        nnInput = tf.concat((yz, u), axis=-1)
        yplus = self.iCNN(nnInput, self.fNLayers)

        # State at the next time step.
        if Np > 0:
            yzplus = tf.concat((yplus, ypseq[..., Ny:], y, upseq[..., Nu:], u),
                               axis=-1)
        else:
            yzplus = yplus

        # Return output and states at the next time-step.
        return (y, yzplus)

class InputConvexModel(tf.keras.Model):
    """ Custom model for the Two reaction model. """
    def __init__(self, Np, Ny, Nu, fNDims):
        """ Create the dense layers for the NN, and 
            construct the overall model. """

        # Get the size and input layer, and initial state layer.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        yz0 = tf.keras.Input(name='yz0', shape=(Ny + Np*(Ny+Nu), ))

        # Get Input Convex NN layers.
        assert len(fNDims) > 2
        fNLayers = [InputConvexLayer(fNDims[1], fNDims[0], Ny,
                                     expActivation=True)]
        for (zDim, zPlusDim) in zip(fNDims[1:-2], fNDims[2:-1]):
            fNLayers += [InputConvexLayer(zPlusDim, zDim, Ny, Wz=True, 
                                              expActivation=True)]
        fNLayers += [tf.keras.layers.Dense(fNims[-1], fNDims[-2], Ny, Wz=True)]

        # Build model.
        icCell = InputConvexCell(Np, Ny, Nu, fNLayers)

        # Construct the RNN layer and the computation graph.
        icRnnLayer = tf.keras.layers.RNN(icCell, return_sequences=True)
        yseq = bbLayer(inputs=useq, initial_state=[yz0])

        # Construct model.
        super().__init__(inputs=[useq, yz0], outputs=yseq)