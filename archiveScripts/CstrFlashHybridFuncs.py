# [depends] BlackBoxFuncs.py
"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import mpctools as mpc
import tensorflow as tf
from BlackBoxFuncs import fnnTF, fnn
from TwoReacHybridFuncs import train_model, get_val_predictions
from TwoReacHybridFuncs import hybrid_hx

class CstrFlashHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_N(xG), y = xG
    """
    def __init__(self, fNLayers, xuyscales, hyb_greybox_pars, **kwargs):
        super(CstrFlashHybridCell, self).__init__(**kwargs)

        # Save attributes.
        self.fNLayers = fNLayers
        self.hyb_greybox_pars = hyb_greybox_pars
        self.xuyscales = xuyscales
    
    @property
    def state_size(self):
        return self.hyb_greybox_pars['Nx']
    
    @property
    def output_size(self):
        return self.hyb_greybox_pars['Nx']       

    def _fxu(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. """
        
        # Extract the parameters.
        alphaA = self.hyb_greybox_pars['alphaA']
        alphaB = self.hyb_greybox_pars['alphaB']
        alphaC = self.hyb_greybox_pars['alphaC']
        pho = self.hyb_greybox_pars['pho']
        Cp = self.hyb_greybox_pars['Cp']
        Ar = self.hyb_greybox_pars['Ar']
        Ab = self.hyb_greybox_pars['Ab']
        kr = self.hyb_greybox_pars['kr']
        kb = self.hyb_greybox_pars['kb']
        delH1 = self.hyb_greybox_pars['delH1']
        delH2 = self.hyb_greybox_pars['delH2']
        Td = self.hyb_greybox_pars['Td']
        Qr = self.hyb_greybox_pars['Qr']
        Qb = self.hyb_greybox_pars['Qb']
        ps = self.hyb_greybox_pars['ps']

        # Get the output of the neural network.
        nnOutput = fnnTF(x, self.fNLayers)
        r1, r2 = nnOutput[..., 0:1], nnOutput[..., 1:2]

        # Scale back to physical states.
        xmean, xstd = self.xuyscales['yscale']
        Castd, Ccstd = xstd[1:2], xstd[3:4]
        umean, ustd = self.xuyscales['uscale']
        x = x*xstd + xmean
        u = u*ustd + umean

        # Extract the plant states into meaningful names.
        (Hr, CAr, CBr, 
         CCr, Tr) = (x[..., 0:1], x[..., 1:2], x[..., 2:3], 
                     x[..., 3:4], x[..., 4:5])
        (Hb, CAb, CBb, 
         CCb, Tb) = (x[..., 5:6], x[..., 6:7], x[..., 7:8], 
                     x[..., 8:9], x[..., 9:10])
        F, D = u[..., 0:1], u[..., 1:2]
        CAf, Tf = ps[0], ps[1]

        # The flash vapor phase mass fractions.
        den = alphaA*CAb + alphaB*CBb + alphaC*CCb
        CAd = alphaA*CAb/den
        CBd = alphaB*CBb/den
        CCd = alphaC*CCb/den

        # The outlet mass flow rates.
        Fr = kr*tf.math.sqrt(Hr)
        Fb = kb*tf.math.sqrt(Hb)

        # Write the CSTR odes.
        dHrbydt = (F + D - Fr)/Ar
        dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1*Castd
        dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1*Castd - 3*r2*Ccstd
        dCCrbydt = (-F*CCr + D*(CCd - CCr))/(Ar*Hr) + r2*Ccstd
        dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
        dTrbydt = dTrbydt + (r1*Castd*delH1 + r2*Ccstd*delH2)/(pho*Cp)
        dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

        # Write the flash odes.
        dHbbydt = (Fr - Fb - D)/Ab
        dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
        dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
        dCCbbydt = (Fr*(CCr - CCb) + D*(CCb - CCd))/(Ab*Hb)
        dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

        # Get the scaled derivative.
        xdot = tf.concat([dHrbydt, dCArbydt, dCBrbydt, dCCrbydt, dTrbydt,
                dHbbydt, dCAbbydt, dCBbbydt, dCCbbydt, dTbbydt], axis=-1)/xstd

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """
        # Extract states/inputs.
        [x] = states
        u = inputs

        # Sample time.
        Delta = self.hyb_greybox_pars['Delta']        

        # Get k1, k2, k3, and k4.
        k1 = self._fxu(x, u)
        k2 = self._fxu(x + Delta*(k1/2), u)
        k3 = self._fxu(x + Delta*(k2/2), u)
        k4 = self._fxu(x + Delta*k3, u)
        
        # Get the current output/state and the next time step.
        y = x
        xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Return output and states at the next time-step.
        return (y, xplus)

class InterpolationLayer(tf.keras.layers.Layer):
    """
    The layer to perform interpolation for RK4 predictions.
    """
    def __init__(self, p, Np, trainable=False, name=None):
        super(InterpolationLayer, self).__init__(trainable, name)
        self.p = p
        self.Np = Np

    def call(self, yseq):
        """ The main call function of the interpolation layer. 
        y is of dimension: (None, (Np+1)*p)
        Return y of dimension: (None, Np*p)
        """
        yseq_interp = []
        for t in range(self.Np):
            yseq_interp.append(0.5*(yseq[..., t*self.p:(t+1)*self.p] + 
                                    yseq[..., (t+1)*self.p:(t+2)*self.p]))
        # Return.
        return tf.concat(yseq_interp, axis=-1)

class CstrFlashModel(tf.keras.Model):
    """ Custom model for the CSTR Flash model. """
    def __init__(self, fNDims, xuyscales, hyb_greybox_pars):

        # Get the size and input layer, and initial state layer.
        Nx, Ny = hyb_greybox_pars['Nx'], hyb_greybox_pars['Ny']
        Nu = hyb_greybox_pars['Nu']

        # Create inputs to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        x0 = tf.keras.Input(name='x0', shape=(Nx, ))
        
        # Dense layers for the NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1])]

        # Build model.
        cstr_flash_cell = CstrFlashHybridCell(fNLayers, xuyscales, 
                                              hyb_greybox_pars)

        # Construct the RNN layer and the computation graph.
        cstr_flash_layer = tf.keras.layers.RNN(cstr_flash_cell,
                                               return_sequences=True)
        yseq = cstr_flash_layer(inputs=useq, initial_state=[x0])

        # Construct model.
        super().__init__(inputs=[useq, x0], outputs=yseq)

def create_model(*, fNDims, xuyscales, hyb_greybox_pars):
    """ Create/compile the two reaction model for training. """
    model = CstrFlashModel(fNDims, xuyscales, hyb_greybox_pars)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def get_hybrid_pars(*, train, hyb_greybox_pars):
    """ Get the hybrid model parameters. """

    # Get black-box model parameters.
    parameters = {}
    parameters['fNWeights'] = train['trained_weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    parameters['Nx'] = hyb_greybox_pars['Nx']
    parameters['Np'] = hyb_greybox_pars['Np']
    parameters['Nu'] = hyb_greybox_pars['Nu']
    parameters['Ny'] = hyb_greybox_pars['Ny']

    # Constraints.
    parameters['ulb'] = hyb_greybox_pars['ulb']
    parameters['uub'] = hyb_greybox_pars['uub']
    
    # Grey-box model parameters.
    parameters['Delta'] = hyb_greybox_pars['Delta'] # min
    parameters['alphaA'] = hyb_greybox_pars['alphaA']
    parameters['alphaB'] = hyb_greybox_pars['alphaB']
    parameters['alphaC'] = hyb_greybox_pars['alphaC']
    parameters['pho'] = hyb_greybox_pars['pho']
    parameters['Cp'] = hyb_greybox_pars['Cp']
    parameters['Ar'] = hyb_greybox_pars['Ar']
    parameters['Ab'] = hyb_greybox_pars['Ab']
    parameters['kr'] = hyb_greybox_pars['kr']
    parameters['kb'] = hyb_greybox_pars['kb']
    parameters['delH1'] = hyb_greybox_pars['delH1']
    parameters['delH2'] = hyb_greybox_pars['delH2']
    parameters['Td'] = hyb_greybox_pars['Td']
    parameters['Qb'] = hyb_greybox_pars['Qb']
    parameters['Qr'] = hyb_greybox_pars['Qr']
    parameters['ps'] = hyb_greybox_pars['ps']

    # Return.
    return parameters

def interpolate_pseq(pseq, p, Np):
    """ y is of dimension: (None, (Np+1)*p)
        Return y of dimension: (None, Np*p). """
    pseq_interp = []
    for t in range(Np):
        pseq_interp += [0.5*(pseq[t*p:(t+1)*p] + pseq[(t+1)*p:(t+2)*p])]
    # Return.
    return np.concatenate(pseq_interp)

def fxu(x, u, p, parameters, xuyscales, fNWeights):
    """ Grey-box part of the hybrid model. """

    # Extract the parameters.
    alphaA = parameters['alphaA']
    alphaB = parameters['alphaB']
    alphaC = parameters['alphaC']
    pho = parameters['pho']
    Cp = parameters['Cp']
    Ar = parameters['Ar']
    Ab = parameters['Ab']
    kr = parameters['kr']
    kb = parameters['kb']
    delH1 = parameters['delH1']
    delH2 = parameters['delH2']
    Td = parameters['Td']
    Qr = parameters['Qr']
    Qb = parameters['Qb']

    # Extract the plant states into meaningful names.
    Hr, CAr, CBr, CCr, Tr = x[0:1], x[1:2], x[2:3], x[3:4], x[4:5]
    Hb, CAb, CBb, CCb, Tb = x[5:6], x[6:7], x[7:8], x[8:9], x[9:10]
    F, D = u[0:1], u[1:2]
    p = p.squeeze()
    CAf, Tf = p[0:1], p[1:2]
    
    # Get the scales.
    xmean, xstd = xuyscales['yscale']
    Castd, Ccstd = xstd[1:2], xstd[3:4]
    umean, ustd = xuyscales['uscale']

    # Scale state, inputs, for the NN.
    x = (x - xmean)/xstd
    u = (u - umean)/ustd
    nnOutput = fnn(x, fNWeights)
    r1, r2 = nnOutput[0:1], nnOutput[1:2]

    # Compute recycle flow-rates.
    den = alphaA*CAb + alphaB*CBb + alphaC*CCb
    CAd = alphaA*CAb/den
    CBd = alphaB*CBb/den
    CCd = alphaC*CCb/den

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fb = kb*np.sqrt(Hb)

    # Write the CSTR odes.
    dHrbydt = (F + D - Fr)/Ar
    dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1*Castd
    dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1*Castd - 3*r2*Ccstd
    dCCrbydt = (-F*CCr + D*(CCd - CCr))/(Ar*Hr) + r2*Ccstd
    dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr)
    dTrbydt = dTrbydt + (r1*Castd*delH1 + r2*Ccstd*delH2)/(pho*Cp)
    dTrbydt = dTrbydt - Qr/(pho*Ar*Cp*Hr)

    # Write the flash odes.
    dHbbydt = (Fr - Fb - D)/Ab
    dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
    dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
    dCCbbydt = (Fr*(CCr - CCb) + D*(CCb - CCd))/(Ab*Hb)
    dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

    # Get the scaled derivative.
    xdot = mpc.vcat([dHrbydt, dCArbydt, dCBrbydt, dCCrbydt, dTrbydt,
            dHbbydt, dCAbbydt, dCBbbydt, dCCbbydt, dTbbydt])

    # Return.
    return xdot

def hybrid_fxup(x, u, p, parameters):
    """ The augmented continuous time model. """

    # Sample time.
    Delta = parameters['Delta']

    # NN weights, scaling.
    xuyscales = parameters['xuyscales']
    fNWeights = parameters['fNWeights']

    # Get k1, k2, k3, and k4.
    k1 = fxu(x, u, p, parameters, xuyscales, fNWeights)
    k2 = fxu(x + Delta*(k1/2), u, p, parameters, xuyscales, fNWeights)
    k3 = fxu(x + Delta*(k2/2), u, p, parameters, xuyscales, fNWeights)
    k4 = fxu(x + Delta*k3, u, p, parameters, xuyscales, fNWeights)
    
    # Get the current output/state and the next time step.
    xplus = x + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Return the sum.
    return xplus