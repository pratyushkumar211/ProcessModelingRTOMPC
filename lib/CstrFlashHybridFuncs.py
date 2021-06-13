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

class CstrFlashHybridCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN Cell
    dxG/dt  = fG(xG, u) + f_N(xG, y_{k-N_p:k-1}, u_{k-N_p:k-1}), y = xG
    """
    def __init__(self, Np, interpLayer, fNLayers, xuyscales, 
                       greybox_pars, **kwargs):
        super(CstrFlashHybridCell, self).__init__(**kwargs)

        # Save attributes.
        self.Np = Np
        self.interpLayer = interpLayer
        self.fNLayers = fNLayers
        self.parameters = greybox_pars
        self.xuyscales = xuyscales
        self.Nx, self.Ny, self.Nu = (greybox_pars['Nx'],
                                     greybox_pars['Ny'],
                                     greybox_pars['Nu'])
    
    @property
    def state_size(self):
        return self.Nx + self.Np*(self.Ny + self.Nu)
    
    @property
    def output_size(self):
        return self.Ny        

    def _fg(self, x, u):
        """ Function to compute the 
            derivative (RHS of the ODE)
            for the two reaction model. """
        
        # Extract the parameters.
        alphaA = self.parameters['alphaA']
        alphaB = self.parameters['alphaB']
        pho = self.parameters['pho']
        Cp = self.parameters['Cp']
        Ar = self.parameters['Ar']
        Ab = self.parameters['Ab']
        kr = self.parameters['kr']
        kb = self.parameters['kb']
        delH1 = self.parameters['delH1']
        E1byR = self.parameters['E1byR']
        k1star = self.parameters['k1star']
        Td = self.parameters['Td']
        Qr = self.parameters['Qr']
        Qb = self.parameters['Qb']
        ps = self.parameters['ps']

        # Scale back to physical states.
        ymean, ystd = self.xuyscales['yscale']
        umean, ustd = self.xuyscales['uscale']
        x = x*ystd + ymean
        u = u*ustd + umean

        # Extract the plant states into meaningful names.
        Hr, CAr, CBr, Tr = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4] 
        Hb, CAb, CBb, Tb = x[..., 4:5], x[..., 5:6], x[..., 6:7], x[..., 7:8] 
        F, D = u[..., 0:1], u[..., 1:2]
        CAf, Tf = ps[0], ps[1]
        
        # Compute recycle flow-rates.
        den = alphaA*CAb + alphaB*CBb
        CAd = alphaA*CAb/den
        CBd = alphaB*CBb/den

        # The outlet mass flow rates.
        Fr = kr*tf.math.sqrt(Hr)
        Fb = kb*tf.math.sqrt(Hb)
        
        # Get rate of reactions.
        k1 = k1star*tf.math.exp(-E1byR/Tr)
        r1 = k1*CAr

        # Write the CSTR odes.
        dHrbydt = (F + D - Fr)/Ar
        dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
        dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
        dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr) - Qr/(pho*Ar*Cp*Hr)
        dTrbydt += (r1*delH1)/(pho*Cp)

        # Write the flash odes.
        dHbbydt = (Fr - Fb - D)/Ab
        dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
        dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
        dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

        # Get the scaled derivative.
        xdot = tf.concat([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
                          dHbbydt, dCAbbydt, dCBbbydt, dTbbydt], axis=-1)/ystd

        # Return the derivative.
        return xdot

    def call(self, inputs, states):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Ng + Np*(Ny + Nu))
            Dimension of input: (None, Nu)
        """
        # Extract variables.
        [xGz] = states
        [xG, z] = tf.split(xGz, [self.Nx, self.Np*(self.Ny+self.Nu)],
                           axis=-1)
        (ypseq, upseq) = tf.split(z, [self.Np*self.Ny, self.Np*self.Nu],
                                  axis=-1)
        u = inputs

        # Sample time.
        Delta = self.parameters['Delta']

        # Get k1.
        nnInput = tf.concat((xG, z, u), axis=-1)
        k1 = self._fg(xG, u) + fnnTF(nnInput, self.fNLayers)

        # Interpolate for k2 and k3.
        ypseqInterp = self.interpLayer(tf.concat((ypseq, xG), axis=-1))
        z = tf.concat((ypseqInterp, upseq), axis=-1)
        
        # Get k2.
        nnInput = tf.concat((xG + Delta*(k1/2), z, u), axis=-1)
        k2 = self._fg(xG + Delta*(k1/2), u) + fnnTF(nnInput, self.fNLayers)

        # Get k3.
        nnInput = tf.concat((xG + Delta*(k2/2), z, u), axis=-1)
        k3 = self._fg(xG + Delta*(k2/2), u) + fnnTF(nnInput, self.fNLayers)

        # Get k4.
        ypseqShifted = tf.concat((ypseq[..., self.Ny:], xG), axis=-1)
        z = tf.concat((ypseqShifted, upseq), axis=-1)
        nnInput = tf.concat((xG + Delta*k3, z, u), axis=-1)
        k4 = self._fg(xG + Delta*k3, u) + fnnTF(nnInput, self.fNLayers)
        
        # Get the current output/state and the next time step.
        y = xG
        xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
        zplus = tf.concat((ypseqShifted, upseq[..., self.Nu:], u), axis=-1)
        xplus = tf.concat((xGplus, zplus), axis=-1)
        
        # Return output and states at the next time-step.
        return y, xplus

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
    def __init__(self, Np, fNDims, xuyscales, greybox_pars):

        # Get the size and input layer, and initial state layer.
        Nx, Ny = greybox_pars['Nx'], greybox_pars['Ny']
        Nu = greybox_pars['Nu']

        # Create inputs to the model.
        useq = tf.keras.Input(name='u', shape=(None, Nu))
        xGz0 = tf.keras.Input(name='xGz0', shape=(Nx + Np*(Ny+Nu), ))
        
        # Dense layers for the NN.
        fNLayers = []
        for dim in fNDims[1:-1]:
            fNLayers += [tf.keras.layers.Dense(dim, activation='tanh')]
        fNLayers += [tf.keras.layers.Dense(fNDims[-1], 
                                           kernel_initializer='zeros')]

        # Build model.
        interpLayer = InterpolationLayer(p=Ny, Np=Np)
        cstr_flash_cell = CstrFlashHybridCell(Np, interpLayer, fNLayers,
                                              xuyscales, greybox_pars)

        # Construct the RNN layer and the computation graph.
        cstr_flash_layer = tf.keras.layers.RNN(cstr_flash_cell,
                                               return_sequences=True)
        yseq = cstr_flash_layer(inputs=useq, initial_state=[xGz0])

        # Construct model.
        super().__init__(inputs=[useq, xGz0], outputs=yseq)

def create_model(*, Np, fNDims, xuyscales, greybox_pars):
    """ Create/compile the two reaction model for training. """
    model = CstrFlashModel(Np, fNDims, xuyscales, greybox_pars)
    # Compile the nn model.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Return the compiled model.
    return model

def get_CstrFlash_hybrid_pars(*, train, greybox_pars):
    """ Get the hybrid model parameters. """

    # Get black-box model parameters.
    parameters = {}
    parameters['fNWeights'] = train['trained_weights'][-1]
    parameters['xuyscales'] = train['xuyscales']

    # Sizes.
    Nx, Ny, Nu = greybox_pars['Nx'], greybox_pars['Ny'], greybox_pars['Nu']
    parameters['Ny'], parameters['Nu'] = Ny, Nu
    parameters['Nx'] = Nx + train['Np']*(Ny + Nu)
    parameters['Np'] = train['Np']

    # Constraints.
    parameters['ulb'] = greybox_pars['ulb']
    parameters['uub'] = greybox_pars['uub']
    
    # Grey-box model parameters.
    parameters['Delta'] = greybox_pars['Delta'] # min
    parameters['alphaA'] = greybox_pars['alphaA']
    parameters['alphaB'] = greybox_pars['alphaB']
    parameters['pho'] = greybox_pars['pho']
    parameters['Cp'] = greybox_pars['Cp']
    parameters['Ar'] = greybox_pars['Ar']
    parameters['Ab'] = greybox_pars['Ab']
    parameters['kr'] = greybox_pars['kr']
    parameters['kb'] = greybox_pars['kb']
    parameters['E1byR'] = greybox_pars['E1byR']
    parameters['delH1'] = greybox_pars['delH1']
    parameters['k1star'] = greybox_pars['k1star']
    parameters['Td'] = greybox_pars['Td']
    parameters['Qb'] = greybox_pars['Qb']
    parameters['Qr'] = greybox_pars['Qr']
    parameters['ps'] = greybox_pars['ps']

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

def fgreybox(x, u, parameters):
    """ Grey-box part of the hybrid model. """

    # Extract the parameters.
    alphaA = parameters['alphaA']
    alphaB = parameters['alphaB']
    pho = parameters['pho']
    Cp = parameters['Cp']
    Ar = parameters['Ar']
    Ab = parameters['Ab']
    kr = parameters['kr']
    kb = parameters['kb']
    delH1 = parameters['delH1']
    E1byR = parameters['E1byR']
    k1star = parameters['k1star']
    Td = parameters['Td']
    Qr = parameters['Qr']
    Qb = parameters['Qb']
    ps = parameters['ps']

    # Extract the plant states into meaningful names.
    Hr, CAr, CBr, Tr = x[0:4]
    Hb, CAb, CBb, Tb = x[4:8]
    F, D = u[0:2]
    CAf, Tf = ps[0:2]
    
    # Compute recycle flow-rates.
    den = alphaA*CAb + alphaB*CBb
    CAd = alphaA*CAb/den
    CBd = alphaB*CBb/den

    # The outlet mass flow rates.
    Fr = kr*np.sqrt(Hr)
    Fb = kb*np.sqrt(Hb)
    
    # Get rate of reaction.
    k1 = k1star*np.exp(-E1byR/Tr)
    r1 = k1*CAr

    # Write the CSTR odes.
    dHrbydt = (F + D - Fr)/Ar
    dCArbydt = (F*(CAf - CAr) + D*(CAd - CAr))/(Ar*Hr) - r1
    dCBrbydt = (-F*CBr + D*(CBd - CBr))/(Ar*Hr) + r1
    dTrbydt = (F*(Tf - Tr) + D*(Td - Tr))/(Ar*Hr) - Qr/(pho*Ar*Cp*Hr)
    dTrbydt += (r1*delH1)/(pho*Cp)

    # Write the flash odes.
    dHbbydt = (Fr - Fb - D)/Ab
    dCAbbydt = (Fr*(CAr - CAb) + D*(CAb - CAd))/(Ab*Hb)
    dCBbbydt = (Fr*(CBr - CBb) + D*(CBb - CBd))/(Ab*Hb)
    dTbbydt = (Fr*(Tr - Tb))/(Ab*Hb) + Qb/(pho*Ab*Cp*Hb)

    # Get the derivative.
    xdot = np.array([dHrbydt, dCArbydt, dCBrbydt, dTrbydt,
                     dHbbydt, dCAbbydt, dCBbbydt, dTbbydt])

    # Return.
    return xdot

def CstrFlashHybrid_fxu(x, u, parameters):
    """ The augmented continuous time model. """

    # Sizes.
    Nx, Ny, Nu = parameters['Nx'], parameters['Ny'], parameters['Nu']
    Np = parameters['Np']

    # Sample time.
    Delta = parameters['Delta']

    # NN weights.
    fNWeights = parameters['fNWeights']

    # Get scaling.
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    xmean = np.concatenate((np.tile(ymean, (Np + 1, )), 
                            np.tile(umean, (Np, ))))
    xstd = np.concatenate((np.tile(ystd, (Np + 1, )), 
                           np.tile(ustd, (Np, ))))

    # Scale.
    x = (x - xmean)/xstd
    u = (u - umean)/ustd

    # Extract vectors.
    xG, ypseq, upseq = x[:Ny], x[Ny:Ny + Np*Ny], x[-Np*Nu:]
    z = x[Ny:]
    
    # Get k1.
    nnInput = np.concatenate((xG, z, u))
    k1 = fgreybox(xG*ystd + ymean, u*ustd + umean, parameters)/ystd
    k1 +=  fnn(nnInput, fNWeights)

    # Interpolate for k2 and k3.
    ypseqInterp = interpolate_pseq(np.concatenate((ypseq, xG)), Ny, Np)
    z = np.concatenate((ypseqInterp, upseq))
    
    # Get k2.
    nnInput = np.concatenate((xG + Delta*(k1/2), z, u))
    k2 = fgreybox((xG + Delta*(k1/2))*ystd + ymean, u*ustd + umean, 
                       parameters)/ystd
    k2 += fnn(nnInput, fNWeights)

    # Get k3.
    nnInput = np.concatenate((xG + Delta*(k2/2), z, u))
    k3 = fgreybox((xG + Delta*(k2/2))*ystd + ymean, u*ustd + umean, 
                        parameters)/ystd
    k3 += fnn(nnInput, fNWeights)

    # Get k4.
    ypseqShifted = np.concatenate((ypseq[Ny:], xG))
    z = np.concatenate((ypseqShifted, upseq))
    nnInput = np.concatenate((xG + Delta*k3, z, u))
    k4 = fgreybox((xG + Delta*k3)*ystd + ymean, u*ustd + umean, 
                      parameters)/ystd
    k4 += fnn(nnInput, fNWeights)
    
    # Get the state at the next time step.
    xGplus = xG + (Delta/6)*(k1 + 2*k2 + 2*k3 + k4)
    zplus = np.concatenate((ypseqShifted, upseq[Nu:], u))
    xplus = np.concatenate((xGplus, zplus))
    xplus = xplus*xstd + xmean

    # Return the sum.
    return xplus

def CstrFlashHybrid_hx(x, parameters):
    """ Measurement function. """
    Ny = parameters['Ny']
    # Return only the x.
    return x[:Ny]