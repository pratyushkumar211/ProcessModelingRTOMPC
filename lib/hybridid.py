# Pratyush Kumar, pratyushkumar@ucsb.edu

import sys
import numpy as np
import mpctools as mpc
import scipy.linalg
import cvxopt as cvx
import matplotlib.pyplot as plt
import casadi
import collections
import pickle
import plottools

FIGURE_SIZE_A4 = (9, 10)
PRESENTATION_FIGSIZE = (6, 6)
PAPER_FIGSIZE = (4, 4)

SimData = collections.namedtuple('SimData', 
                                ['t', 'x', 'u', 'y'])

class PickleTool:
    """Class which contains a few static methods for saving and
    loading pkl data files conveniently."""
    @staticmethod
    def load(filename, type='write'):
        """Wrapper to load data."""
        if type == 'read':
            with open(filename, "rb") as stream:
                return pickle.load(stream)
        if type == 'write':
            with open(filename, "wb") as stream:
                return pickle.load(stream)
    
    @staticmethod
    def save(data_object, filename):
        """Wrapper to pickle a data object."""
        with open(filename, "wb") as stream:
            pickle.dump(data_object, stream)

def c2d(A, B, sample_time):
    """ Custom c2d function for linear systems."""
    
    # First construct the incumbent matrix
    # to take the exponential.
    (Nx, Nu) = B.shape
    M1 = np.concatenate((A, B), axis=1)
    M2 = np.zeros((Nu, Nx+Nu))
    M = np.concatenate((M1, M2), axis=0)
    Mexp = scipy.linalg.expm(M*sample_time)

    # Return the extracted matrices.
    Ad = Mexp[:Nx, :Nx]
    Bd = Mexp[:Nx, -Nu:]
    return (Ad, Bd)
    
class NonlinearPlantSimulator:
    """Custom class for simulating non-linear plants."""
    def __init__(self, *, fxup, hx, Rv, Nx, Nu, Np, Ny, 
                 sample_time, x0):
        
        # Set attributes.
        (self.Nx, self.Nu, self.Ny, self.Np) = (Nx, Nu, Ny, Np)
        self.fxup = mpc.getCasadiFunc(fxup, [Nx, Nu, Np], 
                                      ['x', 'u', 'p'], 'fxup', 
                                      rk4=True, Delta=sample_time, M=1)
        self.hx = mpc.getCasadiFunc(hx, [Nx], ["x"], funcname="hx")
        self.measurement_noise_std = np.sqrt(np.diag(Rv)[:, np.newaxis])
        self.sample_time = sample_time

        # Create lists to save data.
        self.x = [x0]
        self.u = []
        self.p = []
        self.y = [np.asarray(self.hx(x0)) + 
                 self.measurement_noise_std*np.random.randn(self.Ny, 1)]
        self.t = [0.]

    def step(self, u, p):
        """ Inject the control input into the plant."""
        x = np.asarray(self.fxup(self.x[-1], u, p))
        y = np.asarray(self.hx(x))
        y = y + self.measurement_noise_std*np.random.randn(self.Ny, 1)
        self._append_data(x, u, p, y)
        return y

    def _append_data(self, x, u, p, y):
        """ Append the data into the lists.
            Used for plotting in the specific subclasses.
        """
        self.x.append(x)
        self.u.append(u)
        self.p.append(p)
        self.y.append(y)
        self.t.append(self.t[-1]+self.sample_time)

def _sample_repeats(num_change, num_simulation_steps,
                       mean_change, sigma_change):
    """ Sample the number of times a repeat in each
    of the sampled vector is required."""
    repeat = sigma_change*np.random.randn(num_change-1) + mean_change
    repeat = np.floor(repeat)
    repeat = np.where(repeat<=0., 0., repeat)
    repeat = np.append(repeat, num_simulation_steps-np.int(np.sum(repeat)))
    return repeat.astype(int)

def sample_prbs_like(*, num_change, num_steps, 
                    lb, ub, mean_change, sigma_change, seed=1):
    """Sample a PRBS like sequence.
    num_change: Number of changes in the signal.
    num_simulation_steps: Number of steps in the signal.
    mean_change: mean_value after which a 
                 change in the signal is desired.
    sigma_change: standard deviation of changes in the signal.
    """
    signal_dimension = lb.shape[0]
    lb = lb.squeeze() # Squeeze the vectors.
    ub = ub.squeeze() # Squeeze the vectors.
    np.random.seed(seed)
    values = (ub-lb)*np.random.rand(num_change, signal_dimension) + lb
    repeat = _sample_repeats(num_change, num_steps,
                             mean_change, sigma_change)
    return np.repeat(values, repeat, axis=0)

def get_tworeac_train_val_data(*, Np, parameters, data_list):
    """ Get the data for training in appropriate format. """
    tsteps_steady = parameters['tsteps_steady']
    (Ny, Nu) = (parameters['Ny'], parameters['Nu'])
    (inputs, xGz0, outputs) = ([], [], [])
    for data in data_list:
        t = tsteps_steady
        
        # Get input trajectory.
        u_traj = data.u[t:][np.newaxis, :, np.newaxis]
        
        # Get initial state.
        x0 = data.y[t, :][np.newaxis, :]
        yp0seq = data.y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = data.u[t-Np:t][np.newaxis, :]
        xGz0_traj = np.concatenate((x0, yp0seq, up0seq), axis=-1)

        # Get output trajectory.       
        y_traj = data.y[t:, :][np.newaxis, ...]
        
        # Collect the trajectories in list.
        inputs.append(u_traj)
        xGz0.append(xGz0_traj)
        outputs.append(y_traj)
    
    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      xGz0=np.concatenate(xGz0[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], xGz0=xGz0[-2],
                         outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], xGz0=xGz0[-1],
                    outputs=outputs[-1])
    # Return.
    return (train_data, trainval_data, val_data)

def plot_profit_curve(*, us, costs, colors, legends, 
                         figure_size=PAPER_FIGSIZE,
                         ylabel_xcoordinate=-0.12, 
                         left_label_frac=0.15):
    """ Plot the profit curves. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size, 
                                    gridspec_kw=dict(left=left_label_frac))
    xlabel = r'$C_{Af} \ (\textnormal{mol/m}^3)$'
    ylabel = r'Cost ($\$ $)'
    for (cost, color) in zip(costs, colors):
        # Plot the corresponding data.
        axes.plot(us, cost, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=False)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    axes.set_xlim([np.min(us), np.max(us)])
    figlabel = r'$\ell(y, u), \ \textnormal{subject to} \ f(x, u)=0, y=h(x)$'
    figure.suptitle(figlabel,
                    x=0.55, y=0.94)
    # Return the figure object.
    return [figure]

def _eigval_eigvec_test(X,Y):
    """Return True if an eigenvector of X corresponding to 
    an eigenvalue of magnitude greater than or equal to 1
    is not in the nullspace of Y.
    Else Return False."""
    (eigvals, eigvecs) = np.linalg.eig(X)
    eigvecs = eigvecs[:, np.absolute(eigvals)>=1.]
    for eigvec in eigvecs.T:
        if np.linalg.norm(Y @ eigvec)<=1e-8:
            return False
    else:
        return True

def assert_detectable(A, C):
    """Assert if the provided (A, C) pair is detectable."""
    assert _eigval_eigvec_test(A, C)

def get_augmented_matrices_for_filter(A, B, C, Bd, Cd, Qwx, Qwd):
    """ Get the Augmented A, B, C, and the noise covariance matrix."""
    Nx = A.shape[0]
    Nu = B.shape[1]
    Nd = Bd.shape[1]
    # Augmented A.
    Aaug1 = np.concatenate((A, Bd), axis=1)
    Aaug2 = np.concatenate((np.zeros((Nd, Nx)), np.eye(Nd)), axis=1)
    Aaug = np.concatenate((Aaug1, Aaug2), axis=0)
    # Augmented B.
    Baug = np.concatenate((B, np.zeros((Nd, Nu))), axis=0)
    # Augmented C.
    Caug = np.concatenate((C, Cd), axis=1)
    # Augmented Noise Covariance.
    Qwaug = scipy.linalg.block_diag(Qwx, Qwd)
    # Check that the augmented model is detectable. 
    assert_detectable(Aaug, Caug)
    return (Aaug, Baug, Caug, Qwaug)

def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T
    return (L, P)
    
class KalmanFilter:

    def __init__(self, *, A, B, C, Qw, Rv, xprior):
        """ Class to construct and perform state estimation
        using Kalman Filtering.
        """

        # Store the matrices.
        self.A = A
        self.B = B 
        self.C = C
        self.Qw = Qw
        self.Rv = Rv
        
        # Compute the kalman filter gain.
        self._computeFilter()
        
        # Create lists for saving data. 
        self.xhat = [xprior]
        self.xhat_pred = []
        self.y = []
        self.uprev = []

    def _computeFilter(self):
        "Solve the DARE to compute the optimal L."
        (self.L, _) = dlqe(self.A, self.C, self.Qw, self.Rv) 

    def solve(self, y, uprev):
        """ Take a new measurement and do 
            the prediction and filtering steps."""
        xhat = self.xhat[-1]
        xhat_pred = self.A @ xhat + self.B @ uprev
        xhat = xhat_pred + self.L @ (y - self.C @ xhat_pred)
        # Save data.
        self._save_data(xhat, xhat_pred, y, uprev)
        return xhat
        
    def _save_data(self, xhat, xhat_pred, y, uprev):
        """ Save the state estimates,
            Can be used for plotting later."""
        self.xhat.append(xhat)
        self.xhat_pred.append(xhat_pred)
        self.y.append(y)
        self.uprev.append(uprev)