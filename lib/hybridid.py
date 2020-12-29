# Pratyush Kumar, pratyushkumar@ucsb.edu

import sys
import numpy as np
import mpctools as mpc
import scipy.linalg
import matplotlib.pyplot as plt
import casadi
import collections
import pickle
import plottools

FIGURE_SIZE_A4 = (9, 10)
PRESENTATION_FIGSIZE = (6, 6)
PAPER_FIGSIZE = (5, 6)

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

class PIController:
    """ PI controller class."""
    def __init__(self, *, K, tau, us, sample_time, 
                 ulb, uub, pitype, taus=None):
        (self.K, self.tau) = (K, tau)
        (self.ulb, self.uub, self.us) = (ulb, uub, us)
        self.sample_time = sample_time
        self.i = [np.array([[0.]])]
        if pitype == "ideal":
            self.control_law = self.idealpi_control_law
        elif pitype == "antiwindup":
            self.control_law = self.piaw_control_law
        elif pitype == "foxboro":
            self.control_law = self.foxpi_control_law
        elif pitype == "astrom":
            self.taus = taus
            self.control_law = self.astrompi_control_law

    def idealpi_control_law(self, y, ysp):
        e = ysp - y
        u = self.us + self.K*e + (self.K/self.tau)*self.i[-1]
        self.i.append(self.i[-1] + e*self.sample_time)
        return np.clip(u, self.ulb, self.uub)

    def piaw_control_law(self, y, ysp):
        e = ysp - y
        u = self.us + self.K*e + (self.K/self.tau)*self.i[-1]
        if u > self.ulb and u < self.uub:
            self.i.append(self.i[-1] + e*self.sample_time)
            return u
        else:
            self.i.append(self.i[-1])
            return np.clip(u, self.ulb, self.uub)

    def foxpi_control_law(self, y, ysp):
        e = ysp - y
        u = self.K*e + (self.K/self.tau)*self.i[-1]
        usat = np.clip(u, self.ulb - self.us, self.uub - self.us)
        self.i.append(self.i[-1] + 
                      (-self.i[-1]/self.tau + usat/self.K)*self.sample_time)
        return usat + self.us

    def astrompi_control_law(self, y, ysp):
        e = ysp - y
        u = self.K*e + (self.K/self.tau)*self.i[-1]
        usat = np.clip(u, self.ulb - self.us, self.uub - self.us)
        es = usat - u
        self.i.append(self.i[-1] + 
                      self.sample_time*(e + (self.tau/(self.K*self.taus))*es))
        return usat + self.us

class RTOPIController:

    def __init__(self):



    def control_law(self):

        return

class NonlinearEMPCController:

    def __init__(self, *, Akf, Bkf, Bpkf, Rv,
                 Ampc, Bmpc, Bpmpc, C, Nhours_forecast,
                 K, tau, uspi, pitype, xp0, xi0,
                 Qcool_bounds, Tzone_bounds, 
                 roc_penalty, slack_penalty,
                 plantDelta, mpcDelta, 
                 energy_prices, disturbances, alpha):

        # Set some default values.
        (self.Akf, self.Bkf, self.Bpkf) = (Akf, Bkf, Bpkf) 
        (self.C, self.Rv) = (C, Rv)
        (self.Ampc, self.Bmpc, self.Bpmpc) = (Ampc, Bmpc, Bpmpc)
        self.N = int(Nhours_forecast*3600/mpcDelta)
        self.roc_penalty = roc_penalty
        self.slack_penalty = slack_penalty
        self.alpha = alpha
        (self.K, self.tau, self.uspi, self.pitype) = (K, tau, uspi, pitype) 
        (self.Qcool_lb, self.Qcool_ub) = Qcool_bounds
        (self.Tlb, self.Tub) = Tzone_bounds
        (self.Nx, self.Nu, self.Ny, self.Np, self.Ns) = (3, 1, 1, 3, 4)
        (self.xp0, self.xi0) = (xp0, xi0)
        self.mpcparameters = np.concatenate((energy_prices, 
                                             disturbances), axis=1)

        # Get the sample times of the two controllers.
        self.plantDelta = plantDelta
        self.mpcDelta = mpcDelta
        self.Delta_ratio = int(mpcDelta/plantDelta)

        # Create lists to save and examine open-loop solutions.
        self.xseqs = []
        self.useqs = []
        self.eseqs = []
        self.upiseqs = []

        # Setup the filter and the regulator.
        self._setup_filter()
        self._setup_regulator()

    def _setup_filter(self):
        """ Setup the Kalman filter."""
        Bkf = np.concatenate((self.Bkf, self.Bpkf), axis=1)
        Qw = (self.Bkf @ self.Bkf.T)*(self.Rv**2)
        Qw = Qw + 1e-8*np.eye(2)
        xprior = self.xp0
        self.filter = KalmanFilter(A=self.Akf, B=Bkf, 
                                   C=self.C, Qw=Qw, 
                                   Rv=self.Rv, xprior=xprior)

    def _get_state_estimate(self, past_y, past_upi, past_p):
        """ Reconcile the past PI control inputs, disturbances 
            and get a state estimate of the plant. """
        for (y, upi, p) in zip(past_y, past_upi, past_p):
            xhatp = self.filter.solve(y, np.concatenate((upi, p), axis=0))
        return xhatp    

    def control_law(self, simt, xi, past_y, past_upi, past_p):
        """ Get the MPC control law. """
        xhatp = self._get_state_estimate(past_y, past_upi, past_p)
        xhat = np.concatenate((xhatp, xi), axis=0)
        u = self._solve_empc(xhat, simt)
        return np.tile(u, (self.Delta_ratio, 1))

    def _solve_empc(self, xhat, simt):
        """ Use the parameters and solve the nonlinear economic MPC problem."""
        # Get the MPC forecasting horizon here.
        mpc_N = slice(int((simt+1)*(1/self.Delta_ratio)), 
                      int((simt+1)*(1/self.Delta_ratio))+self.N, 1)
        self.regulator.par["p"] = list(self.mpcparameters[mpc_N, :])
        self.regulator.par['u_prev'] = self.uprev
        self.regulator.fixvar("x", 0, xhat)
        self.regulator.solve()
        self.regulator.saveguess()
        self._save_open_loop_solution()
        self.uprev = np.asarray(self.regulator.var["u"][0])
        return self.uprev

    def _setup_regulator(self):
        """ Construct a MHE solver. """
        N = dict(x=self.Nx, u=self.Nu, p=self.Np,
                 t=self.N, e=self.Ns, s=self.Ns)
        funcargs = dict(f=["x", "u", "p"], 
                        l=["x", "u", "Du", "p", "s"],
                        e=["x", "u", "s"])
        fxup = mpc.getCasadiFunc(lambda x, u, p: 
                                 self._fxup(x, u, p, self.pitype), 
                                 [self.Nx, self.Nu, self.Np], 
                                 funcargs['f'])
        l = mpc.getCasadiFunc(self._stage_cost, 
                              [self.Nx, self.Nu, 
                               self.Nu, self.Np, self.Ns],
                              funcargs["l"])
        e = mpc.getCasadiFunc(self._temp_cool_bounds, 
                              [self.Nx, self.Nu, self.Ns],
                              funcargs["e"])
        # Setup the regulator and solve.
        x0 = np.concatenate((self.xp0, self.xi0), axis=0)[:, 0]
        mpcparameters = self.mpcparameters[0:self.N, :]
        uprev = self.xp0[0:1]
        self.regulator = mpc.nmpc(f=fxup, l=l, e=e,
                                  N=N, funcargs=funcargs, 
                                  x0=x0,
                                  p=mpcparameters, 
                                  uprev=uprev)
        self.regulator.solve()
        self.uprev = np.asarray(self.regulator.var["u"][0])
        self.regulator.saveguess()
        self._save_open_loop_solution()

    def _save_open_loop_solution(self):
        """ Save open solutions from the regulator to examine later. """
        xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
        useq = np.asarray(self.regulator.var["u"])
        eseq = useq - xseq[:-1, 0]
        upiseq = self.K*eseq + (self.K/self.tau)*xseq[:-1, 2]
        self.xseqs.append(xseq)
        self.useqs.append(useq)
        self.eseqs.append(eseq)
        self.upiseqs.append(upiseq)

    def _stage_cost(self, x, u, Du, p, s):
        """ Compute the PI control input, 
            economic objective, and the rate of penalty. """
        (xzone, xi) = (x[0:1], x[2:])
        upi = self.K*(u-xzone) + (self.K/self.tau)*xi
        stagecost = p[0:1]*upi + self.roc_penalty*mpc.mtimes(Du.T, Du)
        stagecost += mpc.mtimes(self.slack_penalty, s)
        return stagecost
    
    def _temp_cool_bounds(self, x, u, s):
        """ Compute the PI control input 
            economic objective, and the rate of penalty."""
        (xzone, xi) = (x[0:1], x[2:])
        e1 = -xzone + self.Tlb -s[0:1] # temperature lower bound.
        e2 = xzone - self.Tub -s[1:2] # temperature upper bound.
        upi = self.K*(u-xzone) + (self.K/self.tau)*xi
        e3 = -upi + self.Qcool_lb -s[2:3] # Cooling duty lower bound.
        e4 =  upi - self.Qcool_ub -s[3:4] # Cooling duty upper bound.
        return mpc.vcat([e1, e2, e3, e4])

    def _fdelta(self, *, u, ulb, uub, alpha):
        """ Return an expanded differentiable
        dirac delta function. """
        u = 1./(1+casadi.exp(-alpha*(uub-u)))
        u = u + 1./(1+casadi.exp(-alpha*(u-ulb))) - 1
        return u

    def _fsat(self, *, u, ulb, uub, alpha):
        """ Approximation to the saturation function. """
        u = 2*(u-ulb)/(uub-ulb) - 1
        gsatu = u/casadi.power(1+casadi.power(u, alpha), 1/alpha)
        return (uub-ulb)*(gsatu+1)/2 + ulb

    def _fxup(self, x, u, p, pitype):
        """ The augmented time evolution function for economic MPC."""
        (xp, xi, pamb) = (x[0:2], x[2:], p[1:3])
        # Compute the PI control input and the increment in the 
        # integral mode.
        upi = self.K*(u-xp[0:1]) + (self.K/self.tau)*xi
        if pitype == "ideal":
            fi = u - xp[0:1]
        if pitype == "antiwindup":
            fi = u - xp[0:1]
            fi *= self._fdelta(u=upi, ulb=self.Qcool_lb-self.uspi, 
                               uub=self.Qcool_ub-self.uspi, 
                               alpha=self.alpha)
        if pitype == "foxboro":
            fi = -xi/self.tau
            fi += self._fsat(u=upi, ulb=self.Qcool_lb-self.uspi, 
                             uub=self.Qcool_ub-self.uspi, 
                             alpha=self.alpha)/self.K
        upi += self.uspi
        xp = mpc.mtimes(self.Ampc, xp) + mpc.mtimes(self.Bmpc, upi) 
        xp += mpc.mtimes(self.Bpmpc, pamb)
        xi += self.mpcDelta*fi
        return mpc.vcat([xp, xi])
        
def online_simulation():
    """ Online simulation with either the RTO-PI controller
        or nonlinear economic MPC controller. """


    return

class NonlinearMHEEstimator:

    def __init__(self, *, fxu, hx, Nmhe, Nx, Nu, Ny,
                 xprior, u, y, P0inv, Qwinv, Rvinv):
        """ Class to construct and perform state estimation
        using moving horizon estimation.
        
        Problem setup:
        The current time is T
        Measurements available: u_[T-N:T-1], y_[T-N:T]

        Optimization problem:
        min_{x_[T-N:T]} |x(T-N)-xprior|_{P0inv} + 
                        sum{t=T-N to t=N-1} |x(k+1)-x(k)|_{Qwinv} + 
                        sum{t=T-N to t=T} |y(k)-h(x(k))|_{Rvinv}

        subject to: x(k+1) = f(x(k), u(k), w(k)), 
                    y(k) = h(x) + v(k), k=T-N to T-1
        x is the augmented state.

        xprior is an array of previous smoothed estimates, xhat(k:k) from -T:-1
        y: -T:0
        u: -T:-1
        The constructor solves and gets a smoothed estimate of x at time 0.
        """
        self.fxu = fxu
        self.hx = hx

        # Penalty matrices.
        self.P0inv = P0inv
        self.Qwinv = Qwinv
        self.Rvinv = Rvinv

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.N = Nmhe

        # Create lists for saving data. 
        self.xhat = list(xprior)
        self.y = list(y)
        self.u = list(u)
        
        # Build the estimator.
        self._setup_mhe_estimator()

    def _setup_mhe_estimator(self):
        """ Construct a MHE solver. """
        N = dict(x=self.Nx, u=self.Nu, y=self.Ny, t=self.N)
        funcargs = dict(f=["x", "u"], h=["x"], l=["w", "v"], lx=["x", "x0bar"])
        l = mpc.getCasadiFunc(self._stage_cost, [N["x"], N["y"]],
                              funcargs["l"])
        lx = mpc.getCasadiFunc(self._prior_cost, [N["x"], N["x"]],
                               funcargs["lx"])
        guess = dict(x=self.xhat[-1], w=np.zeros((self.Nx,)),
                     v=np.zeros((self.Ny,)))
        self.mhe_estimator = mpc.nmhe(f=self.fxu,
                                      h=self.hx, wAdditive=True,
                                      N=N, l=l, lx=lx, u=self.u, y=self.y,
                                      funcargs=funcargs,
                                      guess=guess,
                                      x0bar=self.xhat[0],
                                      verbosity=0)
        self.mhe_estimator.solve()
        self.mhe_estimator.saveguess()
        xhat = np.asarray(self.mhe_estimator.var["x"][-1]).squeeze(axis=-1)
        self.xhat.append(xhat)

    def _stage_cost(self, w, v):
        """ Stage cost in moving horizon estimation. """
        return mpc.mtimes(w.T, self.Qwinv, w) + mpc.mtimes(v.T, self.Rvinv, v)

    def _prior_cost(self, x, xprior):
        """Prior cost in moving horizon estimation."""
        dx = x - xprior
        return mpc.mtimes(dx.T, self.P0inv, dx)
        
    def solve(self, y, uprev):
        """ Use the new data, solve the NLP, and store data.
            At this time:
            xhat: list of length T+1
            y: list of length T+1
            uprev: list of length T
        """
        N = self.N
        self.mhe_estimator.par["x0bar"] = [self.xhat[-N]]
        self.mhe_estimator.par["y"] = self.y[-N:] + [y]        
        self.mhe_estimator.par["u"] = self.u[-N+1:] + [uprev]
        self.mhe_estimator.solve()
        self.mhe_estimator.saveguess()
        xhat = np.asarray(self.mhe_estimator.var["x"][-1]).squeeze(axis=-1)
        self._append_data(xhat, y, uprev)
        return xhat

    def _append_data(self, xhat, y, uprev):
        """ Append the data to the lists. """
        self.xhat.append(xhat)
        self.y.append(y)
        self.u.append(uprev)

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

def get_scaling(*, data):
    """ Scale the input/output. """
    xscale = 0.5*(np.max(data.x, axis=0) - np.min(data.x, axis=0))
    uscale = 0.5*(np.max(data.u, axis=0) - np.min(data.u, axis=0))
    yscale = 0.5*(np.max(data.y, axis=0) - np.min(data.y, axis=0))
    # Return.
    return dict(xscale=xscale, uscale=uscale, yscale=yscale)

def get_cstr_flash_train_val_data(*, Np, parameters,
                                     greybox_processed_data):
    """ Get the data for training in appropriate format. """
    tsteps_steady = parameters['tsteps_steady']
    (Ng, Ny, Nu) = (parameters['Ng'], parameters['Ny'], parameters['Nu'])
    xuyscales = get_scaling(data=greybox_processed_data[0])
    (inputs, xGz0, yz0, outputs) = ([], [], [], [])
    for data in greybox_processed_data:
        
        # Scale data.
        u = data.u/xuyscales['uscale']
        y = data.y/xuyscales['yscale']
        x = data.x/xuyscales['xscale']

        t = tsteps_steady
        # Get input trajectory.
        u_traj = u[t:, :][np.newaxis, ...]

        # Get initial state.
        yp0seq = y[t-Np:t, :].reshape(Np*Ny, )[np.newaxis, :]
        up0seq = u[t-Np:t:, ].reshape(Np*Nu, )[np.newaxis, :]
        z0 = np.concatenate((yp0seq, up0seq), axis=-1)
        xG0 = x[t, :][np.newaxis, :]
        y0 = y[t, :][np.newaxis, :]
        xGz0_traj = np.concatenate((xG0, z0), axis=-1)
        yz0_traj = np.concatenate((y0, z0), axis=1)

        # Get output trajectory.
        y_traj = y[t:, :][np.newaxis, ...]

        # Collect the trajectories in list.
        inputs.append(u_traj)
        xGz0.append(xGz0_traj)
        yz0.append(yz0_traj)
        outputs.append(y_traj)

    # Get the training and validation data for training in compact dicts.
    train_data = dict(inputs=np.concatenate(inputs[:-2], axis=0),
                      xGz0=np.concatenate(xGz0[:-2], axis=0),
                      yz0=np.concatenate(yz0[:-2], axis=0),
                      outputs=np.concatenate(outputs[:-2], axis=0))
    trainval_data = dict(inputs=inputs[-2], xGz0=xGz0[-2],
                         yz0=yz0[-2], outputs=outputs[-2])
    val_data = dict(inputs=inputs[-1], xGz0=xGz0[-1],
                    yz0=yz0[-1], outputs=outputs[-1])
    # Return.
    return (train_data, trainval_data, val_data, xuyscales)

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
    #figlabel = r'$\ell(y, u), \ \textnormal{subject to} \ f(x, u)=0, y=h(x)$'
    #figure.suptitle(figlabel,
    #                x=0.55, y=0.94)
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