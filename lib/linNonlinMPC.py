# Pratyush Kumar, pratyushkumar@ucsb.edu

import sys
import numpy as np
import mpctools as mpc
import scipy.linalg
import casadi
import collections
import pickle
import plottools
import time
from hybridid import SimData
    
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

class RTOController:

    def __init__(self, *, fxu, hx, lyup, Nx, Nu, Np, 
                 ulb, uub, init_guess, init_opt_pars,
                 Ntsep_solve):
        """ Class to construct and solve steady state optimization
            problems. 
                
        Optimization problem:
        min_{xs, us} l(ys, us, p)
        subject to:
        xs = f(xs, us), ys = h(xs), ulb <= us <= uub
        """

        # Model.
        self.fxu = fxu
        self.hx = hx
        self.lyup = lyup

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Np = Np

        # Input constraints.
        self.ulb = ulb
        self.uub = uub

        # Inital guess/parameters.
        self.init_guess = init_guess
        self.init_opt_pars = init_opt_pars
        self.Ntsep_solve = Ntsep_solve

        # Setup the optimization problem.
        self._setup_ss_optimization()
         
    def _setup_ss_optimization(self):
        """ Setup the steady state optimization. """
        # Construct NLP and solve.
        xs = casadi.SX.sym('xs', self.Nx)
        us = casadi.SX.sym('us', self.Nu)
        p = casadi.SX.sym('us', self.Np)
        ys = self.hx(xs)
        plant_nlp = dict(x = casadi.vertcat(xs, us), 
                         f = self.lyup(ys, us, p), 
                         g = casadi.vertcat(plant(xs, us), us))
        plant_nlp = casadi.nlpsol('plant_nlp', 'ipopt', plant_nlp)
        xuguess = np.concatenate((init_guess['x'], 
                                  init_guess['u']))[:, np.newaxis]
        lbg = np.concatenate((np.zeros((self.Nx,)), 
                              self.ulb))[:, np.newaxis]
        ubg = np.concatenate((np.zeros((self.Nx,)), 
                              self.uub))[:, np.newaxis]
        plant_nlp_soln = plant_nlp(x0=xuguess, lbg=lbg, ubg=ubg)

        return

    def control_law(self, simt):
        """ RTO Controller, no feedback. """


        return

class NonlinearEMPCRegulator:

    def __init__(self, *, fxu, lxup, Nx, Nu, Np,
                 Nmpc, ulb, uub, init_guess, init_empc_pars):
        """ Class to construct and solve nonlinear MPC -- Regulation. 
        
        Problem setup:
        The current time is T, we have x.

        Optimization problem:
        min_{u[0:N-1]} sum_{k=0^k=N-1} l(x(k), u(k), p(k))
        subject to:
        x(k+1) = f(x(k), u(k)), k=0 to N-1, ulb <= u(k) <= uub
        """
        # Model.
        self.fxu = fxu
        self.lxup = lxup

        # Sizes.
        self.Nmpc = Nmpc
        self.Nx = Nx
        self.Nu = Nu
        self.Np = Np

        # Create lists for saving data. 
        self.x0 = []
        self.useq = []
        self.xseq = []

        # Initial guess and parameters.
        self.init_guess = init_guess
        self.init_empc_pars = init_empc_pars

        # Get the hard constraints on inputs and the soft constraints.
        self.ulb = ulb
        self.uub = uub

        # Build the nonlinear MPC regulator.
        self._setup_regulator()

    def _setup_regulator(self):
        """ Construct a Nonlinear economic MPC regulator. """
        N = dict(x=self.Nx, u=self.Nu, p=self.Np, t=self.Nmpc)
        funcargs = dict(f=["x", "u"], 
                        l=["x", "u", "p"])
        lxup = mpc.getCasadiFunc(self.lxup,
                                 [self.Nx, self.Nu, self.Np],
                                 funcargs["l"])
        # Some parameters for the regulator.
        empc_pars = self.init_empc_pars
        guess = self.init_guess
        if len(guess['x'].shape) == 2:
            x0 = guess['x'][0, :]
        else:
            x0 = guess['x']
        lb = dict(u=self.ulb)
        ub = dict(u=self.uub)
        self.regulator = mpc.nmpc(f=self.fxu, l=lxup, N=N, funcargs=funcargs,
                                  x0=x0, p=empc_pars, lb=lb, ub=ub,
                                  guess=guess)
        self.regulator.solve()
        self.regulator.saveguess()
        useq = np.asarray(casadi.horzcat(*self.regulator.var['u'])).T
        xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
        self._append_data(x0, useq, xseq)
    
    def solve(self, x0, empc_pars):
        """Setup and the solve the dense QP, output is
        the first element of the sequence.
        If the problem is reparametrized, go back to original
        input variable.
        """
        self.regulator.par["p"] = list(empc_pars)
        self.regulator.fixvar("x", 0, x0)
        self.regulator.solve()
        self.regulator.saveguess()
        useq = np.asarray(casadi.horzcat(*self.regulator.var['u'])).T
        xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
        self._append_data(x0, useq, xseq)
        return useq

    def _append_data(self, x0, useq, xseq):
        " Append data. "
        self.x0.append(x0)
        self.useq.append(useq)
        self.xseq.append(xseq)

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

class NonlinearEMPCController:
    """ Class that instantiates the NonlinearMHE, 
        NonlinearEMPCRegulator classes
        into one and solves nonlinear economic MPC problems.

        fxu is a continous time model.
        hx is same in both the continous and discrete time.
        Bd is in continuous time.
    """
    def __init__(self, *, fxu, hx, lyup, Bd, Cd,
                     Nx, Nu, Ny, Nd,
                     xs, us, ds, ys,
                     empc_pars, ulb, uub, Nmpc,
                     Qwx, Qwd, Rv, Nmhe):
        
        # Model and stage cost.
        self.fxu = fxu
        self.hx = hx
        self.lyup = lyup
        self.Bd = Bd
        self.Cd = Cd

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Nd = Nd
        self.Np = empc_pars.shape[1]

        # Steady states of the system.
        self.xs = xs
        self.us = us
        self.ds = ds
        self.ys = ys
        self.uprev = us

        # MPC Regulator parameters.
        self.empc_pars = empc_pars
        self.ulb = ulb
        self.uub = uub
        self.Nmpc = Nmpc
        self._setup_regulator()

        # MHE Parameters.
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.Nmhe = Nmhe
        self._setup_estimator()

        # Parameters to save.
        self.computation_times = []
        
    def _aug_ss_model(self):
        """Augmented state-space model for moving horizon estimation."""
        return lambda x, u: np.concatenate((self.fxu(x[0:self.Nx],
                                                u) + self.Bd @ x[-self.Nd:],
                                             x[-self.Nd:]))
    
    def _mhe_hx_model(self):
        """ Augmented measurement model for moving horizon estimation. """
        return lambda x : self.hx(x[0:self.Nx]) + self.Cd @ x[-self.Nd:]

    def _get_mhe_models_and_matrices(self):
        """ Get the models, proir estimates and data, and the penalty matrices to setup an MHE solver."""

        # Prior estimates and data.
        xprior = np.concatenate((self.xs, self.ds), axis=0)[:, np.newaxis]
        xprior = np.repeat(xprior.T, self.Nmhe, axis=0)
        u = np.repeat(self.us[np.newaxis, :], self.Nmhe, axis=0)
        y = np.repeat(self.ys[np.newaxis, :], self.Nmhe+1, axis=0)

        # Penalty matrices.
        Qwxinv = np.linalg.inv(self.Qwx)
        Qwdinv = np.linalg.inv(self.Qwd)
        Qwinv = scipy.linalg.block_diag(Qwxinv, Qwdinv)
        P0inv = Qwinv
        Rvinv = np.linalg.inv(self.Rv)

        # Get the augmented models.
        fxud = mpc.getCasadiFunc(self._aug_ss_model(),
                                [self.Nx + self.Nd, self.Nu], ["x", "u"])
        hx = mpc.getCasadiFunc(self._mhe_hx_model(),
                                [self.Nx + self.Nd], ["x"])
        # Return the required quantities for MHE.
        return (fxud, hx, P0inv, Qwinv, Rvinv, xprior, u, y)

    def _setup_estimator(self):
        """ Setup MHE. """
        (fxud, hx, P0inv,
         Qwinv, Rvinv,
         xprior, u, y) = self._get_mhe_models_and_matrices()
        self.estimator = NonlinearMHEEstimator(fxu=fxud, hx=hx, 
                                     Nmhe=self.Nmhe, 
                                     Nx=self.Nx+self.Nd, Nu=self.Nu, Ny=self.Ny,
                                     xprior=xprior, u=u, y=y, P0inv=P0inv, Qwinv=Qwinv, Rvinv=Rvinv)

    def _setup_regulator(self):
        """ Augment the system for rate of change penalty and 
        build the regulator. """
        fxud = mpc.getCasadiFunc(self._aug_ss_model(),
                                [self.Nx + self.Nd, self.Nu], ["x", "u"])
        init_guess = dict(x=np.concatenate((self.xs, self.ds)), u=self.us)
        init_empc_pars = self.empc_pars[0:self.Nmpc, :]
        lxup = lambda x, u, p: self.lyup(self.hx(x), u, p)
        self.regulator  = NonlinearEMPCRegulator(fxu=fxud, lxup=lxup,
                                                 Nx=self.Nx + self.Nd,
                                                 Nu=self.Nu, Np=self.Np,
                                                 Nmpc=self.Nmpc,
                                                 ulb=self.ulb, uub=self.uub,
                                                 init_guess=init_guess,
                                                 init_empc_pars=init_empc_pars)

    def control_law(self, simt, y):
        """
        Takes the measurement and the previous control input
        and compute the current control input.
        """
        tstart = time.time()
        xdhat =  self.estimator.solve(y, self.uprev)
        mpc_N = slice(simt, simt + self.Nmpc, 1)
        empc_pars = self.empc_pars[mpc_N, :]
        useq = self.regulator.solve(xdhat, empc_pars)
        self.uprev = useq[:1, :].T
        tend = time.time()
        self.computation_times.append(tend - tstart)
        return self.uprev

def online_simulation(plant, controller, *, plant_lyup, Nsim=None,
                      disturbances=None, stdout_filename=None):
    """ Online simulation with either the RTO controller
        or the nonlinear economic MPC controller. """

    sys.stdout = open(stdout_filename, 'w')
    measurement = plant.y[0] # Get the latest plant measurement.
    disturbances = disturbances[..., np.newaxis]
    avg_stage_costs = [0.]

    # Start simulation loop.
    for (simt, disturbance) in zip(range(Nsim), disturbances):

        # Compute the control and the current stage cost.
        print("Simulation Step:" + f"{simt}")
        control_input = controller.control_law(simt, measurement)
        print("Computation time:" + str(controller.computation_times[-1]))
        stage_cost = plant_lyup(plant.y[-1], control_input,
                                controller.empc_pars[simt:simt+1, :].T)[0]
        avg_stage_costs += [(avg_stage_costs[-1]*simt + stage_cost)/(simt+1)]

        # Inject control/disturbance to the plant.
        measurement = plant.step(control_input, disturbance)

    # Create a sim data/stage cost array.
    cl_data = SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u),
                y=np.asarray(plant.y[0:-1]).squeeze())
    avg_stage_costs = np.array(avg_stage_costs[1:])

    # Return.
    return cl_data, avg_stage_costs

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