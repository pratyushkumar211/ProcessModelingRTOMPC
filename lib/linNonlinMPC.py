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

class NonlinearPlantSimulator:
    """Custom class for simulating non-linear plants."""
    def __init__(self, *, fxup, hx, Rv, Nx, Nu, Np, Ny,
                 sample_time, x0):
        
        # Set attributes.
        self.Nx, self.Nu, self.Ny, self.Np = Nx, Nu, Ny, Np
        self.fxup = mpc.getCasadiFunc(fxup, [Nx, Nu, Np], 
                                      ['x', 'u', 'p'], 'fxup', 
                                      rk4=True, Delta=sample_time, M=1)
        self.hx = mpc.getCasadiFunc(hx, [Nx], ["x"], funcname="hx")
        self.ynoise_std = np.sqrt(np.diag(Rv)[:, np.newaxis])
        self.sample_time = sample_time

        # Create lists to save data.
        self.x = [x0]
        self.u = []
        self.p = []
        self.y = [np.asarray(self.hx(x0)) + 
                  self.ynoise_std*np.random.randn(self.Ny, 1)]
        self.t = [0.]

    def step(self, u, p):
        """ Inject the control input into the plant."""
        x = np.asarray(self.fxup(self.x[-1], u, p))
        y = np.asarray(self.hx(x))
        y = y + self.ynoise_std*np.random.randn(self.Ny, 1)
        self._appendData(x, u, p, y)
        return y

    def _appendData(self, x, u, p, y):
        """ Append the data into the lists.
            Used for plotting in the specific subclasses.
        """
        self.x.append(x)
        self.u.append(u)
        self.p.append(p)
        self.y.append(y)
        self.t.append(self.t[-1]+self.sample_time)

# class RTOController:

#     def __init__(self, *, fxu, hx, lyup, Nx, Nu, Np, 
#                  ulb, uub, init_guess, opt_pars, Ntstep_solve):
#         """ Class to construct and solve steady state optimization
#             problems.
                
#         Optimization problem:
#         min_{xs, us} l(ys, us, p)
#         subject to:
#         xs = f(xs, us), ys = h(xs), ulb <= us <= uub
#         """

#         # Model.
#         self.fxu = fxu
#         self.hx = hx
#         self.lyup = lyup

#         # Sizes.
#         self.Nx = Nx
#         self.Nu = Nu
#         self.Np = Np

#         # Input constraints.
#         self.ulb = ulb
#         self.uub = uub

#         # Inital guess/parameters.
#         self.init_guess = init_guess
#         self.opt_pars = opt_pars
#         self.Ntstep_solve = Ntstep_solve

#         # Setup the optimization problem.
#         self._setup_ss_optimization()

#         # Lists to save data.
#         self.xs = []
#         self.us = []
#         self.computation_times = []

#     def _setup_ss_optimization(self):
#         """ Setup the steady state optimization. """
#         # Construct NLP and solve.
#         xs = casadi.SX.sym('xs', self.Nx)
#         us = casadi.SX.sym('us', self.Nu)
#         p = casadi.SX.sym('p', self.Np)
#         lyup = lambda x, u, p: self.lyup(self.hx(x), u, p)
#         lyup = mpc.getCasadiFunc(lyup,
#                           [self.Nx, self.Nu, self.Np],
#                           ["x", "u", "p"])
#         fxu = mpc.getCasadiFunc(self.fxu,
#                           [self.Nx, self.Nu],
#                           ["x", "u"])
#         nlp = dict(x=casadi.vertcat(xs, us), 
#                    f=lyup(xs, us, p),
#                    g=casadi.vertcat(xs -  fxu(xs, us), us), 
#                    p=p)
#         self.nlp = casadi.nlpsol('nlp', 'ipopt', nlp)
#         xuguess = np.concatenate((self.init_guess['x'], 
#                                   self.init_guess['u']))[:, np.newaxis]
#         self.lbg = np.concatenate((np.zeros((self.Nx,)), 
#                                    self.ulb))[:, np.newaxis]
#         self.ubg = np.concatenate((np.zeros((self.Nx,)), 
#                                    self.uub))[:, np.newaxis]
#         nlp_soln = self.nlp(x0=xuguess, lbg=self.lbg, ubg=self.ubg, 
#                             p=self.opt_pars[0:1, :])
#         self.xuguess = np.asarray(nlp_soln['x'])        

#     def control_law(self, simt, y):
#         """ RTO Controller, no use of feedback. 
#             Only solve every certain interval. """

#         tstart = time.time()
#         if simt%self.Ntstep_solve == 0:
#             nlp_soln = self.nlp(x0=self.xuguess, lbg=self.lbg, ubg=self.ubg, 
#                                 p=self.opt_pars[simt:simt+1, :])
#             self.xuguess = np.asarray(nlp_soln['x'])
#         xs, us = np.split(self.xuguess, [self.Nx,], axis=0)
#         tend = time.time()
#         self._append_data(xs, us)
#         self.computation_times.append(tend-tstart)
#         # Return the steady input.
#         return us

#     def _append_data(self, xs, us):
#         " Append data. "
#         self.xs.append(xs)
#         self.us.append(us)

class NonlinearEMPCRegulator:

    def __init__(self, *, fxu, lxup, Nx, Nu, Np,
                 Nmpc, ulb, uub, t0Guess, t0EmpPars):
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
        self.t0guess = t0Guess
        self.t0EmpcPars = t0EmpcPars

        # Get the hard constraints on inputs and the soft constraints.
        self.ulb = ulb
        self.uub = uub

        # Build the nonlinear MPC regulator.
        self._setupRegulator()

    def _setupRegulator(self):
        """ Construct a Nonlinear economic MPC regulator. """

        N = dict(x=self.Nx, u=self.Nu, p=self.Np, t=self.Nmpc)
        
        funcargs = dict(f=["x", "u"], l=["x", "u", "p"])
        
        # Some parameters for the regulator.
        empcPars = self.t0EmpcPars
        guess = self.t0Guess
        x0 = guess['x']
        lb = dict(u=self.ulb)
        ub = dict(u=self.uub)
        
        # Construct the EMPC regulator.
        self.regulator = mpc.nmpc(f=self.fxu, l=self.lxup, N=N, 
                                  funcargs=funcargs, x0=x0, p=empcPars, lb=lb, 
                                  ub=ub, guess=guess)
        self.regulator.solve()
        self.regulator.saveguess()

        # Get the x and u sequences and save data.
        useq = np.asarray(casadi.horzcat(*self.regulator.var['u'])).T
        xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
        self._append_data(x0, useq, xseq)
    
    def solve(self, x0, empcPars):
        """Setup and the solve the dense QP, output is
        the first element of the sequence.
        If the problem is reparametrized, go back to original
        input variable.
        """
        self.regulator.par["p"] = list(empcPars)
        self.regulator.fixvar("x", 0, x0)
        self.regulator.solve()
        self.regulator.saveguess()

        # Get the x and u sequences and save data.
        useq = np.asarray(casadi.horzcat(*self.regulator.var['u'])).T
        xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
        self._appendData(x0, useq, xseq)
        return useq

    def _appendData(self, x0, useq, xseq):
        " Append data. "
        self.x0.append(x0)
        self.useq.append(useq)
        self.xseq.append(xseq)

class NonlinearMHEEstimator:

    def __init__(self, *, fxu, hx, Nmhe, Nx, Nu, Ny,
                 xhatPast, uPast, yPast, P0inv, Qwinv, Rvinv):
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
        self.xhat = list(xhatPast)
        self.u = list(uPast)
        self.y = list(yPast)
        
        # Build the estimator.
        self._setupMheEstimator()
        
    def _setupMheEstimator(self):
        """ Construct a MHE solver. """

        N = dict(x=self.Nx, u=self.Nu, y=self.Ny, t=self.N)
        funcargs = dict(f=["x", "u"], h=["x"], l=["w", "v"], lx=["x", "x0bar"])
        
        # Setup stage costs.
        l = mpc.getCasadiFunc(self._stageCost, [N["x"], N["y"]],
                              funcargs["l"])
        lx = mpc.getCasadiFunc(self._priorCost, [N["x"], N["x"]],
                               funcargs["lx"])

        # Get Guess for the NLP.
        guess = dict(x=self.xhat[-1], w=np.zeros((self.Nx,)),
                     v=np.zeros((self.Ny,)))

        # Setup the MHE estimator.        
        self.mheEstimator = mpc.nmhe(f=self.fxu,
                                     h=self.hx, wAdditive=True,
                                     N=N, l=l, lx=lx, u=self.u, y=self.y,
                                     funcargs=funcargs,
                                     guess=guess,
                                     x0bar=self.xhat[0],
                                     verbosity=0)
        self.mhe_estimator.solve()
        self.mhe_estimator.saveguess()

        # Get estimated state sequence.
        xhat = np.asarray(self.mhe_estimator.var["x"][-1]).squeeze(axis=-1)
        self.xhat.append(xhat)

    def _stageCost(self, w, v):
        """ Stage cost in moving horizon estimation. """
        return mpc.mtimes(w.T, self.Qwinv, w) + mpc.mtimes(v.T, self.Rvinv, v)

    def _priorCost(self, x, xprior):
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

        # Assemble data and solve NLP.
        N = self.N
        self.mhe_estimator.par["x0bar"] = [self.xhat[-N]]
        self.mhe_estimator.par["y"] = self.y[-N:] + [y]
        self.mhe_estimator.par["u"] = self.u[-N+1:] + [uprev]
        self.mhe_estimator.solve()
        self.mhe_estimator.saveguess()

        # Get estimated state sequence and save data.
        xhat = np.asarray(self.mhe_estimator.var["x"][-1]).squeeze(axis=-1)
        self._appendData(xhat, y, uprev)
        return xhat

    def _appendData(self, xhat, y, uprev):
        """ Append the data to the lists. """
        self.xhat.append(xhat)
        self.y.append(y)
        self.u.append(uprev)

class NonlinearEMPCController:
    """ Class that instantiates a Kalman Filter, 
        Target-selector, Tracking MPC regulator, and 
        a steady-state nonlinear optimizer classes
        into one and runs in real-time.

        fxu is a continous time model.
        hx is same in both the continous and discrete time.
        Bd is in continuous time.
    """
    def __init__(self, *, fxu, hx, lyup, Bd, Cd,
                     Nx, Nu, Ny, Nd,
                     xs, us, ds, ys,
                     empc_pars, ulb, uub, Nmpc,
                     Qwx, Qwd, Rv, Nmhe, 
                     guess = None):
        
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
        self.opt_pars = empc_pars
        self.ulb = ulb
        self.uub = uub
        self.Nmpc = Nmpc
        self.guess = guess
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
        if self.guess is None:
            init_guess = dict(x=np.concatenate((self.xs, self.ds)), u=self.us)
        else:
            xguess = [np.concatenate((self.xs, self.ds))]
            uguess = self.guess['u']
            for t in range(uguess.shape[0]):
                xguess_tplus = np.asarray(fxud(xguess[-1], uguess[t, :]))[:, 0]
                xguess.append(xguess_tplus)
            xguess = np.asarray(xguess)
            init_guess = dict(x=xguess, u=uguess)
        init_empc_pars = self.opt_pars[0:self.Nmpc, :]
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
        empc_pars = self.opt_pars[mpc_N, :]
        useq = self.regulator.solve(xdhat, empc_pars)
        self.uprev = useq[:1, :].T
        tend = time.time()
        self.computation_times.append(tend - tstart)
        return self.uprev

class TwoTierRTOMPController:
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
                     Qwx, Qwd, Rv, Nmhe, 
                     guess = None):
        
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
        self.opt_pars = empc_pars
        self.ulb = ulb
        self.uub = uub
        self.Nmpc = Nmpc
        self.guess = guess
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
        if self.guess is None:
            init_guess = dict(x=np.concatenate((self.xs, self.ds)), u=self.us)
        else:
            xguess = [np.concatenate((self.xs, self.ds))]
            uguess = self.guess['u']
            for t in range(uguess.shape[0]):
                xguess_tplus = np.asarray(fxud(xguess[-1], uguess[t, :]))[:, 0]
                xguess.append(xguess_tplus)
            xguess = np.asarray(xguess)
            init_guess = dict(x=xguess, u=uguess)
        init_empc_pars = self.opt_pars[0:self.Nmpc, :]
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
        empc_pars = self.opt_pars[mpc_N, :]
        useq = self.regulator.solve(xdhat, empc_pars)
        self.uprev = useq[:1, :].T
        tend = time.time()
        self.computation_times.append(tend - tstart)
        return self.uprev
