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
import cvxopt as cvx

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
                 Nmpc, ulb, uub, t0Guess, t0EmpcPars):
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
        self.N = Nmpc
        self.Nx = Nx
        self.Nu = Nu
        self.Np = Np

        # Create lists for saving data. 
        self.x0 = []
        self.useq = []
        self.xseq = []

        # Initial guess and parameters.
        self.t0Guess = t0Guess
        self.t0EmpcPars = t0EmpcPars

        # Get the hard constraints on inputs and the soft constraints.
        self.ulb = ulb
        self.uub = uub

        # Build the nonlinear MPC regulator.
        self._setupRegulator()

    def _setupRegulator(self):
        """ Construct a Nonlinear economic MPC regulator. """

        N = dict(x=self.Nx, u=self.Nu, p=self.Np, t=self.N)
        
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
        self._saveData(x0, useq, xseq)
    
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
        self._saveData(x0, useq, xseq)
        return useq

    def _saveData(self, x0, useq, xseq):
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
        self.mheEstimator.solve()
        self.mheEstimator.saveguess()

        # Get estimated state sequence.
        xhat = np.asarray(self.mheEstimator.var["x"][-1]).squeeze(axis=-1)
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
        self.mheEstimator.par["x0bar"] = [self.xhat[-N]]
        self.mheEstimator.par["y"] = self.y[-N:] + [y]
        self.mheEstimator.par["u"] = self.u[-N+1:] + [uprev]
        self.mheEstimator.solve()
        self.mheEstimator.saveguess()

        # Get estimated state sequence and save data.
        xhat = np.asarray(self.mheEstimator.var["x"][-1]).squeeze(axis=-1)
        self._saveData(xhat, y, uprev)
        return xhat

    def _saveData(self, xhat, y, uprev):
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
                          Nx, Nu, Ny, Nd, xs, us, ds,
                          empcPars, ulb, uub, Nmpc,
                          Qwx, Qwd, Rv, Nmhe):
        
        # Model and stage cost.
        self.fxu = fxu
        self.hx = hx
        self.lyup = lyup

        # Disturbance model.
        self.Bd = Bd
        self.Cd = Cd

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Nd = Nd
        self.Np = empcPars.shape[1]

        # Steady states of the system.
        # Used for initial Guess/Filling MHE past window.
        self.xs = xs
        self.us = us
        self.ds = ds
        self.ys = hx(xs)
        self.uprev = us

        # MPC Regulator parameters.
        self.empcPars = empcPars
        self.ulb = ulb
        self.uub = uub
        self.Nmpc = Nmpc
        self._setupRegulator()

        # MHE Parameters.
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.Nmhe = Nmhe
        self._setupEstimator()

        # Parameters to save.
        self.computationTimes = []
        
    def _augFxudModel(self):
        """Augmented state-space model for moving horizon estimation."""

        Nx, Nd = self.Nx, self.Nd
        fxu, Bd = self.fxu, self.Bd
        
        # Return the Augmented function.
        return lambda x, u: np.concatenate((fxu(x[:Nx], u) + Bd @ x[-Nd:],
                                             x[-Nd:]))
    
    def _augHxdModel(self):
        """ Augmented measurement model for moving horizon estimation. """

        Nx, Nd = self.Nx, self.Nd
        hx, Cd = self.hx, self.Cd

        # Return the augmented measurement function.
        return lambda x : hx(x[:Nx]) + Cd @ x[-Nd:]

    def _setupEstimator(self):
        """ Setup MHE. """

        # Get some numbers. 
        xs, ds, us, ys = self.xs, self.ds, self.us, self.ys
        Nx, Nu, Nd, Ny, Nmhe = self.Nx, self.Nu, self.Nd, self.Ny, self.Nmhe
        Qwx, Qwd, Rv = self.Qwx, self.Qwd, self.Rv

        # Prior estimates and data.
        xhatPast = np.concatenate((xs, ds), axis=0)[:, np.newaxis]
        xhatPast = np.repeat(xhatPast.T, Nmhe, axis=0)
        uPast = np.repeat(us[np.newaxis, :], Nmhe, axis=0)
        yPast = np.repeat(ys[np.newaxis, :], Nmhe+1, axis=0)

        # Penalty matrices.
        Qwxinv = np.linalg.inv(Qwx)
        Qwdinv = np.linalg.inv(Qwd)
        Qwinv = scipy.linalg.block_diag(Qwxinv, Qwdinv)
        P0inv = Qwinv
        Rvinv = np.linalg.inv(Rv)

        # Get the augmented models.
        fxud = mpc.getCasadiFunc(self._augFxudModel(), [Nx + Nd, Nu], 
                                 ["x", "u"])
        hx = mpc.getCasadiFunc(self._augHxdModel(), [Nx + Nd], ["x"])

        # Construct the MHE estimator.
        self.estimator = NonlinearMHEEstimator(fxu=fxud, hx=hx,
                                     Nmhe=Nmhe, Nx=Nx+Nd, Nu=Nu, Ny=Ny,
                                     xhatPast=xhatPast, uPast=uPast, 
                                     yPast=yPast, P0inv=P0inv, Qwinv=Qwinv, Rvinv=Rvinv)

    def _setupRegulator(self):
        """ Augment the system for rate of change penalty and 
        build the regulator. """

        # Get some numbers.
        xs, ds, us, ys = self.xs, self.ds, self.us, self.ys
        Nx, Nu, Nd, Np, Nmpc = self.Nx, self.Nu, self.Nd, self.Np, self.Nmpc
        ulb, uub = self.ulb, self.uub

        # Get casadi funcs for the model and stage cost.
        fxud = mpc.getCasadiFunc(self._augFxudModel(), [Nx + Nd, Nu], 
                                 ["x", "u"])

        lxup = lambda x, u, p: self.lyup(self._augHxdModel()(x), u, p)
        lxup = mpc.getCasadiFunc(lxup, [Nx + Nd, Nu, Np], ["x", "u", "p"])

        # Get initial guess.
        t0Guess = dict(x=np.concatenate((xs, ds)), u=us)
        t0EmpcPars = self.empcPars[:Nmpc, :]
        
        # Construct the EMPC regulator.
        self.regulator  = NonlinearEMPCRegulator(fxu=fxud, lxup=lxup,
                                                 Nx=Nx+Nd, Nu=Nu, Np=Np,
                                                 Nmpc=Nmpc, ulb=ulb, 
                                                 uub=uub, t0Guess=t0Guess,
                                                 t0EmpcPars=t0EmpcPars)

    def control_law(self, simt, y):
        """
        Takes the measurement and the previous control input
        and compute the current control input.
        """
        tstart = time.time()

        # Get state estimate.
        xdhat =  self.estimator.solve(y, self.uprev)

        # Get EMPC pars.
        mpc_N = slice(simt, simt + self.Nmpc, 1)
        empcPars = self.empcPars[mpc_N, :]

        # Solve nonlinear regulator.
        useq = self.regulator.solve(xdhat, empcPars)
        self.uprev = useq[:1, :].T

        # Get compuatation time and save.
        tend = time.time()
        self.computationTimes.append(tend - tstart)

        # Return control input.
        return self.uprev

class TwoTierMPController:
    """ Class to instantiate Extended Kalman Filter,
        Linear MPC Regulator, and updates linear model/targets
        based on steady state optimums.
    """
    def __init__(self, *, fxu, hx, lyup, empcPars, tSsOptFreq,
                          Nx, Nu, Ny, xs, us, Q, R, S, ulb, uub, Nmpc,
                          xhatPrior, covxPrior, Qw, Rv):
        
        # Model and stage cost.
        self.fxu = fxu
        self.hx = hx
        self.lyup = lyup

        # Sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Np = empcPars.shape[1]

        # Setup lists to save data.
        self.xs, self.us = [], []
        self.x0, self.useq = [xs[:, np.newaxis]], []

        # Setup steady state optimizer.
        self.ssOptXuguess = np.concatenate((xs, us))[:, np.newaxis]
        self.empcPars = empcPars
        self.tSsOptFreq = tSsOptFreq
        self.ulb = ulb[:, np.newaxis]
        self.uub = uub[:, np.newaxis]
        self._setupSSOptimizer()

        # MPC Regulator parameters.
        self.Q = Q
        self.R = R
        self.S = S
        self.Nmpc = Nmpc
        self.uprev = us[:, np.newaxis]
        self._setupRegulator()

        # Setup extended Kalman filter.
        self.estimator = ExtendedKalmanFilter(fxu=fxu, hx=hx, Nx=Nx, Nu=Nu, 
                                              Ny=Ny, Qw=Qw, Rv=Rv, 
                                            xPrior=xhatPrior, PPrior=covxPrior)

        # Parameters to save.
        self.computationTimes = []

    def _setupSSOptimizer(self):
        """ Setup the steady state optimization problem. """

        # Extract sizes/functions.
        Nx, Nu, Np = self.Nx, self.Nu, self.Np
        fxu, hx, lyup = self.fxu, self.hx, self.lyup
        ulb, uub = self.ulb, self.uub

        # Setup casadi variables.
        xs = casadi.SX.sym('xs', Nx)
        us = casadi.SX.sym('us', Nu)
        p = casadi.SX.sym('p', Np)

        # Setup casadi functions.
        lxup = lambda x, u, p: lyup(hx(x), u, p)
        lxup = mpc.getCasadiFunc(lxup, [Nx, Nu, Np], ["x", "u", "p"])
        fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

        # Setup the NLP.
        nlp = dict(x=casadi.vertcat(xs, us), f=lxup(xs, us, p),
                   g=casadi.vertcat(xs -  fxu(xs, us), us), p=p)
        self.ssOptimizer = casadi.nlpsol('nlp', 'ipopt', nlp)

        # Get constraints. 
        self.SsOptlbg = np.concatenate((np.zeros((Nx, 1)), ulb))
        self.SsOptubg = np.concatenate((np.zeros((Nx, 1)), uub))

        # Solve NLP.
        nlpSoln = self.ssOptimizer(x0=self.ssOptXuguess, lbg=self.SsOptlbg, 
                                   ubg=self.SsOptubg, p=self.empcPars[:1, :])

        # Get solution.
        xopt = np.asarray(nlpSoln['x'])
        xs, us = xopt[:Nx, :], xopt[Nx:, :]
        self.xs += [xs]
        self.us += [us]

        # Set Guess for next time.
        self.ssOptXuguess = xopt

    def _getLinearizedModel(self, xs, us):
        """ Get the linearized model matrices. """

        # Get the linearized model matrices A, B.
        fxu, Nx, Nu = self.fxu, self.Nx, self.Nu
        fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])
        linModel = mpc.util.getLinearizedModel(fxu, [xs, us], ["A", "B"])
        A, B = linModel["A"], linModel["B"]

        # Return.
        return (A, B)

    def _setupRegulator(self):
        """ Setup the Dense QP regulator. """

        # First get the linear model.
        xs, us = self.xs[-1], self.us[-1]
        x0, uprev = self.x0[-1], self.uprev
        Nx, Nu, Nmpc = self.Nx, self.Nu, self.Nmpc
        A, B = self._getLinearizedModel(xs, us)

        # Get a few more parameters to setup the Dense QP.
        Q, R, S = self.Q, self.R, self.S
        A, B, Q, R, M = getAugMatricesForROCPenalty(A, B, Q, R, S)
        ulb, uub = self.ulb - us, self.uub - us

        # Setup the regulator.
        self.regulator  = DenseQPRegulator(A=A, B=B, Q=Q, R=R, M=M, N=Nmpc, 
                                           ulb=ulb, uub=uub)

        # Solve for the initial steady state.
        x0 = np.concatenate((x0-xs, uprev - us))
        useq = self.regulator.solve(x0)
        useq += np.tile(us, (Nmpc, 1))
        useq = np.reshape(useq, (Nmpc, Nu))
        self.useq += [useq]

    def control_law(self, simt, y):
        """
        Takes the measurement and the previous control input
        and compute the current control input.
        """
        tstart = time.time()
        tSsOptFreq = self.tSsOptFreq
        Nx, Nu, Nmpc = self.Nx, self.Nu, self.Nmpc

        # Solve steady-state optimization if needed.
        if simt % tSsOptFreq == 0:
            nlpSoln = self.ssOptimizer(x0=self.ssOptXuguess, lbg=self.SsOptlbg, 
                               ubg=self.SsOptubg, 
                               p=self.empcPars[simt:simt+1, :])
            xopt = np.asarray(nlpSoln['x'])
            xs, us = xopt[:Nx, :], xopt[Nx:, :]
            self.ssOptXuguess = xopt

            # Update linear model.
            A, B = self._getLinearizedModel(xs, us)
            self.regulator.ulb = self.ulb - us
            self.regulator.uub = self.uub - us
            Q, R, S = self.Q, self.R, self.S
            A, B, Q, R, M = getAugMatricesForROCPenalty(A, B, Q, R, S)
            self.regulator._updateModel(A, B)
        else:
            xs, us = self.xs[-1], self.us[-1]

        # State estimation.
        xhat =  self.estimator.solve(y, self.uprev)
        
        # Regulation.
        x0 = np.concatenate((xhat-xs, self.uprev - us))
        useq = self.regulator.solve(x0)
        useq += np.tile(us, (Nmpc, 1))
        useq = np.reshape(useq, (Nmpc, Nu))
        
        # Save Uprev.
        self.uprev = useq[:1, :].T

        # Get computation times.
        tend = time.time()
        self.computationTimes.append(tend - tstart)

        # Save data. 
        self._saveData(xs, us, xhat, useq)

        # Return.
        return self.uprev

    def _saveData(self, xs, us, xhat, useq):
        """ Save data to lists. """
        self.xs.append(xs)
        self.us.append(us)
        self.x0.append(xhat)
        self.useq.append(useq)

def arrayToMatrix(*arrays):
    """Convert nummpy arrays to cvxopt matrices."""
    matrices = []
    for array in arrays:
        matrices += [cvx.matrix(array)]
    return tuple(matrices)

def dlqr(A, B, Q, R, M=None):
    """
    Get the discrete-time LQR for the given system.
    Stage costs are
        x'Qx + 2*x'Mu + u'Ru
    with M = 0 if not provided.
    """
    # For M != 0, we can simply redefine A and Q to give a problem with M = 0.
    if M is not None:
        RinvMT = scipy.linalg.solve(R,M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    Pi = scipy.linalg.solve_discrete_are(Atilde,B,Qtilde,R)
    K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A) + M.T)
    return (K, Pi)

def dlqe(A, C, Q, R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T
    return (L, P)

def c2d(A, B, Delta):
    """ Custom c2d function for linear systems."""
    
    # First construct the incumbent matrix
    # to take the exponential.
    (Nx, Nu) = B.shape
    M1 = np.concatenate((A, B), axis=1)
    M2 = np.zeros((Nu, Nx+Nu))
    M = np.concatenate((M1, M2), axis=0)
    Mexp = scipy.linalg.expm(M*Delta)

    # Return the extracted matrices.
    Ad = Mexp[:Nx, :Nx]
    Bd = Mexp[:Nx, -Nu:]
    return (Ad, Bd)

def eigvalEigvecTest(X, Y):
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

def assertDetectable(A, C):
    """Assert if the provided (A, C) pair is detectable."""
    assert eigvalEigvecTest(A, C)

def assertStabilizable(A, B):
    """Assert if the provided (A, B) pair is stabilizable."""
    assert eigvalEigvecTest(A.T, B.T)

def getAugMatricesForROCPenalty(A, B, Q, R, S):
    """ Get the Augmented A, B, C, and the noise covariance matrix."""

    # Get the shapes.
    Nx, Nu = B.shape

    # Augmented A.
    Aaug1 = np.concatenate((A, np.zeros((Nx, Nu))), axis=1)
    Aaug2 = np.zeros((Nu, Nx+Nu))
    Aaug = np.concatenate((Aaug1, Aaug2), axis=0)

    # Augmented B.
    Baug = np.concatenate((B, np.eye((Nu))), axis=0)

    # Augmented Q.
    Qaug = scipy.linalg.block_diag(Q, S)

    # Augmented R.
    Raug = R + S

    # Augmented M.
    Maug = np.concatenate((np.zeros((Nx, Nu)), -S), axis=0)

    # Return augmented matrices.
    return (Aaug, Baug, Qaug, Raug, Maug)

class KalmanFilter:

    def __init__(self, *, A, B, C, Qw, Rv, xPrior):
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
        self.xhat = [xPrior]
        self.xhatPred = []
        self.y = []
        self.uprev = []

    def _computeFilter(self):
        "Solve the DARE to compute the optimal L. "
        (self.L, _) = dlqe(self.A, self.C, self.Qw, self.Rv)

    def solve(self, y, uprev):
        """ Take a new measurement and do 
            the prediction and filtering steps."""
        xhat = self.xhat[-1]
        xhatPred = self.A @ xhat + self.B @ uprev
        xhat = xhatPred + self.L @ (y - self.C @ xhatPred)
        # Save data.
        self._saveData(xhat, xhatPred, y, uprev)
        return xhat
        
    def _saveData(self, xhat, xhatPred, y, uprev):
        """ Save the state estimates,
            Can be used for plotting later."""
        self.xhat.append(xhat)
        self.xhatPred.append(xhatPred)
        self.y.append(y)
        self.uprev.append(uprev)

class ExtendedKalmanFilter:

    def __init__(self, *, fxu, hx, Nx, Nu, Ny, Qw, Rv, xPrior, PPrior):
        """ Class to construct and perform state estimation
            using Kalman Filtering.
        """

        # Model.
        self.fxu, self.hx = fxu, hx

        # Sizes. 
        self.Nx, self.Nu, self.Ny = Nx, Nu, Ny

        # Noise variances.
        self.Qw = Qw
        self.Rv = Rv
        
        # Create lists for saving data.
        self.xhat = [xPrior]
        self.covxhat = [PPrior]
        self.xhatPred = []
        self.y = []
        self.uprev = []

    def _getA(self, xhat, uprev):
        """ Get the dynamic model A. """

        # Get the linearized model A.
        fxu, Nx, Nu = self.fxu, self.Nx, self.Nu
        fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])
        linModel = mpc.util.getLinearizedModel(fxu, [xhat, uprev], 
                                               ["A", "B"])
        A = linModel["A"]

        # Return linearized A matrix.
        return A

    def _getC(self, xhatPred):
        """ Get the dynamic model A. """

        # Get the linearized model C.
        hx, Nx = self.hx, self.Nx
        hx = mpc.getCasadiFunc(hx, [Nx], ["x"])
        linModel = mpc.util.getLinearizedModel(hx, [xhatPred], ["C"])
        C = linModel["C"]

        # Return linearized A matrix.
        return C

    def solve(self, y, uprev):
        """ Take a new measurement and do 
            the prediction and filtering steps."""

        # Get current state estimate.
        xhat = self.xhat[-1]
        P, Qw, Rv = self.covxhat[-1], self.Qw, self.Rv

        # Prediction step and get linear model.
        A = self._getA(xhat, uprev)
        xhatPred = self.fxu(xhat[:, 0], uprev[:, 0])[:, np.newaxis]
        PPred = A @ (P @ A.T) + Qw

        # Filtering step.
        C = self._getC(xhatPred)
        L = PPred @ (C.T @ np.linalg.inv(Rv + C @ (PPred @ C.T)))
        xhat = xhatPred + L @ (y - self.hx(xhatPred[:, 0])[:, np.newaxis])

        # Update covariance.
        P = PPred - L @ (C @ PPred)

        # Save data.
        self._saveData(xhat, P, xhatPred, y, uprev)
        
        # Return state estimate.
        return xhat
        
    def _saveData(self, xhat, P, xhatPred, y, uprev):
        """ Save the state estimates,
            Can be used for plotting later."""
        self.xhat.append(xhat)
        self.covxhat.append(P)
        self.xhatPred.append(xhatPred)
        self.y.append(y)
        self.uprev.append(uprev)

class TargetSelector:

    def __init__(self, *, A, B, C, H, Bd, Cd,
                          Rs, Qs, ulb, uub, ylb=None, yub=None):
        """ Class to construct and solve the following 
            target selector problem.

        min_(xs, us) 1/2*(|us - usp|^2_Rs + |C*xs + Cd*dhats - ysp|^2_Qs)

        s.t [I-A, -B;HC, 0][xs;us] = [Bd*dhats;H*(ysp-Cd*dhats)]
            ylb - Cd*dhat <= C*xs <= yub - Cd*dhat
            ulb <= us <= uub

        Construct the class and use the method "solve"
        for obtaining the solution.
        
        An instance of this class will also
        store the history of the solutions obtained.
        """
        
        # Model matrices.
        self.A = A
        self.B = B
        self.C = C

        # Disturbance model.
        self.Bd = Bd
        self.Cd = Cd

        # QP matrices.
        self.H = H
        self.Qs = Qs
        self.Rs = Rs

        # Get the store the sizes.
        self.Nx, self.Nu = B.shape
        self.Ny, self.Nd = Cd.shape
        self.Nrsp = H.shape[0]

        # Setup lists to store data
        self.usp, self.ysp, self.dhat = [], [], []
        self.xs, self.us = [], []

        # Get the input and output constraints.
        self.ulb, self.uub = ulb, uub
        self.ylb, self.yub = ylb, yub

        # Setup the fixed matrices.
        self._setupFixedMatrices()
    
    def _updateModel(self, A, B, C):
        """ Update linear model. """
        self.A, self.B, self.C = A, B, C
        self._setupFixedMatrices()

    def _setupFixedMatrices(self):
        """ Setup the matrices which don't change in
            an on-line simulation.

            1. Equality constraints.
                Aeq = [I-A, -B;HC, 0], Beq = [0, Bd;H, -H*Cd]
                Aeq = Beq*[ysp;dhat]

            2. Inequality constraints.
                Aineq = [C, 0;-C, 0;0, I_u;0, -I_u]*[xs;us]
                Bineq = [yub;-ylb;uub;-ulb] + [-Cd;Cd;0;0]*dhat
                Aineq <= Bineq

            3. Penalty matrices.
                P = [C'QsC, 0;0, Rs]
                q = [0, -C'Qs, C'*Qs*Cd;-Rs, 0, 0][usp;ysp;dhat]
              Objective: (1/2)[xs;us]'P[xs;us] + q'*[xs;us]

            """
        
        # Get sizes.
        Nx, Nu, Ny, Nd, Nrsp = self.Nx, self.Nu, self.Ny, self.Nd, self.Nrsp

        # Get matrices.
        A, B, C, Bd, Cd = self.A, self.B, self.C, self.Bd, self.Cd
        H, Qs, Rs = self.H, self.Qs, self.Rs

        # Get constraints.
        ulb, uub = self.ulb, self.uub
        ylb, yub = self.ylb, self.yub

        # Get the equality constraint matrices.
        # Get Aeq.
        Aeq11, Aeq12 = np.eye(Nx) - A, -B
        Aeq21, Aeq22 = H @ C, np.zeros((Nrsp, Nu))
        Aeq1 = np.concatenate((Aeq11, Aeq12), axis=1)
        Aeq2 = np.concatenate((Aeq21, Aeq22), axis=1)
        Aeq = np.concatenate((Aeq1, Aeq2), axis=0)

        # Get Beq.
        Beq11, Beq12 = np.zeros((Nx, Ny)), Bd
        Beq21, Beq22 = H, -(H @ Cd)
        Beq1 = np.concatenate((Beq11, Beq12), axis=1)
        Beq2 = np.concatenate((Beq21, Beq22), axis=1)
        Beq = np.concatenate((Beq1, Beq2))

        # Get the inequality constraints.
        Auineq = np.concatenate((np.eye(Nu), -np.eye(Nu)))
        Ayineq = np.concatenate((C, -C))

        # If both input/output constraints.
        if ylb is not None and yub is not None:
            
            # Get Aineq.
            Aineq1 = np.concatenate((Ayineq, np.zeros((2*Ny, Nu))), axis=1)
            Aineq2 = np.concatenate((np.zeros((2*Nu, Nx)), Auineq), axis=1)
            Aineq = np.concatenate((Aineq1, Aineq2))

            # Get Bineq1 and Bineq2.
            Bineq1 = np.concatenate((yub, -ylb, uub, -ulb))
            Bineq2 = np.concatenate((-Cd, Cd, np.zeros((2*Nu, Nd))))

        else: # If only input constraints.
            
            # Get Aineq.
            Aineq = np.concatenate((np.zeros((2*Nu, Nx)), Auineq), axis=1)

            # Get Bineq1 and Bineq2.
            Bineq1 = np.concatenate((uub, -ulb), axis=0)
            Bineq2 = np.zeros((2*Nu, Nd))

        # Get the penalty matrices.
        # Get P.
        P11, P22 = C.T @ (Qs @ C), Rs
        P = scipy.linalg.block_diag(P11, P22)
        
        # Get q.
        q11, q12, q13 = np.zeros((Nx, Ny)), -(C.T @ Qs), C.T @ (Qs @ Cd)
        q21, q22, q23 = -Rs, np.zeros((Nu, Ny)), np.zeros((Nu, Nd))
        q1 = np.concatenate((q11, q12, q13), axis=1)
        q2 = np.concatenate((q21, q22, q23), axis=1)
        q = np.concatenate((q1, q2))

        # Save all matrices.
        self.Aeq, self.Beq, self.Aineq = Aeq, Beq, Aineq
        self.Bineq1, self.Bineq2 = Bineq1, Bineq2
        self.P, self.q = P, q

    def _getQPMatrices(self, usp, ysp, dhat):
        """ Get the matrices which change in real-time."""

        # Get Equality constraint.
        Aeq = self.Aeq
        Beq = self.Beq @ np.concatenate((ysp, dhat), axis=0)

        # Get Inequality constraint.
        Aineq = self.Aineq
        Bineq = self.Bineq1 + self.Bineq2 @ dhat
        
        # Get penalty matrices.
        P = self.P
        q = self.q @ np.concatenate((usp, ysp, dhat), axis=0)

        # Return (P, q, Aeq, Beq, Aineq, Bineq)
        return (P, q, Aineq, Bineq, Aeq, Beq)

    def solve(self, usp, ysp, dhat):
        "Solve the target selector QP, output is the tuple (xs, us)."

        # Get the matrices for the QP which depend of ysp and dhat
        qpMatrices = self._getQPMatrices(usp, ysp, dhat)

        # Solve and save data.
        solution = cvx.solvers.qp(*arrayToMatrix(*qpMatrices))

        # Split solution.
        (xs, us) = np.split(np.asarray(solution['x']), [self.Nx])
        
        # Save Data.
        self._saveData(xs, us, usp, ysp, dhat)

        # Return the solution.
        return (xs, us)

    def _saveData(self, xs, us, usp, ysp, dhat):
        """ Save the state estimates,
            Can be used for plotting later."""
        self.xs.append(xs)
        self.us.append(us)
        self.usp.append(usp)
        self.ysp.append(ysp)
        self.dhat.append(dhat)

class DenseQPRegulator:
    """ Class to construct and solve the linear MPC regulator QP
        using a dense QP formulation.

        The problem:

        V(x0, \mathbf{u}) = (1/2) \sum_{k=0}^{N-1} [x(k)'Qx(k) + u(k)'Ru(k) + 2x(k)'Mu(k)] + (1/2)x(N)'Pfx(N)
        
        subject to:
            x(k+1) = Ax(k) + Bu(k)
            ulb <= u(k) <= uub

        This class eliminates all the states
        from the set of decision variables and 
        solves a dense formulation of the QP. 
    """
    def __init__(self, *, A, B, Q, R, M, N, ulb, uub):

        # Model.
        self.A, self.B = A, B

        # Penalty matrices.
        self.Q, self.R, self.M = Q, R, M

        # Horizon length.
        self.N = N

        # Input constraints.
        self.ulb, self.uub = ulb, uub

        # Set sizes.
        self.Nx, self.Nu = B.shape

        # Get the LQR gain, only used for reparameterization for the dense QP.
        self.Krep, self.Pf = dlqr(A, B, Q, R, M)

        # Reparametrie QP if needed and then setup QP matrices.
        self._reparameterize()
        self._setupFixedMatrices()

        # Create lists to save data.
        self.x0, self.useq = [], []

    def _updateModel(self, A, B):
        """ Update linear model. """

        self.A, self.B = A, B
        Q, R, M = self.Q, self.R, self.M
        self.Krep, self.Pf = dlqr(A, B, Q, R, M)
        self._reparameterize()
        self._setupFixedMatrices()

    def _reparameterize(self):
        """
        Reparameterize A, B, Q, R, M, and G, h.
        A = A+BK, B = B, Q = Q+K'RK+MK+K'M', R=R
        M = K'R + M
        Pf is the solution of the Riccati Eqn using the 
        new parametrized matrices. 
        """
        A, B, Q, R, M, Krep = self.A, self.B, self.Q, self.R, self.M, self.Krep
        (eigvals, _) = np.linalg.eig(A)
        if any(np.absolute(eigvals)>=1.):
            self.A = A + B @ Krep
            self.Q = Q + Krep.T @ (R @ Krep) + M @ Krep + Krep.T @ M.T
            self.M = Krep.T @ R + M
            self.reParamet = True
        else:
            self.reParamet = False

    def _setupFixedMatrices(self):
        """" Setup the fixed matrices which don't change in 
             real-time.
            
             Finally the QP should be in this format.
             min_x  (1/2)x'*P*x + q'*x
             s.t     Aineq*x <= Bineq
                     Aeq*x = beq
             
             in which, x = [u(0), u(1), ..., u(N-1)]
        """

        # Get penalty matrices.
        tA, tB = self._get_tA_tB()
        tQ, tR, tM, tK = self._get_tQ_tR_tM_tK()
        P, q = self._getPq(tA, tB, tQ, tR, tM)

        # Get inequality matrices.
        Aineq, Bineq1, Bineq2 = self._getAineqBineq(tA, tB, tK)

        # Assign computed matrices to attributes.
        self.P, self.q = P, q
        self.Aineq, self.Bineq1, self.Bineq2 = Aineq, Bineq1, Bineq2
        self.tA, self.tB, self.tK = tA, tB, tK

    def _get_tA_tB(self):
        """ Get the Atilde and Btilde.
            N is the number of state outputs is asked for. 
        
        xseq = tA*x0 + tB*useq
        xseq = (N+1)*Nx, useq = N*Nu
        tA = [(N+1)*Nx]*Nx, tB = [(N+1)*Nx]*[N*Nu]

        tA = [I;
              A;
              A^2;
              .;
              A^N]
        tB = [0, 0, 0, ..;
              B, 0, 0, ...;
              AB, B, 0, ...;
              ...;
              A^(N-1)B, A^(N-2)B, ..., B]
        """

        # Extract attributes.
        A, N = self.A, self.N

        # Get tA.
        tA = np.concatenate([np.linalg.matrix_power(A, i) for i in range(N+1)])

        # Get tB.
        tB = np.concatenate([self._get_tBRow(i) for i in range(N+1)])

        # Return.
        return (tA, tB)
    
    def _get_tBRow(self, i):
        """ Returns the ith row of tB. """

        # Extract attributes.
        A, B, N = self.A, self.B, self.N
        (Nx, Nu) = B.shape
        
        # Get tBi.
        tBi = [np.linalg.matrix_power(A, i-j-1) @ B 
                if j<i 
                else np.zeros((Nx, Nu))
                for j in range(N)]
        tBi = np.concatenate(tBi, axis=1)
        
        # Return.
        return tBi

    def _get_tQ_tR_tM_tK(self):
        """ Get the block diagonal matrices for dense QP. 

        tQ = [Q, 0, 0, ...
              0, Q, 0, ...
              0, 0, Q, ...
              ...
              0, 0, 0, ..., Pf]

        tR = [R, 0, 0, ...
              0, R, 0, ...
              0, 0, R, ...
              ...
              0, 0, 0, ... R]

        tM = [M, 0, 0, ...
              0, M, 0, ...
              0, 0, M, ...
              ...
              0, 0, 0, ..., M
              0, 0, 0, ......]

        tK = [K, 0, 0, ......,
              0, K, 0, ......,
              0, 0, K, 0, ...,
              ...............,
              0, 0, 0, ......K]

        Convert 

        V = (1/2) \sum_{k=0}^{N-1} [x(k)'Qx(k) + u(k)'Ru(k) + 2x(k)'Mu(k)] 
        V += (1/2)x(N)'Pfx(N)
        

        to 

        V = (1/2)*[xseq'*tQ*xseq + useq'*tR*useq + 2*xseq'*tM*useq]

        in which, xseq = [x(0), x(1), x(2), ...., x(N)]
                  useq = [u(0), u(1), u(2), ...., u(N-1)] """

        # Extract attributes.
        Q, R, M, Krep, Pf = self.Q, self.R, self.M, self.Krep, self.Pf
        Nx, Nu, N = self.Nx, self.Nu, self.N

        # Construct matrices.
        # tQ
        tQ = scipy.linalg.block_diag(*[Q if i<N else Pf for i in range(N+1)])

        # tR
        tR = scipy.linalg.block_diag(*[R for _ in range(N)])

        # tM
        tM = scipy.linalg.block_diag(*[M for _ in range(N)])
        tM = np.concatenate((tM, np.zeros((Nx, N*Nu))), axis=0)

        # tK.
        tK = scipy.linalg.block_diag(*[Krep for _ in range(N)])

        # Return.
        return (tQ, tR, tM, tK)

    def _getPq(self, tA, tB, tQ, tR, tM):
        """ Get the penalites for solving the QP.
            P = tB'*tQ*tB + tR + tB'*tM + tM'*tB 
            tq = (tB'*Q + M)*tA
        """

        # Get P and q matrices. 
        P = tB.T @ (tQ @ tB) + tR + tB.T @ tM + tM.T @ tB
        q = (tB.T @ tQ  + tM.T) @ tA

        # Return.
        return (P, q)

    def _getAineqBineq(self, tA, tB, tK):
        """ Get the inequality matrices. """

        # Extract some parameters.
        Nx, Nu, N = self.Nx, self.Nu, self.N
        ulb, uub = self.ulb, self.uub
        Krep = self.Krep

        # Aineq.
        Auineq = np.concatenate((np.eye(Nu), -np.eye(Nu)), axis=0)
        Auineq = scipy.linalg.block_diag(*[Auineq for _ in range(N)])
        if self.reParamet:
            Aineq = Auineq @ (tK @ tB[:N*Nx, :]) + Auineq
        else:
            Aineq = Auineq

        # Get Bineq1 and Bineq2.
        Bineq1 = np.concatenate((uub, -ulb), axis=0)
        Bineq1 = np.concatenate([Bineq1 for _ in range(N)])

        # If the QP is a reparametrized QP, get concatenations of K as well.
        if self.reParamet:
            Bineq2 = -Auineq @ (tK @ tA[:N*Nx, :])
        else:
            Bineq2 = np.zeros((2*Nu*N, Nx))

        # Return inequality matrices. 
        return (Aineq, Bineq1, Bineq2)

    def _getQPMatrices(self, x0):
        """ Get the RHS of the equality constraint."""

        # Get Inequality constraints.
        Aineq = self.Aineq
        Bineq = self.Bineq1 + self.Bineq2 @ x0
        
        # Get penalty matrices.
        P = self.P
        q = self.q @ x0

        # Return QP matrices.
        return (P, q, Aineq, Bineq)

    def solve(self, x0):
        """Setup and the solve the dense QP, output is 
        the first element of the sequence.
        If the problem is reparametrized, go back to original 
        input variable. 
        """
        # Get QP matrices.
        qpMatrices = self._getQPMatrices(x0)

        # Solve.
        solution = cvx.solvers.qp(*arrayToMatrix(*qpMatrices))
        
        # Extract input sequence.
        useq = np.asarray(solution['x'])
        if self.reParamet:
            
            # Extract attributes.
            tA, tB, tK = self.tA, self.tB, self.tK
            N, Nx = self.N, self.Nx

            # Get useq in original space.
            useq = useq + tK @ (tA[:N*Nx, :] @ x0 + tB[:N*Nx, :] @ useq)

        # Save data.
        self._saveData(x0, useq)

        # Return optimal input sequence.
        return useq

    def _saveData(self, x0, useq):
        # Save data.
        self.x0.append(x0)
        self.useq.append(useq)

class LinearMPCController:
    """Class that instantiates the KalmanFilter, 
    the TargetSelector, and the MPCRegulator classes
    into one and solves tracking MPC problems with 
    linear models.
    """
    def __init__(self, *, A, B, C, H,
                 Qwx, Qwd, Rv, xprior, dprior,
                 Rs, Qs, Bd, Cd, usp, uprev,
                 Q, R, S, ulb, uub, N):
        
        # Save attributes.
        self.A = A
        self.B = B
        self.C = C
        self.H = H
        self.Qwx = Qwx
        self.Qwd = Qwd
        self.Rv = Rv
        self.xprior = xprior
        self.dprior = dprior
        self.Rs = Rs
        self.Qs = Qs
        self.Bd = Bd
        self.Cd = Cd
        self.usp = usp
        self.uprev = uprev
        self.useq = np.tile(uprev, (N, 1))
        self.Q = Q
        self.R = R
        self.S = S
        self.ulb = ulb
        self.uub = uub
        self.N = N

        # Sizes.
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]
        self.Nd = Bd.shape[1]

        # Instantiate the required classes. 
        self.filter = LinearMPCController.setup_filter(A=A, B=B, C=C, Bd=Bd, 
                                                       Cd=Cd,
                                                       Qwx=Qwx, Qwd=Qwd, Rv=Rv, 
                                                       xprior=xprior, dprior=dprior)
        self.target_selector = LinearMPCController.setup_target_selector(A=A, 
                                                        B=B, C=C, H=H,
                                                        Bd=Bd, Cd=Cd, 
                                                        usp=usp, Qs=Qs, Rs=Rs, 
                                                        ulb=ulb, uub=uub)
        self.regulator = LinearMPCController.setup_regulator(A=A, B=B, Q=Q, 
                                                        R=R, S=S, 
                                                        N=N, ulb=ulb, uub=uub)

        # List object to store the average stage 
        # costs and the average computation times.
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A,  
                                                                    B, Q, R, S)
        (_, _, self.Qaug, self.Raug, self.Maug) = aug_mats
        self.average_stage_costs = [np.zeros((1, 1))]
        self.computation_times = []

    @staticmethod
    def setup_filter(A, B, C, Bd, Cd, Qwx, Qwd, Rv, xprior, dprior):
        """ Augment the system with an integrating 
        disturbance and setup the Kalman Filter."""
        (Aaug, Baug, Caug, Qwaug) = LinearMPCController.get_augmented_matrices_for_filter(A, B, C, Bd, Cd, Qwx, Qwd)
        return KalmanFilter(A=Aaug, B=Baug, C=Caug, Qw=Qwaug, Rv=Rv,
                            xprior = np.concatenate((xprior, dprior)))
    
    @staticmethod
    def setup_target_selector(A, B, C, H, Bd, Cd, usp, Qs, Rs, ulb, uub):
        """ Setup the target selector for the MPC controller."""
        return TargetSelector(A=A, B=B, C=C, H=H, Bd=Bd, Cd=Cd, 
                              usp=usp, Rs=Rs, Qs=Qs, ulb=ulb, uub=uub)
    
    @staticmethod
    def setup_regulator(A, B, Q, R, S, N, ulb, uub):
        """ Augment the system for rate of change penalty and 
        build the regulator."""
        aug_mats = LinearMPCController.get_augmented_matrices_for_regulator(A,  
                                                                    B, Q, R, S)
        (Aaug, Baug, Qaug, Raug, Maug) = aug_mats
        return DenseQPRegulator(A=Aaug, B=Baug, Q=Qaug, R=Raug, 
                                N=N, M=Maug, ulb=ulb, uub=uub)
    
    @staticmethod
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
    
    def control_law(self, ysp, y):
        """
        Takes the measurement, the previous control input,
        and compute the current control input.

        Count times only for solving the regulator QP.
        """
        (xhat, dhat) =  LinearMPCController.get_state_estimates(self.filter, y, 
                                                            self.uprev, self.Nx)
        (xs, us) = LinearMPCController.get_target_pair(self.target_selector, 
                                                       ysp, dhat)
        tstart = time.time()
        self.useq = LinearMPCController.get_control_sequence(self.regulator, 
                                                    xhat, self.uprev, xs, us,
                                                    self.ulb, self.uub)
        tend = time.time()
        avg_ell = LinearMPCController.get_updated_average_stage_cost(xhat, 
                    self.uprev, xs, us, self.useq[0:self.Nu, :], 
                    self.Qaug, self.Raug, self.Maug, 
                    self.average_stage_costs[-1], len(self.average_stage_costs))
        self.average_stage_costs.append(avg_ell)
        self.uprev = self.useq[0:self.Nu, :]
        self.computation_times.append(tend - tstart)
        return self.uprev
    
    @staticmethod
    def get_state_estimates(filter, y, uprev, Nx):
        """Use the filter object to perform state estimation."""
        return np.split(filter.solve(y, uprev), [Nx])

    @staticmethod
    def get_target_pair(target_selector, ysp, dhat):
        """ Use the target selector object to 
            compute the targets."""
        return target_selector.solve(ysp, dhat)

    @staticmethod
    def get_control_sequence(regulator, x, uprev, xs, us, ulb, uub):
        # Change the constraints of the regulator. 
        regulator.ulb = ulb - us
        regulator.uub = uub - us
        # x0 in deviation from the steady state.
        x0 = np.concatenate((x-xs, uprev-us))
        return regulator.solve(x0) + np.tile(us, (regulator.N, 1))

    @staticmethod
    def get_updated_average_stage_cost(x, uprev, xs, us, u, 
                                       Qaug, Raug, Maug, 
                                       average_stage_cost, time_index):
        # Get the augmented state and compute the stage cost.
        x = np.concatenate((x-xs, uprev-us), axis=0)
        u = u - us
        stage_cost = x.T @ (Qaug @ x) + u.T @ (Raug @ u) 
        stage_cost = stage_cost + x.T @ (Maug @ u) + u.T @ (Maug.T @ x)
        # x0 in deviation from the steady state.
        return (average_stage_cost*(time_index-1) + stage_cost)/time_index