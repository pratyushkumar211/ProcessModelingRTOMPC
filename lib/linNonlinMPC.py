# [depends] economicopt.py
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
        y += self.ynoise_std*np.random.randn(self.Ny, 1)
        self._appendData(x, u, p, y)
        # Return.
        return y

    def _appendData(self, x, u, p, y):
        """ Append the data into the lists.
            Used for plotting in the specific subclasses.
        """
        self.x.append(x)
        self.u.append(u)
        self.p.append(p)
        self.y.append(y)
        self.t.append(self.t[-1] + self.sample_time)

# class NonlinearEMPCRegulator:

#     def __init__(self, *, fxup, lxup, Nx, Nu, Np,
#                  Nmpc, ulb, uub, t0Guess, t0EmpcPars):
#         """ Class to construct and solve nonlinear economic model 
#             predictive control problem.
        
#         Problem setup:
#         The current time is T, we have x.

#         Optimization problem:
#         min_{u[0:N-1]} sum_{k=0^k=N-1} l(x(k), u(k), p(k))
#         subject to:
#         x(k+1) = f(x(k), u(k), p(k)), k=0 to N-1, ulb <= u(k) <= uub
#         """
#         # Model.
#         self.fxup = fxup
#         self.lxup = lxup

#         # Sizes.
#         self.Nmpc = Nmpc
#         self.Nx = Nx
#         self.Nu = Nu
#         self.Np = Np

#         # Create lists for saving data. 
#         self.x0 = []
#         self.useq = []
#         self.xseq = []

#         # Initial guess and parameters.
#         self.t0Guess = t0Guess
#         self.t0EmpcPars = t0EmpcPars

#         # Get the hard constraints on inputs and the soft constraints.
#         self.ulb = ulb
#         self.uub = uub

#         # Build the nonlinear MPC regulator.
#         self._setupRegulator()

#     def _setupRegulator(self):
#         """ Construct a Nonlinear economic MPC regulator. """

#         # Sizes and arguments.
#         N = dict(x=self.Nx, u=self.Nu, p=self.Np, t=self.Nmpc)
#         funcargs = dict(f=["x", "u", "p"], l=["x", "u", "p"])
        
#         # Some parameters for the regulator.
#         empcPars = self.t0EmpcPars
#         guess = self.t0Guess
#         x0 = guess['x']
#         lb = dict(u=self.ulb)
#         ub = dict(u=self.uub)
        
#         # Construct the EMPC regulator.
#         breakpoint()
#         self.regulator = mpc.nmpc(f=self.fxup, l=self.lxup, N=N, 
#                                   funcargs=funcargs, x0=x0, p=empcPars, lb=lb, ub=ub, guess=guess)
#         self.regulator.solve()
#         self.regulator.saveguess()

#         # Get the x and u sequences and save data.
#         useq = np.asarray(casadi.horzcat(*self.regulator.var['u'])).T
#         xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
#         self._saveData(x0, useq, xseq)
    
#     def solve(self, x0, empcPars):
#         """Setup and the solve the dense QP, output is
#         the first element of the sequence.
#         If the problem is reparametrized, go back to original
#         input variable.
#         """
#         self.regulator.par["p"] = list(empcPars)
#         self.regulator.fixvar("x", 0, x0)
#         self.regulator.solve()
#         self.regulator.saveguess()

#         # Get the x and u sequences and save data.
#         useq = np.asarray(casadi.horzcat(*self.regulator.var['u'])).T
#         xseq = np.asarray(casadi.horzcat(*self.regulator.var['x'])).T
#         self._saveData(x0, useq, xseq)

#         # Return the full sequence.
#         return useq

#     def _saveData(self, x0, useq, xseq):
#         " Append data. "
#         self.x0.append(x0)
#         self.useq.append(useq)
#         self.xseq.append(xseq)

# class NonlinearMHEEstimator:

#     def __init__(self, *, fxu, hx, Nmhe, Nx, Nu, Ny,
#                  xhatPast, uPast, yPast, P0inv, Qwinv, Rvinv):
#         """ Class to construct and perform state estimation
#             using moving horizon estimation.
        
#         Problem setup:
#         The current time is T
#         Measurements available: u_[T-N:T-1], y_[T-N:T]

#         Optimization problem:
#         min_{x_[T-N:T]} |x(T-N)-xprior|_{P0inv} + 
#                         sum{t=T-N to t=N-1} |x(k+1)-x(k)|_{Qwinv} + 
#                         sum{t=T-N to t=T} |y(k)-h(x(k))|_{Rvinv}

#         subject to: x(k+1) = f(x(k), u(k), w(k)),
#                     y(k) = h(x) + v(k), k=T-N to T-1
#         x is the augmented state.

#         xprior is an array of previous smoothed estimates, xhat(k:k) from -T:-1
#         y: -T:0
#         u: -T:-1
#         The constructor solves and gets a smoothed estimate of x at time 0.
#         """
#         self.fxu = fxu
#         self.hx = hx

#         # Penalty matrices.
#         self.P0inv = P0inv
#         self.Qwinv = Qwinv
#         self.Rvinv = Rvinv

#         # Sizes.
#         self.Nx = Nx
#         self.Nu = Nu
#         self.Ny = Ny
#         self.Nmhe = Nmhe

#         # Create lists for saving data.
#         self.xhat = list(xhatPast)
#         self.u = list(uPast)
#         self.y = list(yPast)

#         # Build the estimator.
#         self._setupMheEstimator()

#     def _stageCost(self, w, v):
#         """ Stage cost in moving horizon estimation. """
#         return mpc.mtimes(w.T, self.Qwinv, w) + mpc.mtimes(v.T, self.Rvinv, v)

#     def _priorCost(self, x, xprior):
#         """Prior cost in moving horizon estimation."""
#         dx = x - xprior
#         return mpc.mtimes(dx.T, self.P0inv, dx)

#     def _setupMheEstimator(self):
#         """ Construct a MHE solver. """

#         N = dict(x=self.Nx, u=self.Nu, y=self.Ny, t=self.Nmhe)
#         funcargs = dict(f=["x", "u"], h=["x"], l=["w", "v"], 
#                         lx=["x", "x0bar"])
        
#         # Setup stage costs.
#         l = mpc.getCasadiFunc(self._stageCost, 
#                               [self.Nx, self.Ny], funcargs["l"])
#         lx = mpc.getCasadiFunc(self._priorCost, 
#                                [self.Nx, self.Nx], funcargs["lx"])

#         # Get Guess for the NLP.
#         guess = dict(x=self.xhat[-1], w=np.zeros((self.Nx,)),
#                      v=np.zeros((self.Ny,)))

#         # Setup the MHE estimator.        
#         self.mheEstimator = mpc.nmhe(f=self.fxu,
#                                      h=self.hx, wAdditive=True,
#                                      N=N, l=l, lx=lx, u=self.u, y=self.y,
#                                      funcargs=funcargs,
#                                      guess=guess,
#                                      x0bar=self.xhat[0],
#                                      verbosity=0)
#         self.mheEstimator.solve()
#         self.mheEstimator.saveguess()

#         # Get estimated state sequence.
#         xhat = np.asarray(self.mheEstimator.var["x"][-1])
#         self.xhat.append(xhat)
        
#     def solve(self, y, uprev):
#         """ Use the new data, solve the NLP, and store data.
#             At this time:
#             xhat: list of length T+1
#             y: list of length T+1
#             uprev: list of length T
#         """

#         # Assemble data and solve NLP.
#         Nmhe = self.Nmhe
#         self.mheEstimator.par["x0bar"] = self.xhat[-Nmhe]
#         self.mheEstimator.par["y"] = self.y[-Nmhe:] + [y]
#         self.mheEstimator.par["u"] = self.u[-Nmhe+1:] + [uprev]
#         self.mheEstimator.solve()
#         self.mheEstimator.saveguess()

#         # Get estimated state sequence and save data.
#         xhat = np.asarray(self.mheEstimator.var["x"][-1])
#         self._saveData(xhat, y, uprev)

#         # Return the state estimate.
#         return xhat

#     def _saveData(self, xhat, y, uprev):
#         """ Append the data to the lists. """
#         self.xhat.append(xhat)
#         self.y.append(y)
#         self.u.append(uprev)

# class NonlinearEMPCController:
#     """ Class that instantiates a nonlinear MHE/EMPC regulator
#         and solves economic empc problems.

#         fxup is a continous time model.
#         hx is same in both the continous and discrete time.
#         Bd is in continuous time.
#     """
#     def __init__(self, *, fxup, hx, lyup, Bd, Cd, Delta,
#                           Nx, Nu, Ny, Nd, xs, us, ds, ps,
#                           econPars, distPars, ulb, uub, Nmpc,
#                           Qwx, Qwd, Rv, Nmhe):
        
#         # Model, stage cost, sample time.
#         self.fxup = fxup
#         self.hx = hx
#         self.lyup = lyup
#         self.Delta = Delta

#         # Disturbance model.
#         self.Bd = Bd
#         self.Cd = Cd

#         # EMPC parameters. 
#         self.empcPars = np.concatenate((distPars, econPars), axis=1)

#         # Sizes.
#         self.Nx = Nx
#         self.Nu = Nu
#         self.Ny = Ny
#         self.Nd = Nd
#         self.Nmdist = distPars.shape[1]
#         self.Necon = econPars.shape[1]

#         # Steady states of the system.
#         self.xs = xs
#         self.us = us
#         self.ds = ds
#         self.ps = ps
#         self.ys = hx(xs)
#         self.uprev = us[:, np.newaxis]
#         self.pprev = ps[:, np.newaxis]

#         # MPC Regulator parameters.
#         self.ulb = ulb
#         self.uub = uub
#         self.Nmpc = Nmpc
#         self._setupRegulator()
#         breakpoint()
#         # MHE Parameters.
#         self.Qwx = Qwx
#         self.Qwd = Qwd
#         self.Rv = Rv
#         self.Nmhe = Nmhe
#         self._setupEstimator()
#         breakpoint()
#         # Parameters to save.
#         self.computationTimes = []

#     def _setupRegulator(self):
#         """ Setup nonlinear EMPC regulator. """

#         # Dynamic model.
#         fxup_ct = lambda x, u, p: mpc.vcat([self.fxup(x[:self.Nx], u, p) + 
#                                             self.Bd @ x[self.Nx:], 
#                                     np.zeros((self.Nd, 1))])
#         fxup_dt = c2dNonlin(fxup_ct, self.Delta, p=True)
#         fxup = lambda x, u, p: fxup_dt(mpc.vcat([x, p[:self.Nd]]), u, 
#                                     p[self.Nd:self.Nd + self.Nmdist])[:self.Nx]
#         f = mpc.getCasadiFunc(fxup, [self.Nx, self.Nu, 
#                                      self.Nd + self.Nmdist + self.Necon], 
#                               ["x", "u", "p"])

#         # Measurement and stage cost function.
#         hxp = lambda x, p: self.hx(x) + self.Cd @ p[:self.Nd]
#         lxup = lambda x, u, p: self.lyup(hxp(x, p), u, 
#                                          p[self.Nd + self.Nmdist:])
#         l = mpc.getCasadiFunc(lxup, 
#                         [self.Nx, self.Nu, self.Nd + self.Nmdist + self.Necon], 
#                         ["x", "u", "p"])

#         # Get initial guess.
#         t0Guess = dict(x=self.xs, u=self.us)
#         ds = np.repeat(self.ds[np.newaxis, :], self.Nmpc, axis=0)
#         t0EmpcPars = np.concatenate((ds, self.empcPars[:self.Nmpc, :]), axis=1)
        
#         # Construct the EMPC regulator.
#         self.regulator  = NonlinearEMPCRegulator(fxup=f, lxup=l,
#                                                  Nx=self.Nx, Nu=self.Nu, 
#                                         Np=self.Nd + self.Nmdist + self.Necon,
#                                         Nmpc=self.Nmpc, ulb=self.ulb, 
#                                         uub=self.uub, t0Guess=t0Guess,
#                                         t0EmpcPars=t0EmpcPars)

#     def _setupEstimator(self):
#         """ Setup MHE. """

#         # Dynamic model.
#         fxu = lambda x, u: mpc.vcat([self.fxup(x[:self.Nx], 
#                             u[:self.Nu], u[self.Nu:]) + self.Bd @ x[self.Nx:], 
#                                     np.zeros((self.Nd, 1))])
#         f = mpc.getCasadiFunc(fxu, [self.Nx + self.Nd, self.Nu + self.Nmdist], 
#                               ["x", "u"], rk4=True, Delta=self.Delta, M=1)

#         # Measurement function.
#         hx = lambda x: self.hx(x[:self.Nx]) + self.Cd @ x[self.Nx:]
#         h = mpc.getCasadiFunc(hx, [self.Nx + self.Nd], ["x"])

#         # Prior estimates and data.
#         # XhatPast.
#         xhatPast = np.concatenate((self.xs, self.ds))
#         xhatPast = np.expand_dims(xhatPast, axis=(0, 2))
#         xhatPast = np.repeat(xhatPast, self.Nmhe, axis=0)

#         # Upast.
#         uPast = np.concatenate((self.us, self.ps))
#         uPast = np.expand_dims(uPast, axis=(0, 2))
#         uPast = np.repeat(uPast, self.Nmhe, axis=0)

#         # Ypast.
#         yPast = np.expand_dims(self.ys, axis=(0, 2))
#         yPast = np.repeat(yPast, self.Nmhe + 1, axis=0)

#         # Penalty matrices.
#         Qwxinv = np.linalg.inv(self.Qwx)
#         Qwdinv = np.linalg.inv(self.Qwd)
#         Qwinv = scipy.linalg.block_diag(Qwxinv, Qwdinv)
#         P0inv = Qwinv
#         Rvinv = np.linalg.inv(self.Rv)

#         # Construct the MHE estimator.
#         self.estimator = NonlinearMHEEstimator(fxu = f, hx = h,
#                                         Nmhe = self.Nmhe, 
#                                         Nx = self.Nx + self.Nd, 
#                                         Nu = self.Nu + self.Nmdist, 
#                                         Ny = self.Ny,
#                                         xhatPast = xhatPast, uPast = uPast, 
#                                         yPast = yPast, P0inv = P0inv, 
#                                         Qwinv = Qwinv, 
#                                         Rvinv = Rvinv)

#     def control_law(self, simt, y):
#         """
#         Takes the measurement and the previous control input
#         and compute the current control input.
#         """
#         tstart = time.time()

#         # Get state estimate.
#         up = np.concatenate((self.uprev, self.pprev))
#         xdhat = self.estimator.solve(y, up)
#         xhat, dhat = xdhat[:self.Nx, :], xdhat[self.Nx:, :]

#         # Get EMPC pars.
#         mpc_N = slice(simt, simt + self.Nmpc, 1)
#         dhatEmpc = np.repeat(dhat.T, self.Nmpc, axis=0)
#         empcPars = np.concatenate((dhatEmpc, self.empcPars[mpc_N, :]), axis=1)

#         # Solve nonlinear regulator.
#         useq = self.regulator.solve(xhat, empcPars)
#         self.uprev = useq[:1, :].T

#         self.pprev = self.empcPars[simt:simt+1, :self.Nmdist].T

#         # Get compuatation time and save.
#         tend = time.time()
#         self.computationTimes.append(tend - tstart)

#         # Return control input.
#         return self.uprev

# class RTOOptimizer:

#     def __init__(self, *, rto_type, fxup, hx, lyup, Nx, Nu, Necon, ulb, 
#                           uub, xguess, uguess, picnn_lyup, picnn_parids):
#         """ Class to construct and solve steady state optimization
#             problems.
                
#         If the rto_type variable is "dynmodel_optimization".

#         The Optimization problem solved is:
#         min_{xs, us} l(ys, us, p)
#         subject to:
#         xs = f(xs, us)
#         ys = h(xs)
#         ulb <= us <= uub
        
#         If the rto_type variable is "picnn_optimization"
#         min_{us} l(us, p)
#         subject to:
#         ulb <= us <= uub
#         We still provide the fxu, hx functions to compute xs, us 
#         by solving xs = f(xs, us) and ys = h(xs).
        
#         """

#         # RTO type. 
#         if rto_type == "dynmodel_optimization":
#             self.getOptimum = self.getDynModelOptimum
#             self.model_pars = dict(Nx=Nx, Nu=Nu, Np=Necon, 
#                                    ulb=ulb, uub=uub)
#         elif rto_type == "picnn_optimization":
#             self.getOptimum = self.getPicnnOptimum
#             self.picnn_lyup = picnn_lyup
#             self.picnn_parids = picnn_parids
#             self.model_pars = dict(Nx=Nx, Nu=Nu, Np=len(picnn_parids), 
#                                    ulb=ulb, uub=uub)
#         else:
#             raise ValueError("RTO type not supported yet.")

#         # Model.
#         self.fxup = fxup
#         self.hx = hx
#         self.lyup = lyup

#         # Inital guess for the optimization.
#         self.guess = dict(x=xguess, u=uguess)

#     def getDynModelOptimum(self, pecon, pdist):
#         """ Get the steady state optimum of the dynamic model. """

#         # Get the stage cost. 
#         lyu = lambda y, u: self.lyup(y, u, pecon)            
#         fxu = lambda x, u: self.fxup(x, u, pdist)

#         # Get the optimum.
#         self.xs, self.us, ys, _ = get_dynmodel_ss_optimum(fxu=fxu, 
#                                     hx=self.hx, 
#                                     lyu=lyu, parameters=self.model_pars, 
#                                     guess=self.guess)

#         # Return the steady state input and state.
#         return self.xs, self.us

#     def getPicnnOptimum(self, pecon, pdist):
#         """ Get the steady state optimum using the ICNN. """

#         # Get the parameter and dynamic model function.
#         p = np.concatenate((pecon, pdist))[self.picnn_parids]
#         fxu = lambda x, u: self.fxup(x, u, pdist)

#         # Get the optimum. 
#         self.us, _ = get_picnn_ss_optimum(lyup=self.picnn_lyup, 
#                                           parameters=self.model_pars, 
#                                           uguess=self.guess['u'], pval=p)

#         # Get the corresponding xs, ys using the model.
#         lyu = lambda y, u: self.lyup(y, u, pecon) 
#         self.xs, _ = get_xs_sscost(fxu=fxu, hx=self.hx, lyu=lyu,
#                                    us=self.us, parameters=self.model_pars, 
#                                    xguess=self.guess['x'])

#         # Return the steady state input and state.
#         return self.xs, self.us

#     def updateGuess(self):
#         """ Just update the guess for the RTO problem. """
#         if hasattr(self, 'xs') and hasattr(self, 'us'):
#             self.guess = dict(x=self.xs, u=self.us)
#         else:
#             print("Optimization has not been solved yet. Did not update guess.")

# class RTOLinearMPController:
#     """ Class to instantiate a Kalman Filter
#         and Linear MPC Regulator. Updates linear model/targets
#         based on steady state optimums.

#         fxup is in discrete-time.
#     """
#     def __init__(self, *, fxup, hx, lyup, econPars, distPars, rto_type, 
#                           tssOptFreq, picnn_lyup, picnn_parids,
#                           Nx, Nu, Ny, xs, us, ps, Q, R, S, ulb, uub, Nmpc, Qw, Rv):
        
#         # Model and stage cost.
#         self.fxup = fxup
#         self.hx = hx
#         self.lyup = lyup

#         # Economic parameters.
#         self.empcPars = np.concatenate((distPars, econPars), axis=1)

#         # Sizes.
#         self.Nx = Nx
#         self.Nu = Nu
#         self.Ny = Ny
#         self.Nmdist = distPars.shape[1]
#         self.Necon = econPars.shape[1]

#         # Setup lists to save data.
#         self.ps = ps
#         self.xs, self.us = [xs], [us]
#         self.x0, self.useq = [], []

#         # Get the initial linearized model. 
#         self.A, self.B, self.Bp, self.C = self._getLinearizedModel(xs, us, ps)

#         # Setup steady state optimizer.
#         self.tssOptFreq = tssOptFreq
#         self.ulb = ulb
#         self.uub = uub
#         self.RtoOptimizer = RTOOptimizer(rto_type=rto_type, fxup=fxup, hx=hx, 
#                                          lyup=lyup, Nx=Nx, Nu=Nu, 
#                                          Necon=self.Necon, 
#                                          ulb=ulb, uub=uub, xguess=xs, 
#                                          uguess=us, picnn_lyup=picnn_lyup, 
#                                          picnn_parids=picnn_parids)

#         # MPC Regulator parameters.
#         self.Q = Q
#         self.R = R
#         self.S = S
#         self.uprev = np.zeros((Nu, 1))
#         self.Nmpc = Nmpc
#         self._setupRegulator()

#         # Setup the Kalman filter.
#         B = np.concatenate((self.B, self.Bp), axis=1)
#         self.estimator = KalmanFilter(A=self.A, B=B, C=self.C, Qw=Qw, 
#                                       Rv=Rv, xPrior=np.zeros((Nx, 1)))

#         # Parameters to save.
#         self.computationTimes = []

#     def _getLinearizedModel(self, xs, us, ps):
#         """ Get the linearized model matrices. """

#         # Get the linearized model matrices A and B.
#         fxup = mpc.getCasadiFunc(self.fxup, [self.Nx, self.Nu, self.Nmdist], 
#                                  ["x", "u", "p"])
#         linModel = mpc.util.getLinearizedModel(fxup, [xs, us, ps], 
#                                                ["A", "B", "Bp"])
#         A, B, Bp = linModel["A"], linModel["B"], linModel["Bp"]
        
#         # Get the output matrix C.
#         hx = mpc.getCasadiFunc(self.hx, [self.Nx], ["x"])
#         linModel = mpc.util.getLinearizedModel(hx, [xs], ["C"])
#         C = linModel["C"]

#         # Return.
#         return A, B, Bp, C

#     def _setupRegulator(self):
#         """ Setup the Dense QP regulator. """

#         # Get a few more parameters to setup the Dense QP.
#         A, B, Q, R, M = getAugMatricesForROCPenalty(self.A, self.B, 
#                                                     self.Q, self.R, self.S)

#         # Get input constraints in deviation variables for the regulator.
#         xs, us = self.xs[-1], self.us[-1]
#         ulb, uub = self.ulb - us, self.uub - us
#         ulb = ulb[:, np.newaxis]
#         uub = uub[:, np.newaxis]

#         # Setup the regulator.
#         self.regulator  = DenseQPRegulator(A=A, B=B, Q=Q, R=R, M=M, 
#                                            N=self.Nmpc, ulb=ulb, uub=uub)

#         # Solve for the initial steady state.
#         x0 = np.zeros((self.Nx + self.Nu, 1))
#         useq = self.regulator.solve(x0)
#         useq += np.tile(us[:, np.newaxis], (self.Nmpc, 1))
#         useq = np.reshape(useq, (self.Nmpc, self.Nu))

#         # Save data.
#         self.x0 += [xs]
#         self.useq += [useq]

#     def control_law(self, simt, y):
#         """
#         Takes the measurement and the previous control input
#         and compute the current control input.
#         """
#         tstart = time.time()

#         # Solve steady-state optimization if needed.
#         if simt % self.tssOptFreq == 0:
            
#             # Get the steady state as computed by the RTO.
#             pdist = self.empcPars[simt:simt+1, :self.Nmdist].T
#             pecon = self.empcPars[simt:simt+1, self.Nmdist:].T
#             xs, us = self.RtoOptimizer.getOptimum(pecon, pdist)

#             # Get the updated linear model.
#             ps = pdist
#             (self.A, self.B, 
#              self.Bp, self.C) = self._getLinearizedModel(xs, us, ps)

#             # Update regulator parameters.
#             # Constraints.
#             ulb, uub = self.ulb - us, self.uub - us
#             self.regulator.ulb = ulb[:, np.newaxis]
#             self.regulator.uub = uub[:, np.newaxis]
#             A, B, Q, R, M = getAugMatricesForROCPenalty(self.A, self.B, self.Q, 
#                                                         self.R, self.S)
#             self.regulator._updateDynModel(A, B)

#             # Update model used by Kalman Filter.
#             B = np.concatenate((self.B, self.Bp), axis=1)
#             self.estimator._updateDynModel(self.A, B, self.C)

#         else:

#             # If a fresh optimization is not solved. Then just use the 
#             # previous targets.
#             xs, us = self.xs[-1], self.us[-1]

#         # State estimation.
#         up = np.concatenate((self.uprev, np.zeros((self.Nmdist, 1))))
#         ys = self.C @ xs[:, np.newaxis]
#         xhat =  self.estimator.solve(y - ys, up)

#         # Regulation.
#         x0 = np.concatenate((xhat, self.uprev))
#         useq = self.regulator.solve(x0)
#         useq += np.tile(us[:, np.newaxis], (self.Nmpc, 1))
#         useq = np.reshape(useq, (self.Nmpc, self.Nu))

#         # Save Uprev.
#         u = useq[:1, :].T
#         self.uprev = useq[:1, :].T - us[:, np.newaxis]

#         # Get computation times.
#         tend = time.time()
#         self.computationTimes.append(tend - tstart)

#         # Save data. 
#         self._saveData(xhat, useq, xs, us)

#         # Return.
#         return u

#     def _saveData(self, xhat, useq, xs, us):
#         """ Save data to lists. """

#         self.x0 += [xhat]
#         self.useq += [useq]
#         self.xs += [xs]
#         self.us += [us]

def get_model(*, ode, parameters, plant=True):
    """ Return a nonlinear plant simulator object."""

    # Some parameters.
    Nx, Nu = parameters['Nx'], parameters['Nu']
    Np, Ny = parameters['Np'], parameters['Ny']
    yindices = parameters['yindices']
    xs = parameters['xs'][:, np.newaxis]
    Delta = parameters['Delta']

    # Function for ODE and measurement.
    ode_func = lambda x, u, p: ode(x, u, p, parameters)
    meas_func = lambda x: x[yindices]

    # Measurement noise.
    if plant:
        Rv = parameters['Rv']
    else:
        Rv = 0*np.eye(Ny)

    # Return a simulator object.
    return NonlinearPlantSimulator(fxup=ode_func, hx=meas_func,
                                   Rv=Rv, Nx=Nx, Nu=Nu, Np=Np, Ny=Ny,
                                   sample_time=Delta, x0=xs)

# def arrayToMatrix(*arrays):
#     """Convert nummpy arrays to cvxopt matrices."""
#     matrices = []
#     for array in arrays:
#         matrices += [cvx.matrix(array)]
#     return tuple(matrices)

# def dlqr(A, B, Q, R, M=None):
#     """
#     Get the discrete-time LQR for the given system.
#     Stage costs are
#         x'Qx + 2*x'Mu + u'Ru
#     with M = 0 if not provided.
#     """
#     # For M != 0, we can simply redefine A and Q to give a problem with M = 0.
#     if M is not None:
#         RinvMT = scipy.linalg.solve(R,M.T)
#         Atilde = A - B.dot(RinvMT)
#         Qtilde = Q - M.dot(RinvMT)
#     else:
#         Atilde = A
#         Qtilde = Q
#         M = np.zeros(B.shape)
#     Pi = scipy.linalg.solve_discrete_are(Atilde,B,Qtilde,R)
#     K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A) + M.T)
#     return (K, Pi)

# def dlqe(A, C, Q, R):
#     """
#     Get the discrete-time Kalman filter for the given system.
#     """
#     P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
#     L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T
#     return (L, P)

# def c2d(A, B, Delta):
#     """ Custom c2d function for linear systems."""
    
#     # First construct the incumbent matrix
#     # to take the exponential.
#     (Nx, Nu) = B.shape
#     M1 = np.concatenate((A, B), axis=1)
#     M2 = np.zeros((Nu, Nx+Nu))
#     M = np.concatenate((M1, M2), axis=0)
#     Mexp = scipy.linalg.expm(M*Delta)

#     # Return the extracted matrices.
#     Ad = Mexp[:Nx, :Nx]
#     Bd = Mexp[:Nx, -Nu:]
#     return (Ad, Bd)

def c2dNonlin(f, Delta, p=False):
    """ Quick function to 
        convert a ode to discrete
        time using the RK4 method.
        
        fxup is a function such that 
        dx/dt = f(x, u, p)
        assume zero-order hold on u and p.
    """
    if p:

        # Get k1, k2, k3, k4.
        k1 = f
        k2 = lambda x, u, p: f(x + Delta*(k1(x, u, p)/2), u, p)
        k3 = lambda x, u, p: f(x + Delta*(k2(x, u, p)/2), u, p)
        k4 = lambda x, u, p: f(x + Delta*k3(x, u, p), u, p)

        # Final discrete time function.
        xplus = lambda x, u, p: x + (Delta/6)*(k1(x, u, p) + 2*k2(x, u, p) +
                                               2*k3(x, u, p) + k4(x, u, p))

    else:
        
        # Get k1, k2, k3, k4.
        k1 = f
        k2 = lambda x, u: f(x + Delta*(k1(x, u)/2), u)
        k3 = lambda x, u: f(x + Delta*(k2(x, u)/2), u)
        k4 = lambda x, u: f(x + Delta*k3(x, u), u)

        # Final discrete time function.
        xplus = lambda x, u: x + (Delta/6)*(k1(x, u) + 2*k2(x, u) + 
                                            2*k3(x, u) + k4(x, u))

    # Return.
    return xplus

# def eigvalEigvecTest(X, Y):
#     """Return True if an eigenvector of X corresponding to 
#     an eigenvalue of magnitude greater than or equal to 1
#     is not in the nullspace of Y.
#     Else Return False."""
#     (eigvals, eigvecs) = np.linalg.eig(X)
#     eigvecs = eigvecs[:, np.absolute(eigvals)>=1.]
#     for eigvec in eigvecs.T:
#         if np.linalg.norm(Y @ eigvec)<=1e-8:
#             return False
#     else:
#         return True

# def assertDetectable(A, C):
#     """Assert if the provided (A, C) pair is detectable."""
#     assert eigvalEigvecTest(A, C)

# def assertStabilizable(A, B):
#     """Assert if the provided (A, B) pair is stabilizable."""
#     assert eigvalEigvecTest(A.T, B.T)

# def getAugMatricesForROCPenalty(A, B, Q, R, S):
#     """ Get the Augmented A, B, C, and the noise covariance matrix."""

#     # Get the shapes.
#     Nx, Nu = B.shape

#     # Augmented A.
#     Aaug1 = np.concatenate((A, np.zeros((Nx, Nu))), axis=1)
#     Aaug2 = np.zeros((Nu, Nx+Nu))
#     Aaug = np.concatenate((Aaug1, Aaug2), axis=0)

#     # Augmented B.
#     Baug = np.concatenate((B, np.eye((Nu))), axis=0)

#     # Augmented Q.
#     Qaug = scipy.linalg.block_diag(Q, S)

#     # Augmented R.
#     Raug = R + S

#     # Augmented M.
#     Maug = np.concatenate((np.zeros((Nx, Nu)), -S), axis=0)

#     # Return augmented matrices.
#     return (Aaug, Baug, Qaug, Raug, Maug)

# class KalmanFilter:

#     def __init__(self, *, A, B, C, Qw, Rv, xPrior):
#         """ Class to construct and perform state estimation
#             using Kalman Filtering.
#         """

#         # Store the matrices.
#         self.A = A
#         self.B = B
#         self.C = C
#         self.Qw = Qw
#         self.Rv = Rv
        
#         # Compute the kalman filter gain.
#         self.L, _ = dlqe(self.A, self.C, self.Qw, self.Rv)
        
#         # Create lists for saving data. 
#         self.xhat = [xPrior]
#         self.xhatPred = []
#         self.y = []
#         self.uprev = []

#     def _updateDynModel(self, A, B, C):
#         """ Update the Dynamic model used for state 
#             estimation. """
#         self.A = A
#         self.B = B
#         self.C = C
#         self.L, _ = dlqe(self.A, self.C, self.Qw, self.Rv)

#     def solve(self, y, uprev):
#         """ Take a new measurement and do 
#             the prediction and filtering steps."""
        
#         # Prediction.
#         xhat = self.xhat[-1]
#         xhatPred = self.A @ xhat + self.B @ uprev

#         # Filtering.
#         xhat = xhatPred + self.L @ (y - self.C @ xhatPred)

#         # Save data.
#         self._saveData(xhat, xhatPred, y, uprev)

#         # Return state estimate.
#         return xhat
        
#     def _saveData(self, xhat, xhatPred, y, uprev):
#         """ Save the state estimates,
#             Can be used for plotting later. """

#         self.xhat += [xhat]
#         self.xhatPred += [xhatPred]
#         self.y += [y]
#         self.uprev += [uprev]

# class DenseQPRegulator:
#     """ Class to construct and solve the linear MPC regulator QP
#         using a dense QP formulation.

#         The problem:

#         V(x0, \mathbf{u}) = (1/2) \sum_{k=0}^{N-1} [x(k)'Qx(k) + u(k)'Ru(k) + 2x(k)'Mu(k)] + (1/2)x(N)'Pfx(N)
        
#         subject to:
#             x(k+1) = Ax(k) + Bu(k)
#             ulb <= u(k) <= uub

#         This class eliminates all the states
#         from the set of decision variables and 
#         solves a dense formulation of the QP. 
#     """
#     def __init__(self, *, A, B, Q, R, M, N, ulb, uub):

#         # Model.
#         self.A, self.B = A, B

#         # Penalty matrices.
#         self.Q, self.R, self.M = Q, R, M

#         # Horizon length.
#         self.N = N

#         # Input constraints.
#         self.ulb, self.uub = ulb, uub

#         # Set sizes.
#         self.Nx, self.Nu = B.shape

#         # Get the LQR gain, only used for reparameterization for the dense QP.
#         self.Krep, self.Pf = dlqr(A, B, Q, R, M)

#         # Reparametrie QP if needed and then setup QP matrices.
#         self._reparameterize()
#         self._setupFixedMatrices()

#         # Create lists to save data.
#         self.x0, self.useq = [], []

#     def _updateDynModel(self, A, B):
#         """ Update linear model. """

#         self.A, self.B = A, B
#         self.Krep, self.Pf = dlqr(A, B, self.Q, self.R, self.M)
#         self._reparameterize()
#         self._setupFixedMatrices()

#     def _reparameterize(self):
#         """
#         Reparameterize A, B, Q, R, M, and G, h.
#         A = A+BK, B = B, Q = Q+K'RK+MK+K'M', R=R
#         M = K'R + M
#         Pf is the solution of the Riccati Eqn using the 
#         new parametrized matrices. 
#         """
#         A, B, Q, R, M, Krep = self.A, self.B, self.Q, self.R, self.M, self.Krep
#         (eigvals, _) = np.linalg.eig(A)
#         if any(np.absolute(eigvals)>=1.):
#             self.A = A + B @ Krep
#             self.Q = Q + Krep.T @ (R @ Krep) + M @ Krep + Krep.T @ M.T
#             self.M = Krep.T @ R + M
#             self.reParamet = True
#         else:
#             self.reParamet = False

#     def _setupFixedMatrices(self):
#         """" Setup the fixed matrices which don't change in 
#              real-time.
            
#              Finally the QP should be in this format.
#              min_x  (1/2)x'*P*x + q'*x
#              s.t     Aineq*x <= Bineq
#                      Aeq*x = beq
             
#              in which, x = [u(0), u(1), ..., u(N-1)]
#         """

#         # Get penalty matrices.
#         tA, tB = self._get_tA_tB()
#         tQ, tR, tM, tK = self._get_tQ_tR_tM_tK()
#         P, q = self._getPq(tA, tB, tQ, tR, tM)

#         # Get inequality matrices.
#         Aineq, Bineq1, Bineq2 = self._getAineqBineq(tA, tB, tK)

#         # Assign computed matrices to attributes.
#         self.P, self.q = P, q
#         self.Aineq, self.Bineq1, self.Bineq2 = Aineq, Bineq1, Bineq2
#         self.tA, self.tB, self.tK = tA, tB, tK

#     def _get_tA_tB(self):
#         """ Get the Atilde and Btilde.
#             N is the number of state outputs is asked for. 
        
#         xseq = tA*x0 + tB*useq
#         xseq = (N+1)*Nx, useq = N*Nu
#         tA = [(N+1)*Nx]*Nx, tB = [(N+1)*Nx]*[N*Nu]

#         tA = [I;
#               A;
#               A^2;
#               .;
#               A^N]
#         tB = [0, 0, 0, ..;
#               B, 0, 0, ...;
#               AB, B, 0, ...;
#               ...;
#               A^(N-1)B, A^(N-2)B, ..., B]
#         """

#         # Extract attributes.
#         A, N = self.A, self.N

#         # Get tA.
#         tA = np.concatenate([np.linalg.matrix_power(A, i) for i in range(N+1)])

#         # Get tB.
#         tB = np.concatenate([self._get_tBRow(i) for i in range(N+1)])

#         # Return.
#         return (tA, tB)
    
#     def _get_tBRow(self, i):
#         """ Returns the ith row of tB. """

#         # Extract attributes.
#         A, B, N = self.A, self.B, self.N
#         (Nx, Nu) = B.shape
        
#         # Get tBi.
#         tBi = [np.linalg.matrix_power(A, i-j-1) @ B 
#                 if j<i 
#                 else np.zeros((Nx, Nu))
#                 for j in range(N)]
#         tBi = np.concatenate(tBi, axis=1)
        
#         # Return.
#         return tBi

#     def _get_tQ_tR_tM_tK(self):
#         """ Get the block diagonal matrices for dense QP. 

#         tQ = [Q, 0, 0, ...
#               0, Q, 0, ...
#               0, 0, Q, ...
#               ...
#               0, 0, 0, ..., Pf]

#         tR = [R, 0, 0, ...
#               0, R, 0, ...
#               0, 0, R, ...
#               ...
#               0, 0, 0, ... R]

#         tM = [M, 0, 0, ...
#               0, M, 0, ...
#               0, 0, M, ...
#               ...
#               0, 0, 0, ..., M
#               0, 0, 0, ......]

#         tK = [K, 0, 0, ......,
#               0, K, 0, ......,
#               0, 0, K, 0, ...,
#               ...............,
#               0, 0, 0, ......K]

#         Convert 

#         V = (1/2) \sum_{k=0}^{N-1} [x(k)'Qx(k) + u(k)'Ru(k) + 2x(k)'Mu(k)] 
#         V += (1/2)x(N)'Pfx(N)
        

#         to 

#         V = (1/2)*[xseq'*tQ*xseq + useq'*tR*useq + 2*xseq'*tM*useq]

#         in which, xseq = [x(0), x(1), x(2), ...., x(N)]
#                   useq = [u(0), u(1), u(2), ...., u(N-1)] """

#         # Extract attributes.
#         Q, R, M, Krep, Pf = self.Q, self.R, self.M, self.Krep, self.Pf
#         Nx, Nu, N = self.Nx, self.Nu, self.N

#         # Construct matrices.
#         # tQ
#         tQ = scipy.linalg.block_diag(*[Q if i<N else Pf for i in range(N+1)])

#         # tR
#         tR = scipy.linalg.block_diag(*[R for _ in range(N)])

#         # tM
#         tM = scipy.linalg.block_diag(*[M for _ in range(N)])
#         tM = np.concatenate((tM, np.zeros((Nx, N*Nu))), axis=0)

#         # tK.
#         tK = scipy.linalg.block_diag(*[Krep for _ in range(N)])

#         # Return.
#         return (tQ, tR, tM, tK)

#     def _getPq(self, tA, tB, tQ, tR, tM):
#         """ Get the penalites for solving the QP.
#             P = tB'*tQ*tB + tR + tB'*tM + tM'*tB 
#             tq = (tB'*Q + M)*tA
#         """

#         # Get P and q matrices. 
#         P = tB.T @ (tQ @ tB) + tR + tB.T @ tM + tM.T @ tB
#         q = (tB.T @ tQ  + tM.T) @ tA

#         # Return.
#         return (P, q)

#     def _getAineqBineq(self, tA, tB, tK):
#         """ Get the inequality matrices. """

#         # Extract some parameters.
#         Nx, Nu, N = self.Nx, self.Nu, self.N
#         ulb, uub = self.ulb, self.uub
#         Krep = self.Krep

#         # Aineq.
#         Auineq = np.concatenate((np.eye(Nu), -np.eye(Nu)), axis=0)
#         Auineq = scipy.linalg.block_diag(*[Auineq for _ in range(N)])
#         if self.reParamet:
#             Aineq = Auineq @ (tK @ tB[:N*Nx, :]) + Auineq
#         else:
#             Aineq = Auineq

#         # Get Bineq1 and Bineq2.
#         Bineq1 = np.concatenate((uub, -ulb), axis=0)
#         Bineq1 = np.concatenate([Bineq1 for _ in range(N)])

#         # If the QP is a reparametrized QP, get concatenations of K as well.
#         if self.reParamet:
#             Bineq2 = -Auineq @ (tK @ tA[:N*Nx, :])
#         else:
#             Bineq2 = np.zeros((2*Nu*N, Nx))

#         # Return inequality matrices. 
#         return (Aineq, Bineq1, Bineq2)

#     def _getQPMatrices(self, x0):
#         """ Get the RHS of the equality constraint."""

#         # Get Inequality constraints.
#         Aineq = self.Aineq
#         Bineq = self.Bineq1 + self.Bineq2 @ x0
        
#         # Get penalty matrices.
#         P = self.P
#         q = self.q @ x0

#         # Return QP matrices.
#         return (P, q, Aineq, Bineq)

#     def solve(self, x0):
#         """Setup and the solve the dense QP, output is 
#         the first element of the sequence.
#         If the problem is reparametrized, go back to original 
#         input variable. 
#         """
#         # Get QP matrices.
#         qpMatrices = self._getQPMatrices(x0)

#         # Solve.
#         solution = cvx.solvers.qp(*arrayToMatrix(*qpMatrices))
        
#         # Extract input sequence.
#         useq = np.asarray(solution['x'])
#         if self.reParamet:
            
#             # Extract attributes.
#             tA, tB, tK = self.tA, self.tB, self.tK
#             N, Nx = self.N, self.Nx

#             # Get useq in original space.
#             useq = useq + tK @ (tA[:N*Nx, :] @ x0 + tB[:N*Nx, :] @ useq)

#         # Save data.
#         self._saveData(x0, useq)

#         # Return optimal input sequence.
#         return useq

#     def _saveData(self, x0, useq):
#         # Save data.
#         self.x0.append(x0)
#         self.useq.append(useq)

# def online_simulation(plant, controller, *, Nsim=None,
#                       disturbances=None, stdout_filename=None):
#     """ Online simulation with either the RTO controller
#         or the nonlinear economic MPC controller. """

#     sys.stdout = open(stdout_filename, 'w')
#     measurement = plant.y[0] # Get the latest plant measurement.
#     disturbances = disturbances[..., np.newaxis]
#     avgStageCosts = [0*np.eye(1)]

#     # Start simulation loop.
#     for (simt, disturbance) in zip(range(Nsim), disturbances):

#         # Print simulation timestep.
#         print("Simulation Step:" + f"{simt}")

#         # Get the control input.
#         control_input = controller.control_law(simt, measurement)

#         # Print computation time.
#         print("Computation time:" + str(controller.computationTimes[-1]))

#         # Compute the stage cost.
#         stageCost = controller.lyup(measurement, control_input, 
#                         controller.empcPars[simt:simt+1, -controller.Necon:].T)
#         avgStageCosts += [(avgStageCosts[-1]*simt + stageCost)/(simt + 1)]

#         # Inject control/disturbance to the plant.
#         measurement = plant.step(control_input, disturbance)

#     # Create a sim data object and get the stage cost array.
#     clData = SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
#                 x=np.asarray(plant.x[0:-1]).squeeze(),
#                 u=np.asarray(plant.u),
#                 y=np.asarray(plant.y[0:-1]).squeeze())
#     avgStageCosts = np.array(avgStageCosts[1:]).squeeze()

#     # Return.
#     return clData, avgStageCosts

def getSSOptimum(*, fxu, hx, lyu, parameters, guess):
    """ Setup and solve the steady state optimization. """

    # Size and constraints.
    Nx, Nu = parameters['Nx'], parameters['Nu']
    ulb, uub = parameters['ulb'], parameters['uub']

    # Construct NLP and solve.
    xs = casadi.SX.sym('xs', Nx)
    us = casadi.SX.sym('us', Nu)

    # Get casadi functions.
    lxu_func = lambda x, u: lyu(hx(x), u)
    lxu = mpc.getCasadiFunc(lxu_func, [Nx, Nu], ["x", "u"])
    f = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

    # Setup NLP.
    nlpInfo = dict(x=casadi.vertcat(xs, us), f=lxu(xs, us),
               g=casadi.vertcat(xs -  f(xs, us), us))
    nlp = casadi.nlpsol('nlp', 'ipopt', nlpInfo)

    # Make a guess and get the constraint limits.
    xuguess = np.concatenate((guess['x'], guess['u']))[:, np.newaxis]
    lbg = np.concatenate((np.zeros((Nx,)), ulb))[:, np.newaxis]
    ubg = np.concatenate((np.zeros((Nx,)), uub))[:, np.newaxis]

    # Solve.
    nlp_soln = nlp(x0=xuguess, lbg=lbg, ubg=ubg)
    xsol = np.asarray(nlp_soln['x'])[:, 0]
    opt_sscost = np.asarray(nlp_soln['f'])
    xs, us = np.split(xsol, [Nx])
    ys = hx(xs)

    # Return the steady state solution.
    return xs, us, ys, opt_sscost

def getXsSscost(*, fxu, hx, lyu, us, parameters, 
                   xguess=None, lbx=None, ubx=None):
    """ Setup and solve the steady state optimization. """

    # Sizes.
    Nx, Nu = parameters['Nx'], parameters['Nu']

    # Initial Guess.
    if xguess is None:
        xguess = np.zeros((Nx, 1))

    # Decision variable.
    xs = casadi.SX.sym('xs', Nx)

    # Model as a casadi function.
    fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

    # Constraints.
    g = xs - fxu(xs, us)
    lbg = np.zeros((Nx, 1))
    ubg = lbg
    if lbx is not None and ubx is not None:
        lbx, ubx = lbx[:, np.newaxis], ubx[:, np.newaxis]
    else:
        lbx = np.tile(-np.inf, (Nx, 1))
        ubx = np.tile(np.inf, (Nx, 1))

    # Setup dummy NLP.
    nlp = dict(x=xs, f=1, g=g)
    nlp = casadi.nlpsol('nlp', 'ipopt', nlp)

    # Solve.
    nlp_soln = nlp(x0=xguess, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    xs = np.asarray(nlp_soln['x'])[:, 0]

    # Compute the cost based on steady state.
    sscost = lyu(hx(xs), us)
    
    # Return the steady state cost.
    return xs, sscost