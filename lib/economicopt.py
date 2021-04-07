# [depends] hybridid.py
"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import casadi
import mpctools as mpc
from hybridid import SimData

def online_simulation(plant, controller, *, plant_lyup, Nsim=None,
                      disturbances=None, stdout_filename=None):
    """ Online simulation with either the RTO controller
        or the nonlinear economic MPC controller. """

    sys.stdout = open(stdout_filename, 'w')
    measurement = plant.y[0] # Get the latest plant measurement.
    disturbances = disturbances[..., np.newaxis]
    avgStageCosts = [0.]

    # Start simulation loop.
    for (simt, disturbance) in zip(range(Nsim), disturbances):

        # Compute the control and the current stage cost.
        print("Simulation Step:" + f"{simt}")
        control_input = controller.control_law(simt, measurement)
        print("Computation time:" + str(controller.computationTimes[-1]))

        stageCost = plant_lyup(plant.y[-1], control_input, 
                                controller.empcPars[simt:simt+1, :].T)[0]
        avgStageCosts += [(avgStageCosts[-1]*simt + stageCost)/(simt+1)]

        # Inject control/disturbance to the plant.
        measurement = plant.step(control_input, disturbance)

    # Create a sim data/stage cost array.
    clData = SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u),
                y=np.asarray(plant.y[0:-1]).squeeze())
    avgStageCosts = np.array(avgStageCosts[1:])

    # Return.
    return clData, avgStageCosts

def get_kooppars_fxu_hx(*, train, parameters):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    Np = train['Np']
    fN_weights = train['trained_weights'][-1][:-2]
    A = train['trained_weights'][-1][-2].T
    B = train['trained_weights'][-1][-1].T
    xuyscales = train['xuyscales']
    Ny, Nu = parameters['Ny'], parameters['Nu']
    Nx = Ny + Np*(Ny + Nu) + train['fN_dims'][-1]
    ulb, uub = parameters['ulb'], parameters['uub']
    koop_pars = dict(Nx=Nx, Ny=Ny, Nu=Nu, Np=Np, xuyscales=xuyscales,
                     fN_weights=fN_weights, ulb=ulb, uub=uub, A=A, B=B)
    
    # Get function handles.
    fxu = lambda x, u: koop_fxu(x, u, koop_pars)
    hx = lambda x: koop_hx(x, koop_pars)

    # Return.
    return koop_pars, fxu, hx

def koop_fxu(xkp, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        xkp^+ = A*xkp + Bu """
    
    # Get A, B matrices.
    A, B = parameters['A'], parameters['B']
    umean, ustd = parameters['xuyscales']['uscale']

    # Scale control input.
    u = (u - umean)/ustd

    # Add extra axis.
    xkp, u = xkp[:, np.newaxis], u[:, np.newaxis]

    # Get current output.
    xkplus = A @ xkp + B @ u

    # Remove an axis.
    xkplus = xkplus[:, 0]

    # Return the sum.
    return xkplus

def koop_hx(xkp, parameters):
    """ Measurement function. """
    
    # Extract a few parameters.
    Ny = parameters['Ny']
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    
    # Add extra axis.
    y = xkp[:Ny]*ystd + ymean

    # Return the sum.
    return y

def get_koopman_ss_xkp0(train, parameters):

    # Get initial state.
    Np = train['Np']
    us = parameters['us']
    yindices = parameters['yindices']
    ys = parameters['xs'][yindices]
    yz0 = np.concatenate((np.tile(ys, (Np+1, )), 
                          np.tile(us, (Np, ))))

    # Scale initial state and get the lifted state.
    fN_weights = train['trained_weights'][-1][:-2]
    xuyscales = train['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    yzmean = np.concatenate((np.tile(ymean, (Np+1, )), 
                            np.tile(umean, (Np, ))))
    yzstd = np.concatenate((np.tile(ystd, (Np+1, )), 
                            np.tile(ustd, (Np, ))))
    yz0 = (yz0 - yzmean)/yzstd
    xkp0 = np.concatenate((yz0, fnn(yz0, fN_weights, 1.)))

    # Return.
    return xkp0

def get_ss_optimum(*, fxu, hx, lyu, parameters, guess):
    """ Setup and solve the steady state optimization. """

    Nx, Nu = parameters['Nx'], parameters['Nu']
    ulb, uub = parameters['ulb'], parameters['uub']

    # Construct NLP and solve.
    xs = casadi.SX.sym('xs', Nx)
    us = casadi.SX.sym('us', Nu)

    # Get casadi functions.
    lyu_func = lambda x, u: lyu(hx(x), u)
    lyu = mpc.getCasadiFunc(lyu_func, [Nx, Nu], ["x", "u"])
    fxu = mpc.getCasadiFunc(fxu, [Nx, Nu], ["x", "u"])

    # Setup NLP.
    nlp = dict(x=casadi.vertcat(xs, us), f=lyu(xs, us),
               g=casadi.vertcat(xs -  fxu(xs, us), us))
    nlp = casadi.nlpsol('nlp', 'ipopt', nlp)

    # Make a guess, get constraint limits.
    xuguess = np.concatenate((guess['x'], guess['u']))[:, np.newaxis]
    lbg = np.concatenate((np.zeros((Nx,)), ulb))[:, np.newaxis]
    ubg = np.concatenate((np.zeros((Nx,)), uub))[:, np.newaxis]

    # Solve.
    nlp_soln = nlp(x0=xuguess, lbg=lbg, ubg=ubg)
    xsol = np.asarray(nlp_soln['x'])[:, 0]
    xs, us = np.split(xsol, [Nx])
    ys = hx(xs)

    # Return the steady state solution.
    return xs, us, ys

def get_sscost(*, fxu, hx, lyu, us, parameters):
    """ Setup and solve the steady state optimization. """

    # Get the sizes and actuator bounds.
    Nx, Nu = parameters['Nx'], parameters['Nu']
    ulb, uub = parameters['ulb'], parameters['uub']

    # Get resf casadi function.
    xs = casadi.SX.sym('xs', Nx)
    resfx = mpc.getCasadiFunc(lambda x: -x + fxu(x, us), [Nx], ["x"])

    # Use rootfinder to get the SS.
    rootfinder = casadi.rootfinder('resfx', 'newton', resfx)
    xguess = np.zeros((Nx, 1))
    xs = np.asarray(rootfinder(xguess))[:, 0]

    # Setup NLP.
    sscost = lyu(hx(xs), us)[0]

    # Return the steady state cost.
    return sscost

def get_xuguess(*, model_type, plant_pars, Np=None, Nx=None):
    """ Get x, u guess depending on model type. """
    us = plant_pars['us']
    if model_type == 'plant':
        xs = plant_pars['xs']
    elif model_type == 'grey-box':
        gb_indices = plant_pars['gb_indices']
        xs = plant_pars['xs'][gb_indices]
    elif model_type == 'black-box':
        yindices = plant_pars['yindices']
        ys = plant_pars['xs'][yindices]
        xs = np.concatenate((np.tile(ys, (Np, )), 
                             np.tile(us, (Np, ))))
    elif model_type == 'Koopman':
        xs = np.zeros((Nx, ))
    else:
        pass
    # Return as dict.
    return dict(x=xs, u=us)

def c2dNonlin(fxu, Delta):
    """ Quick function to 
        convert a ode to discrete
        time using the RK4 method.
        
        fxu is a function such that 
        dx/dt = f(x, u)
        assume zero-order hold on the input.
    """
    # Get k1, k2, k3, k4.
    k1 = fxu
    k2 = lambda x, u: fxu(x + Delta*(k1(x, u)/2), u)
    k3 = lambda x, u: fxu(x + Delta*(k2(x, u)/2), u)
    k4 = lambda x, u: fxu(x + Delta*k3(x, u), u)
    # Final discrete time function.
    xplus = lambda x, u: x + (Delta/6)*(k1(x, u) + 
                                        2*k2(x, u) + 2*k3(x, u) + k4(x, u))
    return xplus