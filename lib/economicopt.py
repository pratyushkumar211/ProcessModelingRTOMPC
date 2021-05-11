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

def get_sscost(*, fxu, hx, lyu, us, parameters, xguess=None):
    """ Setup and solve the steady state optimization. """

    # Get the sizes and actuator bounds.
    Nx, Nu = parameters['Nx'], parameters['Nu']
    if xguess is None:
        xguess = np.zeros((Nx, 1))

    # Get resf casadi function.
    xs = casadi.SX.sym('xs', Nx)
    resfx = mpc.getCasadiFunc(lambda x: -x + fxu(x, us), [Nx], ["x"])

    # Use rootfinder to get the SS.
    rootfinder = casadi.rootfinder('resfx', 'newton', resfx)
    xs = np.asarray(rootfinder(xguess))[:, 0]

    # Compute the cost based on steady state.
    sscost = lyu(hx(xs), us)

    # Return the steady state cost.
    return sscost

def get_xuguess(*, model_type, plant_pars, Np=None):
    """ Get x, u guess depending on model type. """
    us = plant_pars['us']
    if model_type == 'Plant' or model_type == 'Hybrid':
        xs = plant_pars['xs']
    elif model_type == 'Black-Box-NN':
        yindices = plant_pars['yindices']
        ys = plant_pars['xs'][yindices]
        us = np.array([5., 8.])
        xs = np.concatenate((np.tile(ys, (Np+1, )), 
                             np.tile(us, (Np, ))))
    elif model_type == 'ICNN':
        us = np.array([5., 2.])
        xs = None
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