"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
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
    avg_stage_costs = [0.]

    # Start simulation loop.
    for (simt, disturbance) in zip(range(Nsim), disturbances):

        # Compute the control and the current stage cost.
        print("Simulation Step:" + f"{simt}")
        control_input = controller.control_law(simt, measurement)
        print("Computation time:" + str(controller.computation_times[-1]))
        stage_cost = plant_lyup(plant.y[-1], control_input,
                                controller.opt_pars[simt:simt+1, :].T)[0]
        avg_stage_costs += [(avg_stage_costs[-1]*simt + stage_cost)/(simt+1)]

        # Inject control/disturbance to the plant.
        measurement = plant.step(control_input, disturbance)

    # Create a sim data/stage cost array.
    cl_data = SimData(t=np.asarray(plant.t[0:-1]).squeeze(),
                x=np.asarray(plant.x[0:-1]).squeeze(),
                u=np.asarray(plant.u),
                y=np.asarray(plant.y[0:-1]).squeeze())
    avg_stage_costs = np.array(avg_stage_costs[1:])
    openloop_sol = [np.asarray(controller.regulator.useq[0]), 
                    np.asarray(controller.regulator.xseq[0])]
    # Return.
    return cl_data, avg_stage_costs, openloop_sol

def fnn(nn_input, nn_weights):
    """ Compute the NN output. 
        Assume that the input is a vector with shape size 
        1, and return output with shape size 1.
        """
    nn_output = nn_input[:, np.newaxis]
    for i in range(0, len(nn_weights)-2, 2):
        (W, b) = nn_weights[i:i+2]
        nn_output = W.T @ nn_output + b[:, np.newaxis]
        nn_output = np.tanh(nn_output)
    (Wf, bf) = nn_weights[-2:]
    nn_output = (Wf.T @ nn_output + bf[:, np.newaxis])[:, 0]
    # Return.
    return nn_output

def get_bbpars_fxu_hx(*, train, parameters):
    """ Get the black-box parameter dict and function handles. """

    # Get black-box model parameters.
    Np = train['Np']
    hN_weights = train['trained_weights'][-1]
    xuyscales = train['xuyscales']
    bb_pars = get_bb_parameters(Np=Np, xuyscales=xuyscales, 
                                hN_weights=hN_weights, 
                                parameters=parameters)
    
    # Get function handles.
    fxu = lambda x, u: bb_fxu(x, u, bb_pars)
    hx = lambda x: bb_hx(x, bb_pars)

    # Return.
    return bb_pars, fxu, hx

def get_bb_parameters(*, Np, xuyscales, hN_weights, parameters):
    """ Collect the black-box neural network parameters in 
        a dictionary. """
    
    Ny, Nu = parameters['Ny'], parameters['Nu']
    Nx = Np*(Ny + Nu)
    ulb, uub = parameters['ulb'], parameters['uub']
    # Return dict.
    return dict(Nx=Nx, Ny=Ny, Nu=Nu, Np=Np, xuyscales=xuyscales,
                hN_weights=hN_weights, ulb=ulb, uub=uub)

def bb_fxu(z, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        z^+ = f_z(z, u) """

    # Extract a few parameters.
    Np = parameters['Np']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    hN_weights = parameters['hN_weights']
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                            np.tile(umean, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                           np.tile(ustd, (Np, ))))
    
    # Scale.
    z = (z - zmean)/zstd
    u = (u - umean)/ustd

    # Get current output.
    y = fnn(z, hN_weights)
    
    # Concatenate.
    zplus = np.concatenate((z[Ny:Np*Ny], y, z[-(Np-1)*Nu:], u))

    # Scale back.
    zplus = zplus*zstd + zmean

    # Return the sum.
    return zplus

def bb_hx(z, parameters):
    """ Measurement function. """
    
    # Extract a few parameters.
    Np = parameters['Np']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    hN_weights = parameters['hN_weights']
    xuyscales = parameters['xuyscales']
    ymean, ystd = xuyscales['yscale']
    umean, ustd = xuyscales['uscale']
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                            np.tile(umean, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                           np.tile(ustd, (Np, ))))
    
    # Scale.
    z = (z - zmean)/zstd

    # Get current output.
    y = fnn(z, hN_weights)

    # Scale measurement back.
    y = y*ystd + ymean

    # Return the measurement.
    return y

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

def get_xuguess(*, model_type, plant_pars, Np=None):
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