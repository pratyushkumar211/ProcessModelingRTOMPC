"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf
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

def get_bb_parameters(*, Np, xuyscales, hN_weights, parameters):
    """ Collect the black-box neural network parameters in 
        a dictionary. """
    
    Ny, Nu = parameters['Ny'], parameters['Nu']
    Nx = Np*(Ny + Nu)
    # Return dict.
    return dict(Nx=Nx, Ny=Ny, Nu=Nu, xuyscales=xuyscales,
                hN_weights=hN_weights)

def bb_fxu(z, u, parameters):
    """ Function describing the dynamics 
        of the black-box neural network. 
        z^+ = f_z(z, u) """

    # Extract a few parameters.
    Np = parameters['Np']
    Ny = parameters['Ny']
    Nu = parameters['Nu']
    hN_weights = parameters['hN_weights']
    xuscales = parameters['xuscales']
    ymean, ystd = xuscales['yscale']
    umean, ustd = xuscales['uscale']
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
    zplus = np.concatenate((z[Ny:], y, u[Nu:], u))

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
    xuscales = parameters['xuscales']
    ymean, ystd = xuscales['yscale']
    umean, ustd = xuscales['uscale']
    zmean = np.concatenate((np.tile(ymean, (Np, )), 
                            np.tile(umean, (Np, ))))
    zstd = np.concatenate((np.tile(ystd, (Np, )), 
                           np.tile(ustd, (Np, ))))
    
    # Scale.
    z = (z - zmean)/zstd
    u = (u - umean)/ustd

    # Get current output.
    y = fnn(z, hN_weights)

    # Scale measurement back.
    y = y*ystd + ymean

    # Return the measurement.
    return y