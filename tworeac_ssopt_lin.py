# [depends] %LIB%/hybridid.py tworeac_parameters_lin.pickle
# [makes] pickle
""" Script to use the trained hybrid model for 
    steady-state optimization.
    Pratyush Kumar, pratyushkumar@ucsb.edu """

import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridid import (PickleTool, PAPER_FIGSIZE)

def _plant_ys(*, us, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu = 0 (continuous time SS) 
    """
    # Get the parameters.
    k1 = parameters['k1']
    k2 = parameters['k2']
    k3 = parameters['k3']
    Ny = parameters['Ny']
    tau = parameters['ps'].squeeze()
    # Get the matrices and xs. 
    A = np.array([[-k1-(1/tau), 0., 0.], 
                  [k1, -k2 - (1/tau), k3], 
                  [0., k2, -k3 -(1/tau)]])
    B = np.array([[1/tau], [0.], [0.]]) 
    xs = -np.linalg.inv(A) @ (B @ us)
    # Return.
    return xs[0:Ny]

def _greybox_ys(*, us, parameters):
    """ Return the plant xs. 
        Solve: Ax + Bu = 0 (continuous time SS) 
    """
    # Get the parameters.
    k1 = parameters['k1']
    Ny = parameters['Ny']
    tau = parameters['ps'].squeeze()
    # Get the matrices and xs. 
    A = np.array([[-k1-(1/tau), 0.], 
                  [k1, -(1/tau)]])
    B = np.array([[1/tau], [0.]]) 
    xs = -np.linalg.inv(A) @ (B @ us)
    # Return.
    return xs[0:Ny]

def cost(ys, us):
    """ Compute the steady-state cost. """
    (cost, profit) = (100, 150)
    # Return.
    return cost*us - profit*ys[:, -1:]

def plot_profit_curve(*, us, costs, figure_size=PAPER_FIGSIZE, 
                         ylabel_xcoordinate=-0.12, 
                         left_label_frac=0.15):
    """ Plot the profit curves. """
    (figure, axes) = plt.subplots(nrows=1, ncols=1, 
                                        sharex=True, 
                                        figsize=figure_size, 
                                    gridspec_kw=dict(left=left_label_frac))
    xlabel = r'$C_{A0} \ (\textnormal{mol/m}^3)$'
    ylabel = r'Cost ($\$ $)'
    colors = ['b', 'g']
    legends = ['Plant', 'Grey-box']
    for (cost, color) in zip(costs, colors):
        # Plot the corresponding data.
        axes.plot(us, cost, color)
    axes.legend(legends)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, rotation=False)
    axes.get_yaxis().set_label_coords(ylabel_xcoordinate, 0.5) 
    axes.set_xlim([np.min(us), np.max(us)])
    # Return the figure object.
    return [figure]

def compute_cost_curves(*, parameters):
    """ Compute the profit curves for the three models. """
    (ulb, uub) = (parameters['lb']['u'], parameters['ub']['u'])
    uss = np.linspace(ulb, uub, 100)[:, np.newaxis]
    (ys_plant, ys_greybox) = ([], [])
    for us in uss:
        ys_plant.append(_plant_ys(us=us, parameters=parameters))
        ys_greybox.append(_greybox_ys(us=us, 
                                      parameters=parameters))
    ys_plant = np.asarray(ys_plant).squeeze()
    ys_greybox = np.asarray(ys_greybox).squeeze()
    us = uss.squeeze(axis=-1)
    (cost_plant, cost_greybox) = (cost(ys_plant, us), cost(ys_greybox, us))
    # Return the compiled model.
    return (cost_plant, cost_greybox, us)

def main():
    """ Main function to be executed. """
    # Load data.
    tworeac_parameters = PickleTool.load(filename=
                                         'tworeac_parameters_lin.pickle',
                                         type='read')
    # Create the hybrid model.
    #Np = 3
    #fnn_dims = [8, 2]
    (cost_plant, cost_greybox, us) = compute_cost_curves(parameters=
                             tworeac_parameters['parameters'])
    figures = plot_profit_curve(us=us, costs=[cost_plant, cost_greybox])
    # Save data.
    with PdfPages('tworeac_ssopt_lin.pdf', 'w') as pdf_file:
        for fig in figures:
            pdf_file.savefig(fig)

main()