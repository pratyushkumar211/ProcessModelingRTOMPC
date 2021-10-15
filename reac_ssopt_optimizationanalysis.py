# [depends] %LIB%/plottingFuncs.py %LIB%/hybridId.py
# [depends] %LIB%/BlackBoxFuncs.py %LIB%/ReacHybridFuncs.py
# [depends] %LIB%/linNonlinMPC.py %LIB%/reacFuncs.py
# [depends] reac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hybridId import PickleTool
from linNonlinMPC import (c2dNonlin, doOptimizationAnalysis)
from reacFuncs import cost_lxup_noCc, cost_lxup_withCc, plant_ode

# Import function handles for Black-Box, Full Grey-Box and Hybrid Grey-Box.
from BlackBoxFuncs import get_bbnn_pars, bbnn_fxu, bbnn_hx
from ReacHybridFullGbFuncs import get_hybrid_pars as get_fhyb_pars
from ReacHybridFullGbFuncs import hybrid_fxup as fhyb_fxup
from ReacHybridFullGbFuncs import hybrid_hx as fhyb_hx
from ReacHybridPartialGbFuncs import get_hybrid_pars as get_phyb_pars
from ReacHybridPartialGbFuncs import hybrid_fxup as phyb_fxup
from ReacHybridPartialGbFuncs import hybrid_hx as phyb_hx

# Set numpy seed. 
np.random.seed(10)

def main():
    """ Main function to be executed. """

    # Load data.
    reac_parameters = PickleTool.load(filename=
                                         'reac_parameters.pickle',
                                         type='read')
    reac_bbnntrain = PickleTool.load(filename=
                                    'reac_bbnntrain.pickle',
                                      type='read')
    reac_hybfullgbtrain = PickleTool.load(filename=
                                      'reac_hybfullgbtrain.pickle',
                                      type='read')
    reac_hybpartialgbtrain = PickleTool.load(filename=
                                      'reac_hybpartialgbtrain.pickle',
                                      type='read')

    # Extract out the training data for analysis. 
    reac_bbnntrain = reac_bbnntrain[0]
    reac_hybfullgbtrain = reac_hybfullgbtrain[0]
    reac_hybpartialgbtrain = reac_hybpartialgbtrain[0]

    # Get plant and hybrid model parameters.
    plant_pars = reac_parameters['plant_pars']
    hyb_fullgb_pars = reac_parameters['hyb_fullgb_pars']
    hyb_partialgb_pars = reac_parameters['hyb_partialgb_pars']

    # Plant function handles.
    Delta, ps = plant_pars['Delta'], plant_pars['ps']
    plant_fxu = lambda x, u: plant_ode(x, u, ps, plant_pars)
    plant_f = c2dNonlin(plant_fxu, Delta)
    plant_h = lambda x: x[plant_pars['yindices']]

    # Black-Box NN function handles.
    bbnn_pars = get_bbnn_pars(train=reac_bbnntrain, 
                              plant_pars=plant_pars)
    bbnn_f = lambda x, u: bbnn_fxu(x, u, bbnn_pars)
    bbnn_h = lambda x: bbnn_hx(x, bbnn_pars)

    # Full GB Hybrid model function handles.
    fhyb_pars = get_fhyb_pars(train=reac_hybfullgbtrain, 
                              hyb_fullgb_pars=hyb_fullgb_pars, 
                              plant_pars=plant_pars)
    ps = fhyb_pars['ps']
    fhyb_f = lambda x, u: fhyb_fxup(x, u, ps, fhyb_pars)
    fhyb_h = lambda x: fhyb_hx(x, fhyb_pars)
    
    # Partial GB Hybrid model and function handles.
    phyb_pars = get_phyb_pars(train=reac_hybpartialgbtrain, 
                              hyb_partialgb_pars=hyb_partialgb_pars, 
                              plant_pars=plant_pars)
    ps = phyb_pars['ps']
    phyb_f = lambda x, u: phyb_fxup(x, u, ps, phyb_pars)
    phyb_h = lambda x: phyb_hx(x, phyb_pars)

    # List to store the result of optimization analysis. 
    optAnalysis_list = []

    # Number of initial guesses/cost parameter values.
    Nguess = 10
    Npvals = 500

    ## Optimization analysis for the cost type 1 without a Cc contribution.
    # Get lists of model types.
    model_types = ['Plant', 'Black-Box-NN', 
                   'Hybrid-FullGb', 'Hybrid-PartialGb']
    fxu_list = [plant_f, bbnn_f, fhyb_f, phyb_f]
    hx_list = [plant_h, bbnn_h, fhyb_h, phyb_h]
    par_list = [plant_pars, bbnn_pars, fhyb_pars, phyb_pars]
    # Lower and upper bounds of cost parameters. 
    plb = np.array([100, 500])
    pub = np.array([100, 1500])
    reac_optanalysis = doOptimizationAnalysis(model_types=model_types, 
                                        fxu_list=fxu_list, hx_list=hx_list, 
                                        par_list=par_list, lxup=cost_lxup_noCc,
                                        plb=plb, pub=pub, Npvals=Npvals, 
                                        Nguess=Nguess)
    optAnalysis_list += [reac_optanalysis]

    ## Optimization analysis for the cost type 2 with a Cc contribution.
    model_types = ['Plant', 'Hybrid-FullGb']
    fxu_list = [plant_f, fhyb_f]
    hx_list = [plant_h, fhyb_h]
    par_list = [plant_pars, fhyb_pars]
    # Lower and upper bounds of cost parameters. 
    plb = np.array([100, 1000, 100])
    pub = np.array([100, 2000, 500])
    reac_optanalysis = doOptimizationAnalysis(model_types=model_types, 
                                    fxu_list=fxu_list, hx_list=hx_list, 
                                    par_list=par_list, lxup=cost_lxup_withCc,
                                    plb=plb, pub=pub, Npvals=Npvals, 
                                    Nguess=Nguess)
    optAnalysis_list += [reac_optanalysis]

    # Save.
    PickleTool.save(data_object=optAnalysis_list,
                    filename='reac_ssopt_optimizationanalysis.pickle')

main()