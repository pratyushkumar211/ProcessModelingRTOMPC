# [depends] HybridModelLayers.py hybridid.py
"""
Custom neural network layers for the 
data-based completion of grey-box models 
using neural networks.
Pratyush Kumar, pratyushkumar@ucsb.edu
"""
import sys
import numpy as np
import tensorflow as tf
from HybridModelLayers import BlackBoxModel, KoopmanModel
from hybridid import SimData
