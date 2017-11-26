"""
Created on Nov. 26, 2017
@author: Joseph C. Slater
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


import math
import warnings
import numpy as np
import control as ctrl
# from numpy import ma
import scipy as sp
# import scipy.signal as sig
import scipy.fftpack as fftpack
import scipy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
# rcParams = matplotlib.rcParams
from vibration_toolbox import sdof_cf, mdof_cf
