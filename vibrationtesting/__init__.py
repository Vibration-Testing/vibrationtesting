"""
==================================================
Signal processing (:mod:`vibrationtesting.signal`)
==================================================


Window functions
================

.. autosummary::
   :toctree: generated/

   barthann          -- Bartlett-Hann window
   bartlett          -- Bartlett window
   blackman          -- Blackman window
   boxcar            -- Boxcar window
   chebwin           -- Dolph-Chebyshev window
   flattop           -- Flat top window
   gaussian          -- Gaussian window
   general_gaussian  -- Generalized Gaussian window
   hamming           -- Hamming window
   hann              -- Hann window
   parzen            -- Parzen window
   triang            -- Triangular window

"""
from __future__ import division, print_function, absolute_import

__title__ = 'vibrationtesting'
__version__ = '0.1b03'
__author__ = u'Joseph C. Slater'
__license__ = 'MIT'
__copyright__ = 'Copyright 2002-2017 Joseph C. Slater'

import sys
import matplotlib as mpl

if 'pytest' in sys.argv[0]:
    print('Setting backend to agg to run tests')
    mpl.use('agg')

from .signal import *
from .system import *




# print options were change inside modules to produce better
# outputs at examples. Here we set the print options to the
# default values after importing the modules to avoid changing
# np default print options when importing the module.

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75,
                    nanstr='nan', precision=8, suppress=False,
                    threshold=1000, formatter=None)
