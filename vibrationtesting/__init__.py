"""
================================================
Signal processing (:mod:`vibrationtesting.signal`)
================================================

.. module:: signal

Convolution
===========

Don't believe any of this!   It's still not updated!



.. autosummary::
   :toctree: generated/

   convolve    -- N-dimensional convolution.
   correlate   -- N-dimensional correlation.
   fftconvolve -- N-dimensional convolution using the FFT.
   convolve2d  -- 2-dimensional convolution (more options).
   correlate2d -- 2-dimensional correlation (more options).
   sepfir2d    -- Convolve with a 2-D separable FIR filter.


Window functions
================

.. autosummary::
   :toctree: generated/

   get_window        -- Return a window of a given length and type.
   barthann          -- Bartlett-Hann window
   bartlett          -- Bartlett window
   blackman          -- Blackman window
   blackmanharris    -- Minimum 4-term Blackman-Harris window
   bohman            -- Bohman window
   boxcar            -- Boxcar window
   chebwin           -- Dolph-Chebyshev window
   cosine            -- Cosine window
   flattop           -- Flat top window
   gaussian          -- Gaussian window
   general_gaussian  -- Generalized Gaussian window
   hamming           -- Hamming window
   hann              -- Hann window
   kaiser            -- Kaiser window
   nuttall           -- Nuttall's minimum 4-term Blackman-Harris window
   parzen            -- Parzen window
   slepian           -- Slepian window
   triang            -- Triangular window



Peak finding
============

.. autosummary::
   :toctree: generated/

   find_peaks_cwt -- Attempt to find the peaks in the given 1-D array
   argrelmin      -- Calculate the relative minima of data
   argrelmax      -- Calculate the relative maxima of data
   argrelextrema  -- Calculate the relative extrema of data
"""
from __future__ import division, print_function, absolute_import

__title__ = 'vibrationtesting'
__version__ = '0.1a01'
__author__ = u'Joseph C. Slater'
__license__ = 'MIT'
__copyright__ = 'Copyright 2002-2017 Joseph C. Slater'

import sys
import matplotlib as mpl

if 'pytest' in sys.argv[0]:
    print('Setting backend to agg to run tests')
    mpl.use('agg')

from .signal import *

"""
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75,
                    nanstr='nan', precision=8, suppress=False,
                    threshold=1000, formatter=None)
                    """
# from ._max_len_seq import max_len_seq

# The spline module (a C extension) provides:
#     cspline2d, qspline2d, sepfir2d, symiirord1, symiirord2
# from .spline import *

# from .bsplines import *
# from .cont2discrete import *
# from .dltisys import *
# from .filter_design import *
# from .fir_filter_design import *
# from .ltisys import *
# from .windows import *
# from .signaltools import *
# from ._savitzky_golay import savgol_coeffs, savgol_filter
# from .spectral import *
# from .wavelets import *
# from ._peak_finding import *

# __all__ = [s for s in dir() if not s.startswith('_')]
# from numpy.testing import Tester
# test = Tester().test
# bench = Tester().bench

# print options were change inside modules to produce better
# outputs at examples. Here we set the print options to the
# default values after importing the modules to avoid changing
# np default print options when importing the module.

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75,
                    nanstr='nan', precision=8, suppress=False,
                    threshold=1000, formatter=None)