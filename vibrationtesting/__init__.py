"""


"""
from __future__ import division, print_function, absolute_import

__title__ = 'vibrationtesting'
__version__ = '0.1'
__author__ = u'Joseph C. Slater'
__license__ = 'MIT'
__copyright__ = 'Copyright 2002-2017 Joseph C. Slater'

import sys
import matplotlib as mpl

if 'pytest' in sys.argv[0]:
    print('Setting backend to agg to run tests')
    mpl.use('agg')

from .signals import *
from .system import *
from .identification import *

#  Signal processing (:mod:`vibrationtesting.signal`)

# print options were change inside modules to produce better
# outputs at examples. Here we set the print options to the
# default values after importing the modules to avoid changing
# np default print options when importing the module.
