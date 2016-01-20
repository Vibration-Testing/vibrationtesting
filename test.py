
import math
import warnings

import numpy as np
from numpy import ma
import scipy as sp
from numpy import linalg as la 
import matplotlib
import matplotlib.pyplot as plt
rcParams = matplotlib.rcParams

import matplotlib.cbook as cbook
from matplotlib.cbook import _string_to_bool, mplDeprecation
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates as _  # <-registers a date unit converter
from matplotlib import docstring
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.transforms as mtrans
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.axes._base import _AxesBase
from matplotlib.axes._base import _process_plot_format
import pylab as pl

def example():
    """
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> time = np.arange(0,1,.1)
    >>> time = np.reshape(time, (1, -1))
    >>> print(time)
    >>> plt.plot(time,time)
    >>> plt.show()
    """




if __name__ == "__main__":
    import doctest
    doctest.testmod()
    """ What this does. 
    python (name of this file)  -v
    will test all of the examples in the help.

    Leaving off -v will run the tests without any output. Success will return nothing.

    See the doctest section of the Sphinx manual.
    """
