"""
System expansion, reduction, and corrections functions for Sparse systems.

@author: Joseph C. Slater and Sainag Immidisetty
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


import math
import numpy as np
#import scipy.signal as sig
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import lil_matrix
#from scipy.sparse import csr_matrix
import scipy.sparse as sps


def guyan_forsparse(M, K, master=None, fraction=None):
    r"""Guyan reduced model.

    Applies Guyan Reductions to second order system of equations of the form

    .. math:: M \ddot{x} + K x = 0

    which are reduced to the form

    .. math:: M_r \ddot{x}_m + K_r x_m = 0

    where

    .. math::

        x = T x_m

        M_r= T^T M T

        K_r= T^T K T

    Parameters
    ----------
    M, K : float arrays
        Mass and Stiffness matrices
    master : float array or list, optional
        List of retained degrees of freedom (0 indexing)
    fraction : float, optional
        Fraction of degrees of freedom (0< `fraction` <1.0) to retain in model.
        If both master and fraction are neglected, fraction is set to 0.25.

    Returns
    -------
    Mred, Kred, T : float arrays
        Reduced Mass matric, Reduced Stiffness matrix, Transformation matrix
    master_dofs : int list
        List of master degrees of freedom (0 indexing)
    truncated_dofs : int list
        List of truncated degrees of freedom (0 indexing)

    Examples
    --------

    Notes
    -----
    Reduced coordinate system forces can be obtained by `Fr = T.T @ F`.

    Reduced damping matrix can be obtained using `Cr = T.T @ C @ T`.

    If mode shapes are obtained for the reduced system, full system mode shapes
    are `phi = T @ phi_r`.

    Code is not as efficient as possible. Using block submatrices would be
    more efficient.

    """


    if master is None:
        if fraction is None:
            fraction = 0.25

        ratios = np.diag(M) / np.diag(K)
        ranked = [i[0] for i in sorted(enumerate(ratios), key=lambda x: x[1])]
        thresh = int(fraction * ratios.size)
        if (thresh >= ratios.size) or thresh == 0:
            print("Can't keep", thresh, 'DOFs.')
            print("Fraction of", fraction, "is too low or too high.")
            return 0, 0, 0, 0, 0

        master = ranked[-thresh:]

    master = np.array(master)

    ncoord = M.shape[0]

    i = np.arange(0, ncoord)

    i = i.reshape(1,-1)

    i = i + np.ones((1,i.shape[1]),int)

    lmaster = master.shape[1]

    i[0,master-1] = np.transpose(np.zeros((lmaster,1)))

    i = np.sort((i), axis =1)

    slave = i[0,lmaster + 0:ncoord]

    K= lil_matrix(K)

    slave = slave.reshape(1,-1)

    master = master-np.ones((1,master.shape[0]),int)

    master = master.ravel()

    slave = slave - np.ones((1,slave.shape[0]),int)

    slave = slave.ravel()

    kmm = K[master,:].toarray()

    kmm=kmm[:, master]

    ksm = K[slave,:].toarray()

    ksm=ksm[:, master]

    kss = K[slave,:].toarray()

    kss=kss[:, slave]

    T= np.zeros((len(master)+len(slave), len(master)))

    T= lil_matrix(T)

    T[master,:lmaster] = sps.eye(lmaster,lmaster)

    T[slave,0:lmaster]=spla.spsolve(-kss,ksm)

    Mred = T.T * M * T

    Kred = T.T * K * T

    return Mred, Kred, T, master, slave
