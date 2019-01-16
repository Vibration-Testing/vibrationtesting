"""
System manipulation, reduction, and corrections functions.

@author: Joseph C. Slater
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


import math

import numpy as np
import scipy.signal as sig
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import lil_matrix


def sos_modal_forsparse(M, K, C=False, damp_diag=0.03):
    r"""Eigen analysis of proportionally damped system.

    Optimally find mass normalized mode shapes and natural frequencies
    of a system modelled by :math:`M\ddot{x}+Kx=0`.

    If provided, obtain damping ratios presuming :math:`C` can be decoupled.

    Provides a warning if diagonalization of damping matrix fails worse than
    relative error of `damp_diag`.

    Parameters
    ----------
    M, K : float arrays
        Mass and stiffness matrices
    C : float array, optional
        Damping matrix
    damp_diag : float, optional
        Maximum amount of off-diagonal error allowed in assuming C can be
        diagonalized
    shift : float, optional
        Shift used in eigensolution. Should be approximately equal to the first
        non-zero eigenvalue.

    Returns
    -------
    omega : float array (1xN)
        Vector of natural frequencies (rad/sec)
    zeta : float array (1xN)
        Vector of damping ratios
    Psi : float array (NxN)
        Matrix of mass normalized mode shapes by column


    Examples
    --------
    >>> import numpy as np
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> omega, zeta, Psi = vt.sos_modal(M, K, K/10)

    """
    #K = K + M * shift



    Kdiag = K.diagonal().reshape(-1,1)

    Mdiag = M.diagonal().reshape(-1,1)

    minvals = np.sort((Kdiag/Mdiag),axis=0)

    minvallocs = (Kdiag/Mdiag).argsort(axis=0)

    shift = minvals[min(7,len(minvals))]

    shift = shift[0]

    K = ((K.tocsr() + (K.T).tocsr()).tolil()) * 0.5 + shift * ((M.tocsr() + (M.T).tocsr()).tolil()) * 0.5

    M = ((M.tocsr() + (M.T).tocsr()).tolil()) * 0.5

    [lam, Psi] = spla.eigsh(A = K, k = np.min((K.shape[1], np.max((np.floor(math.sqrt(K.shape[1])), 100))))-1, M = M)

    lam = lam.reshape(-1,1)

    omega = np.sqrt(np.abs(lam - shift))

    norms = np.diag(1.0 / np.sqrt(np.diag(Psi.T@M@Psi)))

    Psi = Psi @ norms

    zeta = np.zeros_like(omega)

    if C is not False:
        diagonalized_C = Psi.T@C@Psi

        diagonal_C = np.diag(diagonalized_C)

        if min(omega) > 1e-5:
            zeta = diagonal_C / 2 / omega  # error if omega = 0
            max_off_diagonals = np.amax(np.abs(diagonalized_C
                                               - np.diag(diagonal_C)), axis=0)
            # error if no damping
            damp_error = np.max(max_off_diagonals / diagonal_C)
        else:
            zeta = np.zeros_like(omega)
            damp_error = np.zeros_like(omega)
            de_diag_C = diagonalized_C - np.diag(diagonal_C)
            for mode_num, omega_i in enumerate(omega):
                if omega[mode_num] > 1e-5:
                    zeta[mode_num] = diagonal_C[mode_num] / 2 / omega_i
                    damp_error = (np.max(np.abs(de_diag_C[:, mode_num]))
                                  / diagonal_C[mode_num])

        if damp_error > damp_diag:
            print('Damping matrix cannot be completely diagonalized.')
            print('Off diagonal error of {:4.0%}.'.format(damp_error))

    return omega, zeta, Psi
