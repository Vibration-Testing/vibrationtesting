"""
System expansion, reduction, and corrections functions for Sparse systems.

@author: Joseph C. Slater and Sainag Immidisetty
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


#import math
import numpy as np
#import scipy.signal as sig
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import lil_matrix
#from scipy.sparse import csr_matrix
import scipy.sparse as sps


def sos_modal_forsparse(M, K, C=False, damp_diag=0.03, shift=1):
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


	"""

	#Kdiag = K.diagonal().reshape(-1,1)

	#Mdiag = M.diagonal().reshape(-1,1)

	#minvals = np.sort((Kdiag/Mdiag),axis=0)

	#shift = minvals[min(7,len(minvals))]

	#shift = shift[0]

	#K = ((K.tocsr() + (K.T).tocsr()).tolil()) * 0.5 + shift * ((M.tocsr() + (M.T).tocsr()).tolil()) * 0.5

	#M = ((M.tocsr() + (M.T).tocsr()).tolil()) * 0.5

	K = lil_matrix(K)

	M = lil_matrix(M)

	K = K + M * shift

	[lam, Psi] = la.eigh(K.toarray(), M.toarray())

	lam = lam.reshape(-1,1)

	omega = np.sqrt(np.abs(lam - shift))

	omega = omega.reshape(-1,)

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

	return Mred, Kred, T, master
