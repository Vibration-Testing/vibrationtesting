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

	#kmm = K[master,:].toarray()

	#kmm=kmm[:, master]

	#ksm = K[slave,:].toarray()

	#ksm=ksm[:, master]

	#kss = K[slave,:].toarray()

	#kss=kss[:, slave]

    #kss = slice_forSparse(K, slave, slave)

    #ksm = slice_forSparse(K, slave, master)

	kss = slice_forSparse(K, slave, slave)

	ksm = slice_forSparse(K, slave, master)

	T= np.zeros((len(master)+len(slave), len(master)))

	T= lil_matrix(T)

	T[master,:lmaster] = sps.eye(lmaster,lmaster)

	T[slave,0:lmaster]=spla.spsolve(-kss,ksm)

	Mred = T.T * M * T

	Kred = T.T * K * T

	return Mred, Kred, master


def mode_expansion_from_model_forsparse(Psi, omega, M, K, measured):
    r"""Deflection extrapolation to full FEM model coordinates, matrix method.

    Provided an equation  of the form:

    :math:`\begin{pmatrix}-\begin{bmatrix}M_{mm}&M_{mu}\\ M_{um}&M_{uu}
    \end{bmatrix} \omega_i^2
    +\begin{bmatrix}K_{mm}&K_{mu}\\ K_{um}&K_{uu}\end{bmatrix}\end{pmatrix}`
    :math:`\begin{bmatrix}\Psi_{i_m}\\ \Psi_{i_u}\end{bmatrix}= 0`

    Where:

    - :math:`M` and :math:`K` are the mass and stiffness matrices, likely from
      a finite element model
    - :math:`\Psi_i` and :math:`\omega_i` represent a mode/frequency pair
    - subscripts :math:`m` and :math:`u` represent measure and unmeasured
      of the mode

    Determines the unknown portion of the mode (or operational deflection)
    shape, :math:`\Psi_{i_u}` by
    direct algebraic solution, aka

    :math:`\Psi_{i_u} = - (K_{uu}- M_{ss} \omega_i^2) ^{-1}
    (K_{um}-M_{um}\omega_i^2)\Psi_{i_m}`

    Parameters
    ----------
    Psi : float array
        mode shape or shapes, 2-D array columns of which are mode shapes
    omega : float or 1-D float array
        natural (or driving) frequencies
    M, K : float arrays
        Mass and Stiffness matrices
    measured : float or integer array or list
        List of measured degrees of freedom (0 indexed)

    Returns
    -------
    Psi_full : float array
        Complete mode shape

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> measured = np.array([[0, 2]])
    >>> omega, zeta, Psi = vt.sos_modal(M, K)
    >>> Psi_measured = np.array([[-0.15], [-0.37]])
    >>> Psi_full = vt.mode_expansion_from_model(Psi_measured, omega[0], M, K,
    ... measured)
    >>> print(np.hstack((Psi[:,0].reshape(-1,1), Psi_full)))
    [[-0.164  -0.15  ]
     [-0.2955  0.2886]
     [-0.3685 -0.37  ]]

    Notes
    -----
    .. seealso:: incomplete multi-mode update. Would require each at a
      different frequency.

    """


    measured = measured.reshape(-1)  # retained dofs
    num_measured = len(measured)
    ndof = int(M.shape[0])  # length(M);
    unmeasured_dofs = list(set(np.arange(ndof)) - set(measured))
    num_unmeasured = len(unmeasured_dofs)

    M= lil_matrix(M)

    K= lil_matrix(K)

	#Muu = M[unmeasured_dofs,:].toarray()

    #Muu = Muu[:, unmeasured_dofs]

    #Kuu = K[unmeasured_dofs,:].toarray()

    #Kuu = Kuu[:, unmeasured_dofs]

    #Mum = M[unmeasured_dofs,:].toarray()

    #Mum = Mum[:, measured]

    #Kum = K[unmeasured_dofs,:].toarray()

    #Kum = Kum[:, measured]

    Muu = slice_forSparse(M, unmeasured_dofs, unmeasured_dofs)

    Kuu = slice_forSparse(K, unmeasured_dofs, unmeasured_dofs)

    Mum = slice_forSparse(M, unmeasured_dofs, measured)

    Kum = slice_forSparse(K, unmeasured_dofs, measured)

    if isinstance(omega, float):
        omega = np.array(omega).reshape(1)

    Psi_full = np.zeros((num_measured + num_unmeasured, Psi.shape[1]))
    Psi_full[measured] = Psi

    for i, omega_n in enumerate(omega):
        Psi_i = Psi[:, i].reshape(-1, 1)
        Psi_unmeasured = la.solve((Kuu - Muu * omega_n**2),
                                  (Kum - Mum * omega_n**2)@Psi_i)
        Psi_unmeasured = Psi_unmeasured.reshape(-1, )
        Psi_full[unmeasured_dofs, i] = Psi_unmeasured
        # Psi_full = Psi_full.reshape(-1, 1)
    return Psi_full

def slice_forSparse(Matrix, a, b):
	"""

	Parameters
	----------
	Matrix : float array
		Arbitrary array
	a, b : int lists or arrays
		list of rows and columns to be selected from `Matrix`

	Returns
	-------
	Matrix : float array

	"""
	Maa = Matrix[a,:].toarray()

	Maa = Maa[:, a]

	Mab = Matrix[a,:].toarray()

	Mab = Mab[:, b]


	if(len(a)==len(b)):
		try:
			if(a==b.all()):
				return Maa

		except:
			return Mab
			raise
	else:
		return Mab
