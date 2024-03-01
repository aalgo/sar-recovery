import numpy as np

def compute_coherence_matrix(scatter_matrix):
    """
    batch compute coherence matrix pixel wise from a scatter matrix 
    (no statistics for sliding window, just per pixel)

    scatter matrix: np array of complex numbers with shape [h,w,2,2]

    returns: coherence matrix of complex numbers with shape [h,w,3,3]       
    """
    
    sm = scatter_matrix
    # obtain kl = [Shh, sqrt(2) Shv, Svv]
    kl = np.r_[[sm[:,:,0,0], np.sqrt(2)*sm[:,:,0,1], sm[:,:,1,1]]]
    kl = np.transpose(kl, [1,2,0])

    # obtain coherence matrix for each pixel by doing kl * kl.T
    _kl = kl.reshape(*kl.shape, 1)                                # shape is [h,w,3,1]
    _klc = kl.conjugate().reshape(*kl.shape[:2], 1, kl.shape[2])  # shape is [h,w,1,3]
    cm = _kl*_klc                                                 # shape is [h,w,3,3]
    
    return cm

def compute_coherency_matrix_pauli(pauli_scattering_vector):
    """
    batch compute coherency matrix pixel wise from a Pauli scattering vector 
    (no statistics for sliding window, just per pixel)

    scatter pauli_scattering_vector: np array of complex numbers with shape [h,w,3]

    returns: coherency matrix of complex numbers with shape [h,w,3,3]       
    """
    
    wp = pauli_scattering_vector
    T = np.einsum("...i,...j->...ij", wp, wp.conj())
    
    return T

def generate_Pauli_RGB_from_T(T, scale_factor=2):
    """
    Generate a Pauli RGB image from a coherency matrix T

    Parameters
    ----------
    T : array of shape (h, w, 3, 3)
        Coherency matrix (covariance matrix of the Pauli scattering vector)
    scale_factor : float, optional
        Scaling factor of the RGB representation. The default is 2.

    Returns
    -------
    rgb : array of shape (h, w, 3)
        RGB representation in the range [0,1].

    """
    rgb = np.zeros(T.shape[:-2] + (3,))
    rgb[..., 0] = np.sqrt(T[...,1,1].real)
    rgb[..., 1] = np.sqrt(T[...,2,2].real)
    rgb[..., 2] = np.sqrt(T[...,0,0].real)
    pmean = np.mean(rgb, axis=(0,1))
    rgb /= pmean * scale_factor
    return rgb.clip(0, 1)

def get_H_A_alpha(T3, eps=1e-6):
    """
    Returns the H/A/alpha decomposition of a 3x3 coherency matrix T3.
    
    The decomposition is described in the paper:
    Cloude, S. R., & Pottier, E. (1996). A review of target decomposition
    theorems in radar polarimetry. IEEE transactions on geoscience and remote
    sensing, 34(2), 498-518.

    Parameters
    ----------
    T3 : ndarray of shape (..., 3, 3)
        Coherency matrix (covariance matrix of the Pauli scattering vector)
    eps : float, optional
        Small value to avoid numerical errors when computing the entropy for
        zero or very small eigenvalues. The default is 1e-6.

    Returns
    -------
    H : ndarray of shape (...)
        Entropy value for each pixel [0..1]
    A : ndarray of shape (...)
        Anisotropy value for each pixel [0..1]
    alpha : ndarray of shape (...)
        Mean alpha angle value for each pixel [0..np.pi/2]

    """
    l, V = np.linalg.eigh(T3)
    # Sort eigenvalues and eigenvectors in descending order, instead of ascending
    l = l[..., ::-1]
    V = V[..., ::-1]
    pi = l / np.sum(l,axis=-1)[..., np.newaxis] + eps
    H = np.sum(-pi*np.log(pi)/np.log(T3.shape[-1]), axis=-1)
    #H = scipy.stats.entropy(pi, base=T3.shape[-1])
    alphai = np.arccos(np.abs(V[...,0,:]))
    alpha = np.sum(pi*alphai, axis=-1)
    A = (pi[...,1] - pi[...,2]) / (pi[...,1] + pi[...,2])
    return H, A, alpha

def symmetric_revised_Wishart_distance(T1, T2, eps=1e-6):
    """
    Returns the Symmetric Revised Wishart distance (more precisely,
    dissimilarity measure) between two covariance matrices T1 and T2 pixelwise.
    
    It also applies a regularization factor based on eps.
    
    References:
    [1] A. Alonso-Gonzalez, C. Lopez-Martinez and P. Salembier, "Filtering and
    Segmentation of Polarimetric SAR Data Based on Binary Partition Trees,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 50, no. 2,
    pp. 593-605, Feb. 2012, doi: 10.1109/TGRS.2011.2160647
    [2] Qin, X., Zhang, Y., Li, Y., Cheng, Y., Yu, W., Wang, P., & Zou, H.
    (2022). Distance measures of polarimetric SAR image data: A survey.
    Remote Sensing, 14(22), 5873.

    Parameters
    ----------
    T1 : ndarray of shape (..., 3, 3)
        Covariance or coherency matrix of the first image.
    T2 : ndarray of shape (..., 3, 3)
        Covariance or coherency matrix of the second image.
    eps : float, optional
        Small value to avoid numerical errors when computing the dissimilarity
        for zero or very small eigenvalues. The default is 1e-6.

    Returns
    -------
    ndarray of shape (...)
        Symmetric Revised Wishart dissimilarity measure for each pixel.

    """
    T1 = T1 + np.eye(T1.shape[-1]) * eps
    T2 = T2 + np.eye(T2.shape[-1]) * eps
    return (np.sum(np.einsum("...ii->...i", np.linalg.solve(T1, T2)).real, axis=-1) + 
            np.sum(np.einsum("...ii->...i", np.linalg.solve(T2, T1)).real, axis=-1)
            ) / 2 - T1.shape[-1]
