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