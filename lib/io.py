import numpy as np


def load_bcn_slc(basepath, datestr, pol):
    """
    loads an SLC image from the Barcelona dataset with resolution 4402x1602
    
    returns numpy array os shape 4402x1602 with complex entries
    """
    img_path = f"{basepath}/{datestr}.rst2.qp/{datestr}_slc_{pol}.dat"
    img = np.fromfile(img_path, dtype=np.complex64)
    img = img.reshape((4402,1602))
    return img

def load_bcn_scatter_matrix(base_path, date):
    """
    loads the four hh hv vh vv files for a 'date' and puts them together
    into an array of shape 4402x1602x2x2 so that
    [:,:,0,0] is Shh
    [:,:,0,1] is Shv
    [:,:,1,0] is Svh
    [:,:,1,1] is Svv    
    """
    shh = load_bcn_slc(base_path, date, 'HH')
    shv = load_bcn_slc(base_path, date, 'HV')
    svh = load_bcn_slc(base_path, date, 'VH')
    svv = load_bcn_slc(base_path, date, 'VV')
    sm = np.r_[[shh, shv], [svh, svv]]
    sm = np.transpose(sm, [1,2,0])
    sm = sm.reshape(*sm.shape[:2], 2, 2)

    return sm
