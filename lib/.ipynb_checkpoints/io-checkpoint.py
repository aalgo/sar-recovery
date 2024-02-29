import numpy as np


def load_bcn_slc(basepath, datestr, pol):
    """
    loads an SLC image from the Barcelona dataset with resolution 4402x1602
    
    returns numpy array with complex entries
    """
    img_path = f"{basepath}/{datestr}.rst2.qp/{datestr}_slc_{pol}.dat"
    img = np.fromfile(img_path, dtype=np.complex64)
    img = img.reshape((4402,1602))
    return img