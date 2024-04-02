import numpy as np
TRAIN, TEST, VAL = 0,1,2

def cv_splitpixels_spatial(h, w, pixels_train, pixels_test, pixels_val, angle):
    
    """
    marks pixels in an image of size [h,w] as train, test or val by "drawing" consecutive
    bands with the specified number of pixels wide and angle.
    
    Parameters
    ----------
    
    h,w: int height and width of the image whose pixels to split
    
    pixels_train, 
    pixels_test, 
    pixels_val: the number of pixels for each split in a band. For instance, with 
                values 10,5,5 it will split the image in consecutive bands 20 pixels
                wide with 10 pixels for train 5 for test and 5 for val
    
    angle: the angle of the bands
    
    Returns
    -------
    
    ndarray of shape [h,w] with values 0 to mark pixels in train, 1 in test, 2 in val
    
    the actual pct of pixels in train, test and val might differ from the proportions
    of pixels_train, pixels_test and pixels_val depending on their relative size to the 
    image and the angle (i.e. a band might just be starting when reaching an image border)
                
    """

    if angle<-np.pi/2 or angle>np.pi/2:
        raise ValueError("angle must be between -pi/2 and pi/2")

    r = np.zeros((h,w))

    # change angle so that it makes intuitive sense when looking at the map
    # i.e. pi/2 is vertical, pi/6 is 30° wrt to the x axis, etc.
    angle = -angle
    if np.abs(angle)>np.pi/4:
        _w = w
        w = h
        h = _w
        switched = True
        angle = np.pi/2 - angle
    else:
        switched = False
                
    # build a band template
    band  = np.r_[ [TRAIN] * pixels_train + [TEST] * pixels_test + [VAL] * pixels_val]
    band_pixels = pixels_train + pixels_test + pixels_val
    
    # create mask by shifting each row according to the desired angle
    r = np.zeros((h,w))
    for col in range(w):
        i = int(np.round(col*np.tan(angle),0)) % h
        r[i,col]=4
        for row in range(h):
            r[(row + i)%h, col] = band[row%band_pixels]
            
            
    if switched:
        r = r.T
            
    return r
                 

def cv_splitpixels_random(h, w, train_pct, test_pct, val_pct):
    """
    splits pixels of an image randomly
    
    Parameters
    -----------
    h,w: int height and width of the image whose pixels to split

    train_pct,
    test_pct,
    val_pct:  float the percentage of pixels to mark for each split

    Returns
    -------
    ndarray of shape [h,w] with values 0 to mark pixels in train, 1 in test, 2 in val
    
    """
    
    if not np.allclose(train_pct, test_pct, val_pct, 1):
        raise ValueError(f"pcts must add up to 1, but they add up to {train_pct + test_pct + val_pct}")
    
    r = np.random.random(size=(h,w))
    r[r<train_pct]=0
    r[r>train_pct + test_pct] = 2
    r[(r!=0)&(r!=2)]=1
    return r