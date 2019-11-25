import numpy as np

ITL_AMP_GEOM = {'ncols' : 509,
                'nrows' : 2000,
                'num_serial_prescan' : 3,
                'num_serial_overscan' : 64,
                'num_parallel_overscan': 48,
                'last_pixel_index' : 511}

E2V_AMP_GEOM = {'ncols' : 512,
                'nrows' : 2002,
                'num_serial_prescan' : 10,
                'num_serial_overscan' : 64,
                'num_parallel_overscan' : 48,
                'last_pixel_index' : 521}

def calculate_cti(imarr, last_pix_num, num_overscan_pixels=1):
    """Calute the serial CTI of an image array."""
    
    last_pix = np.mean(imarr[:, last_pix_num])
    
    if num_overscan_pixels == 1:
        overscan = np.mean(imarr[:, last_pix_num+1])
    elif num_overscan_pixels == 2:
        overscan1 = np.mean(imarr[:, last_pix_num+1])
        overscan2 = np.mean(imarr[:, last_pix_num+2])
        overscan = overscan1 + overscan2
    else:
        raise ValueError("num_overscan_pixels must be 1 or 2")
    
    cti = overscan/(last_pix*last_pix_num)
    
    return cti
