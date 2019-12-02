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
    """Calculate the serial CTI of an image array."""
    
    last_pix = np.mean(imarr[:, last_pix_num])

    overscan = np.mean(imarr[:, last_pix_num+1:], axis=0)
    cti = np.sum(overscan[:num_overscan_pixels])/(last_pix*last_pix_num)
                           
    return cti
