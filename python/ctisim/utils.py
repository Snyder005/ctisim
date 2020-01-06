import numpy as np
from lsst.eotest.sensor.AmplifierGeometry import AmplifierGeometry, amp_loc

ITL_AMP_GEOM = AmplifierGeometry(prescan=3, nx=509, ny=2000, 
                                 detxsize=4608, detysize=4096,
                                 amp_loc=amp_loc['ITL'], vendor='ITL')

E2V_AMP_GEOM = AmplifierGeometry(prescan=10, nx=512, ny=2002,
                                 detxsize=4688, detysize=4100,
                                 amp_loc=amp_loc['E2V'], vendor='E2V')

def calculate_cti(imarr, last_pix_num, num_overscan_pixels=1):
    """Calculate the serial CTI of an image array."""
    
    last_pix = np.mean(imarr[:, last_pix_num])

    overscan = np.mean(imarr[:, last_pix_num+1:], axis=0)
    cti = np.sum(overscan[:num_overscan_pixels])/(last_pix*last_pix_num)
                           
    return cti
