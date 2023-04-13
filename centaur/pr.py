import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import copy
import scipy

import poppy

from . import utils
reload(utils)

import misc_funs as misc

try:
    from magpyx.utils import ImageStream
except:
    print('Could not find magpyx')
    
def get_pr():

    fdpr_amp = ImageStream('fdpr2_amp')
    fdpr_phase = ImageStream('fdpr2_phase')

    amp = fdpr_amp.grab_latest()
    phase = fdpr_phase.grab_latest()
    
    return amp, phase

from matplotlib.patches import Circle

def fix_pr_data(amp, phase, 
                pupil_diam=10.2*u.mm, npix=100, 
                center=(0,0), angle=0, 
                wavelength_c=632.8e-9*u.m, 
                return_mask=False, 
                plot=False):
    n = 2**int(np.ceil(np.log2(npix)))
    amp_err = misc.pad_or_crop(amp, n)
    phs_err = misc.pad_or_crop(phase, n)

    opd_err = phs_err * wavelength_c.value/(2*np.pi)

    pixelscale = pupil_diam/(npix*u.pix)
    
    radius = pupil_diam.value/2
    if plot:
        misc.imshow2(amp_err, opd_err, 
                     pxscl=pixelscale,
                     patches1=[Circle(center, radius, fill=False, color='c')],
                     patches2=[Circle(center, radius, fill=False, color='c')])

    xs = (np.linspace(-amp_err.shape[0]/2, amp_err.shape[0]/2-1, amp_err.shape[0])+1/2)*pixelscale.value
    x,y = np.meshgrid(xs,xs)
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    err_mask = r<radius

    amp_err *= err_mask
    opd_err *= err_mask
    if plot:
        misc.imshow2(amp_err, opd_err, 
                     pxscl=pixelscale,
                     patches1=[Circle(center, radius, fill=False, color='c')],
                     patches2=[Circle(center, radius, fill=False, color='c')])

    # center the error mask and the error data
    mask_centroid = misc.centroid(err_mask, rounded=True)
    print(mask_centroid)

    new_mask = np.zeros((2*np.max(mask_centroid), 2*np.max(mask_centroid)))
    nnew = new_mask.shape[0]
    new_mask[:n, (nnew-n)//2:n+(nnew-n)//2] = err_mask
    new_cent = misc.centroid(new_mask)

    new_amp_err = np.zeros((2*np.max(mask_centroid), 2*np.max(mask_centroid)))
    new_amp_err[:n, (nnew-n)//2:n+(nnew-n)//2] = amp_err
    new_opd_err = np.zeros((2*np.max(mask_centroid), 2*np.max(mask_centroid)))
    new_opd_err[:n, (nnew-n)//2:n+(nnew-n)//2] = opd_err

    new_amp_err = scipy.ndimage.rotate(new_amp_err, angle, reshape=False, order=1)
    new_opd_err = scipy.ndimage.rotate(new_opd_err, angle, reshape=False, order=3)

    new_amp_err /= new_amp_err.max()

    err_optic = poppy.ArrayOpticalElement(transmission=cp.array(new_amp_err), opd=cp.array(new_opd_err), pixelscale=pixelscale)
    if plot:
        misc.imshow2(err_optic.amplitude, err_optic.opd,
                     pxscl=pixelscale)
    
    if return_mask:
        return err_optic, new_mask
    else:
        return err_optic


