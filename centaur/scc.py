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

from scipy import interpolate

import poppy

from . import utils

from importlib import reload
reload(utils)

import misc

def est_coherent(image, coordinates, mask, separation, wavelength):   
    # Center sideband using tilt
    xf, yf = coordinates 
    image = image + 0*1j
    image *= np.exp(1j*2*np.pi/wavelength*xf*separation) * np.exp(1j*2*np.pi/wavelength*yf*separation)
    
    image = cp.array(image)
    
    # Fourier Transform image
#     image_fft = fft.forward(image)
    image_fft = poppy.accel_math.fft_2d(image, forward=True, normalization=None, fftshift=True)
    
    # Isolate sideband
    image_fft *= mask
    
    # Inverse transform 
#     sideband = fft.backward(image_fft)
    sideband = poppy.accel_math.fft_2d(image, forward=False, normalization=None, fftshift=True)
    return sideband



