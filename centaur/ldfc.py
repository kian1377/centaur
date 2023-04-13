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

from esc_coro_suite import misc
from . import utils

def calibrate(sysi, amp, modes, bright_mask):
    print('Calibrating with given modes ...')
    Nact = sysi.Nact
    
    R = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, mode in enumerate(modes):
        try:
            sysi.add_dm(amp * mode.reshape(Nact,Nact))
            im_plus = sysi.snap()
            sysi.add_dm(-amp * mode.reshape(Nact,Nact))
            
            sysi.add_dm(-amp * mode.reshape(Nact,Nact))
            im_minus = sysi.snap()
            sysi.add_dm(amp * mode.reshape(Nact,Nact))
            
            R.append((im_plus.flatten()-im_minus.flatten())[bright_mask]/(2*amp))
            
            print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1, modes.shape[0], time.time()-start))
        except KeyboardInterrupt: 
            print('Calibration interrupted.')
            break
    print('Calibration complete.')
    
    return np.array(R).T

def run_sim(sysi, WFEs, reference, bright_mask, G,  M_eigen, frequency=1, loop_gain=0.2, modal_gain=1, display=True, display_all=False):
    
    dm_ref = sysi.get_dm()
    
    for i,wfe in enumerate(WFEs):
        try:
            print('\tRunning LDFC to maintain dark-hole: Iteration {:d}.'.format(i))
            
            im = sysi.snap()
            
            delI = reference.flatten()[bright_mask] - im.flatten()[bright_mask]
            
            measurement_vector = G.dot(delI)
            measurement_vector *= modal_gain
            
            del_dm = loop_gain * M_eigen.T.dot(measurement_vector).reshape(sysi.Nact, sysi.Nact)
            
            sysi.set_dm(dm_ref - del_dm)
            
            sysi.OPD = wfe
            
            if display:
                misc.myimshow3(del_dm, sysi.get_dm(), im, 
                               'DM Command', 'Total DM', 'Image',
                               lognorm3=True, vmin3=im.max()/1e5) 
            
                if not display_all: clear_output(wait=True)
            time.sleep(1/frequency)
                        
        except KeyboardInterrupt:
            break

def run(sysi, reference, bright_mask, G, M_eigen, frequency=1, loop_gain=0.2):
    
    while True:
        try:
            print('\tRunning LDFC to maintain dark-hole: Iteration {:d}.'.format(i))
            im = sysi.snap()
            
            delI = reference.flatten()[bright_mask] - im.flatten()[bright_mask]
            
            measurement_vector = G.dot(delI)
            
            del_dm = M_eigen.T.dot(measurement_vector).reshape(sysi.Nact, sysi.Nact)
            
            sysi.add_dm(del_dm)
            
            sysi.OPD = wfe
            
            misc.myimshow3(del_dm, sysi.get_dm(), im, 
                           'DM Command', 'Total DM', 'Image',
                           lognorm3=True, vmin3=im.max()/1e5) 
            time.sleep(1/frequency)
            
            clear_output(wait=True) 
        except KeyboardInterrupt:
            break
            

    
    
