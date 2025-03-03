#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:38:55 2025

@author: victorionescu
"""

import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import cv2
from scipy.signal import detrend
from scipy.signal import argrelextrema
from cellpose_omni import io, transforms
from omnipose.utils import normalize99
from copy import deepcopy
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES
import time
from scipy import interpolate
from scipy.signal import find_peaks_cwt,peak_prominences
from scipy.signal import medfilt,argrelmax,iirfilte
def addrandomsins(t,signal,finterval = [0,30],ainterval = [0,10],N=100):
    
    for i in range(N):
        f = np.random.uniform(finterval[0],finterval[1])
        A = np.random.uniform(ainterval[0],ainterval[1])
        
        sine = A*np.sin(2*np.pi*f*t) 
        signal += sine
        
T = 10
t = np.linspace(0,T,400)
f = 1 
A = 10
signal  = A*np.sin(2*np.pi*f*t)
addrandomsins(t,signal)
m = np.random.uniform(-0.5,0.5)
b = np.random.uniform(3000,5000)

signal+=m*t+b


#window signal
# window = np.hamming(len(signal))
# signal *= window

# detrend signal|


signal = signal-np.mean(signal)

plt.plot(t,signal)
plt.show()
sfft = np.fft.fft(signal)
sfft = 2*np.abs(sfft)[0:len(signal)//2]
sfft /= len(signal)
sfreq = np.fft.fftfreq(len(signal),T/len(signal))[0:len(signal)//2]
widths = np.linspace(0.25,1,10)
peaks = find_peaks_cwt(sfft,widths)
peaks = [int(p) for p in peaks]
plt.plot(sfreq,sfft)
plt.vlines(1,0,np.max(sfft),color = 'k')
peakprominences,_,_ = peak_prominences(sfft,peaks)
print(sfreq[peaks],peakprominences)
plt.show()
#%%
import numpy as np
from scipy import signal, datasets, ndimage
rng = np.random.default_rng()
face = datasets.face(gray=True) - datasets.face(gray=True).mean()
face = ndimage.zoom(face[30:500, 400:950], 0.5)  # extract the face
template = np.copy(face[135:165, 140:175])  # right eye
template -= template.mean()
face = face + rng.standard_normal(face.shape) * 50  # add noise
corr = signal.correlate2d(face, template, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

import matplotlib.pyplot as plt
fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
                                                    figsize=(6, 15))
ax_orig.imshow(face, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_template.imshow(template, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()
ax_corr.imshow(corr, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
ax_orig.plot(x, y, 'ro')
fig.show()