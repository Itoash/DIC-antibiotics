#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:31:37 2025

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
from scipy.signal import medfilt,argrelmax,iirfilter
f = 1
N = 400
framerate = 16
DCoffset = 5000
ramp = 0.5
t = np.linspace(0,N/framerate,N)
A = 5
def addrandomsins(t,signal,finterval = [0,30],ainterval = [0,5],N=10):
    
    for i in range(N):
        f = np.random.uniform(finterval[0],finterval[1])
        A = np.random.uniform(ainterval[0],ainterval[1])
        
        sine = A*np.sin(2*np.pi*f*t) 
        signal += sine
    
def addrandompolynomial(t,signal,coefflower = -0.0001,coeffupper = 0.0001):
    coeffs = np.random.uniform(coefflower,coeffupper,5)
    poly = coeffs[0]*t**4+coeffs[1]*t**3+coeffs[2]*t**2+coeffs[3]*t**1+coeffs[4]
    signal = signal+poly
    return signal

heights = []
heightstds = []
for k in range(0,10):
    heights = []
    for i in range(1000):
        signal=0
        #signal = DCoffset+A*np.sin(2*np.pi*f*t)
        #addrandomsins(t,signal)
        signal = addrandompolynomial(t,signal,-0.0001*k/2,0.0001*k/2)
        #signal = signal+ np.random.uniform(-A,A,len(t))
       
        fft = np.fft.fft(signal)
        fft = 2*np.abs(fft[0:N//2])/N
        freqs = np.fft.fftfreq(N,1/framerate)[0:N//2]
        if i ==0:
            plt.plot(t,signal)
            plt.show()
            plt.plot(freqs[1:],fft[1:])
            plt.show()
        fheight = 0
        good_freq=freqs.flat[np.abs(freqs - f).argmin()]
        good_freq = np.argwhere(freqs==good_freq)[0]
        fheight = fft[good_freq]
        heights.append(fheight[0])


    # plt.hist(heights,bins=100)
    # plt.show()
    heightstds.append(np.std(heights))

print(heightstds)

#%%

f = 1
N = 100
framerate = 16
DCoffset = 5000
ramp = 0.5
t = np.linspace(0,N/framerate,N)
A = 1
signal = np.zeros((N,200,200))
for i in range(200):
    for j in range(200):
        newsig=  A*np.sin(2*np.pi*t+np.random.uniform(0,np.pi))
        
        signal[:,i,j] = newsig
        
def computecorrelation(signalmatrix,reference):
    if signalmatrix.shape[0] != reference.shape[0]:
        print("Different lengths for signals!")
        return signalmatrix,np.mean(signalmatrix,axis=(1,2))
    if len(reference.shape)==1:
        corrmat = np.sum(np.multiply(signalmatrix,reference[:,np.newaxis,np.newaxis]),axis = 0)
    else:
        corrmat = np.sum(np.multiply(signalmatrix,reference),axis = 0)
    
    corrint = np.sum(signalmatrix**2,axis = 0)
    corrint[corrint==0]=1
    bckgint = np.sum(reference**2)
    if bckgint == 0:
        bckgint =1
    corrmat /= np.sqrt(corrint*bckgint)
    signalcorr = [] 
    signalcorr.append(np.mean(corrmat[:25,:25]))
    signalcorr.append(np.mean(corrmat[-25:,:25]))
    signalcorr.append(np.mean(corrmat[:25,-25:]))
    signalcorr.append(np.mean(corrmat[-25:,-25:])) 
    print(f"corrmatrix max is {np.max(corrmat)}")
    print(f"corrmatrix min is {np.min(corrmat)}")
    corrmat = np.arccos(corrmat)
    return corrmat,signalcorr

ref = A*np.sin(2*np.pi*t)
corrmat,_ = computecorrelation(signal,ref)
plt.imshow(corrmat)
print(np.min(corrmat))
print(np.max(corrmat))
print(np.mean(corrmat))
    
    