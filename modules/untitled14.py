#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:45:11 2025

@author: victorionescu
"""

import numpy as np
from os import listdir
import os
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
from scipy.stats import linregress
from scipy.signal import find_peaks_cwt,detrend,butter,sosfiltfilt
from scipy.signal import peak_prominences,correlate,detrend
from scipy.ndimage import convolve
import concurrent.futures
import matplotlib as mpl
import matplotlib.pyplot as plt

def segmentDComni(imgs):
    # read images in a list!!!
    
    
    # print some info about the images.
    for i in imgs:
        print('Original image shape:',i.shape)
        print('data type:',i.dtype)
        print('data range: min {}, max {}\n'.format(i.min(),i.max()))
    nimg = len(imgs)
    print('\number of images:',nimg)
    
    print('\n')
    
            
    # invert images (16-bit) 
    # initialize model
    model_name = 'bact_phase_omni'
    model = models.CellposeModel( model_type=model_name)
    
    chans = [0,0] #this means segment based on first channel, no second channel 
    
    
    n = range(nimg) # segment all images in list
    
    # define parameters
    params = {'channels':chans, # always define this with the model
              'rescale': None, # upscale or downscale your images, None = no rescaling 
              'mask_threshold': -2, # erode or dilate masks with higher or lower values between -5 and 5 
              'flow_threshold': 1, # default is .4, but only needed if there are spurious masks to clean up; slows down output
              'transparency': False, # transparency in flow output
              'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
              'cluster': False, # use DBSCAN clustering
              'resample': True, # whether or not to run dynamics on rescaled grid or original grid 
              'verbose': False, # turn on if you want to see more output 
              'tile': True, # average the outputs from flipped (augmented) images; slower, usually not needed 
              'niter': 5, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
              'augment': True, # Can optionally rotate the image and average network outputs, usually not needed 
              'affinity_seg': True, # new feature, stay tuned...
              
             }
    
    tic = time.time() 
    masks, flows, styles = model.eval([imgs[i] for i in n],**params)
    
    net_time = time.time() - tic
    
    print('total segmentation time: {}s'.format(net_time))
    
    return masks

sorted_files = [f for f in listdir('/Users/victorionescu/Desktop/Segtest') if '.tif' in f]
sorted_files = list(sorted(sorted_files,key=lambda x: int(x[-8:-4])))
images = np.empty(len(sorted_files), dtype=object)
for n in range(0, len(sorted_files)):
  images[n] = cv2.imread( '/Users/victorionescu/Desktop/Segtest/'+sorted_files[n],-1 ) 
dims = np.shape(images[0])
images_copy = np.empty((len(images),dims[0],dims[1]))

for i in range(len(images)):
    images_copy[i,:,:] = images[i]
masks = segmentDComni(images)
#%%
for i,m in zip(images,masks):
    fig,ax = plt.subplots(2)
    ax[0].imshow(i,cmap='Greys')
    ax[1].imshow(m,cmap='rainbow')
    plt.show()
    