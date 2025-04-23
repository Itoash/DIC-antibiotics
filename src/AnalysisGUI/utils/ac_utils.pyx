# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from libc.math cimport floor, round, abs
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import cv2
from scipy.signal import argrelextrema
from omnipose.utils import normalize99
from copy import deepcopy
from cellpose_omni import models
import time
from scipy import interpolate
from scipy.signal import detrend, butter, sosfiltfilt
import concurrent.futures
import os
import time as tm
from numpy cimport ndarray
from functools import cache
import os


cdef parallel_bandpass3D(np.ndarray[double, ndim=3] series, double framerate, np.ndarray frequencies=np.array([0.1, 6]), int num_workers=8):
    """Parallelize 3D bandpass filter by splitting the spatial dimensions
    
    Args:
        series: Input 3D array with dimensions (time, height, width)
        framerate: Original sampling rate in Hz
        frequencies: Cutoff frequencies to use
        num_workers: Number of parallel workers
        
    Returns:
        tuple: (interpolated_series, new_framerate)
    """
    cdef:
        np.ndarray[double, ndim=3] filtered_series
        np.ndarray[double,ndim=2] sos
        int height = series.shape[1]
        int chunk_size
        int start_row, end_row
        list chunks = []
        list framerates = []
        list cutoffs = []
        list filters = []
        int i
        
    # Create filter
    sos = butter(4, btype='bandpass', fs=framerate, Wn=frequencies, output='sos')
    # Create empty output array
    filtered_series = np.zeros((series.shape[0], series.shape[1], series.shape[2]), dtype=np.float64)
    
    # Split the image into horizontal chunks
    chunk_size = max(1, height // num_workers)
    
    # Prepare chunks of data for parallel processing
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        # Each chunk contains (chunk_data, real_time, synthetic_time)
        chunks.append(series[:, start_row:end_row, :])
        framerates.append(framerate)
        cutoffs.append(frequencies)
        filters.append(sos)
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(filter_chunk, chunks,framerates,cutoffs,filters))
    
    # Combine results
    current_idx = 0
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        filtered_series[:, start_row:end_row, :] = results[current_idx]
        current_idx += 1
    
    return filtered_series

cdef filter_chunk(np.ndarray[double,ndim=3] chunk,double framerate,np.ndarray[double,ndim=1] frequencies,np.ndarray[double,ndim=2] sos):
    
    cdef np.ndarray[double, ndim=3] filtered_chunk = sosfiltfilt(sos, chunk, axis=0)
    return filtered_chunk



    
cdef figure_limits(np.ndarray[double, ndim=3] series, double framerate, int start, int end):
    """Determine optimal start and end indices for analysis.
    
    Args:
        series: Input 3D array with dimensions (time, height, width)
        framerate: Sampling rate in Hz
        start: Initial start index
        end: Initial end index
        
    Returns:
        tuple: (start, end, nperiods)
    """
    cdef:
        np.ndarray[double, ndim=1] time = np.linspace(0, series.shape[0]/framerate, series.shape[0])
        np.ndarray[double, ndim=1] meanseries = np.mean(np.mean(series, axis=1), axis=1)
        np.ndarray[long, ndim=1] defaults = np.array([start, end], dtype=int)
        np.ndarray[double, ndim=1] diff
        np.ndarray[long, ndim=1] outliers
        np.ndarray[long, ndim=1] outlierdiffs
        np.ndarray[long, ndim=1] selectedrange
        double nperiods
        int i
    
    diff = np.diff(meanseries)
    outliers = np.argwhere(abs(diff) > 3*np.std(meanseries)).flatten().astype(long)
    outliers = np.append(outliers, defaults)
    outliers = np.sort(outliers)
    outlierdiffs = np.diff(outliers).astype(long)
    
    # Find the maximum difference between outliers
    cdef long max_idx = np.argmax(outlierdiffs)
    selectedrange = np.array([outliers[max_idx]+1, outliers[max_idx+1]-1]).astype(long)
    
    start = int(selectedrange[0])
    end = int(selectedrange[1])
    
    nperiods = (end-start)/framerate
    nperiods = floor(nperiods)
    
    return start, end, nperiods
    
cdef interpolate_chunk(tuple chunk_data):
    """Process a single chunk for interpolation"""
    cdef:
        np.ndarray[double,ndim=3] series_chunk = chunk_data[0]
        np.ndarray[double,ndim=1] real_t = chunk_data[1]
        np.ndarray[double,ndim=1] synthetic_t = chunk_data[2]
        np.ndarray[double,ndim=3] interp_chunk
    cdef object spl = interpolate.PchipInterpolator(real_t, series_chunk, axis=0)
    
    interp_chunk = spl(synthetic_t)
    return interp_chunk

cdef parallel_interpolate3D(np.ndarray[double, ndim=3] series, double framerate, int num_workers=8):
    """Parallelize 3D interpolation by splitting the spatial dimensions
    
    Args:
        series: Input 3D array with dimensions (time, height, width)
        framerate: Original sampling rate in Hz
        num_workers: Number of parallel workers
        
    Returns:
        tuple: (interpolated_series, new_framerate)
    """
    cdef:
        double currentinterval_s = 1/framerate
        double desiredinterval_s = 1/np.round(framerate)
        int N = series.shape[0]
        int newN = int(round(N*currentinterval_s/desiredinterval_s))
        np.ndarray[double, ndim=3] interpseries
        np.ndarray[double, ndim=1] synthetic_t
        np.ndarray[double, ndim=1] real_t
        int height = series.shape[1]
        int chunk_size
        int start_row, end_row
        list chunks = []
        int i
    synthetic_t = np.linspace(0, (newN)*desiredinterval_s, newN)
    real_t = np.linspace(0, N*currentinterval_s, N)
    
    # Create empty output array
    interpseries = np.zeros((newN, series.shape[1], series.shape[2]), dtype=np.float64)
    
    # Split the image into horizontal chunks
    chunk_size = max(1, height // num_workers)
    
    # Prepare chunks of data for parallel processing
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        # Each chunk contains (chunk_data, real_time, synthetic_time)
        chunks.append((series[:, start_row:end_row, :], real_t, synthetic_t))
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(interpolate_chunk, chunks))
    
    # Combine results
    current_idx = 0
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        interpseries[:, start_row:end_row, :] = results[current_idx]
        current_idx += 1
    
    return interpseries, 1/desiredinterval_s

# Function to process FFT on a chunk of data
cdef process_fft_chunk(tuple chunk_data):
    """Process FFT on a single spatial chunk"""
    cdef:
        np.ndarray[double,ndim=3] series_chunk = chunk_data[0]
        double framerate = chunk_data[1]
        double frequency = chunk_data[2]
        int N = chunk_data[3]
        np.ndarray[complex,ndim=3] series_fft
        np.ndarray[double,ndim=3] series_fft_abs
        np.ndarray[double,ndim=1] series_frequencies
        int good_freq
        int nfft
        np.ndarray[double,ndim=2] ACarray_chunk
        
    
    # Calculate FFT on this chunk
    series_fft = np.fft.fft(series_chunk, axis=0)
    nfft = series_fft.shape[0]
    series_fft_abs = 2*np.abs(series_fft)[0:nfft//2]
    series_fft_abs /= nfft
    series_frequencies = np.fft.fftfreq(nfft, 1/framerate)[0:nfft//2]
    
    # Find the frequency index closest to target
    good_freq = np.abs(series_frequencies - frequency).argmin()
    
    # Get the AC image for this chunk
    ACarray_chunk = series_fft_abs[good_freq, :, :]
    return ACarray_chunk

cdef parallel_getACimage(np.ndarray[double, ndim=3] series, double framerate, double frequency, int num_workers=8):
    """Extract amplitude-coupling image at a specific frequency using parallel processing.
    
    Args:
        series: Input 3D array with dimensions (time, height, width)
        framerate: Sampling rate in Hz
        frequency: Target frequency in Hz
        num_workers: Number of parallel workers
        
    Returns:
        2D array representing the AC image at the target frequency
    """
    cdef:
        int N = series.shape[0]
        int height = series.shape[1]
        int width = series.shape[2]
        int chunk_size
        int start_row, end_row
        np.ndarray[double, ndim=2] full_ACarray
        list chunks = []
        int i
    # Split the image into horizontal chunks
    chunk_size = max(1, height // num_workers)
    
    # Prepare chunks of data
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        # Each chunk contains (data_chunk, framerate, frequency, N)
        chunks.append((series[:, start_row:end_row, :], framerate, frequency, N))
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_fft_chunk, chunks))
    
    # Combine results
    full_ACarray = np.zeros((height, width), dtype=np.float64)
    current_idx = 0
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        full_ACarray[start_row:end_row, :] = results[current_idx]
        current_idx += 1
    
    return full_ACarray


# Updated main function to use parallel processing
cpdef get_AC_data(np.ndarray[double, ndim=3] images, double framerate=16.7, double tolerance=0.2, 
                double frequency=1, double periods=10, int start=150, int end=399, 
                bint interpolation=True, bint detr=True, bint hardlimits=False, bint filt=True,
                int num_workers=32):
    """
    Extract amplitude coupling data from time series images with parallel processing.
    
    Args:
        images: Input 3D array with dimensions (time, height, width)
        framerate: Sampling rate in Hz
        tolerance: Frequency tolerance
        frequency: Target frequency in Hz
        periods: Number of periods to analyze
        start: Start index
        end: End index
        interpolation: Whether to interpolate the time series
        detr: Whether to detrend the time series
        hardlimits: Whether to use hard limits for start and end
        filt: Whether to apply bandpass filtering
        num_workers: Number of parallel workers
        
    Returns:
        tuple: (ACarray, DCarray, (time, series), (start, end))
    """
    cdef:
        np.ndarray[double, ndim=3] series = images.copy()
        np.ndarray[double, ndim=2] DCarray
        np.ndarray[double, ndim=2] ACarray
        np.ndarray[double, ndim=1] time
        np.ndarray[double, ndim=1] meanseries
        int newlastindex
        
    num_workers = os.cpu_count()
    if not hardlimits:
        start, end, nperiods = figure_limits(series, framerate, start, end)
    else:
        nperiods = periods
   
    
    newlastindex = int(round(nperiods * framerate))
    while (newlastindex + start > end):
        nperiods -= 1
        newlastindex = int(round(nperiods * framerate))
    
    end = newlastindex + start
    series = series[start:end, :, :]
    DCarray = np.mean(series, axis=0)
    
    
    if interpolation:
        series, framerate = parallel_interpolate3D(series, framerate, num_workers)
    
    
    
    
    if detr:
        series = detrend(series, axis=0)
    
    
    
    if filt:
        series = parallel_bandpass3D(series, framerate)
    
    
    
    ACarray = parallel_getACimage(series, framerate, frequency, num_workers)
    
    # Calculate mean intensity time series
    meanseries = np.mean(np.mean(series, axis=1), axis=1)
    time = np.linspace(0, len(meanseries)/framerate, len(meanseries))
    
    return ACarray, DCarray, (time, series), (start, end)

        