# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True, profile=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from libc.math cimport floor, round, abs
from libc.stdlib cimport malloc, free
from scipy import interpolate
from scipy.signal import detrend, butter, sosfiltfilt
import concurrent.futures
import os
from numpy cimport ndarray
from functools import cache
import os

# Pre-compute and cache filter coefficients
@cache
def cached_butter_filter(float framerate, float start, float end):
    """Cache butter filter coefficients for reuse"""
    return butter(4, btype='bandpass', fs=framerate, Wn=np.array([start, end], dtype=np.float32), output='sos')


cdef ndarray parallel_bandpass3D(ndarray[float, ndim=3] series, float framerate, float start = 0.1, float end = 6, int num_workers=8):
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
        ndarray[float, ndim=3] filtered_series
        ndarray[float, ndim=2] sos
        int height = series.shape[1]
        int chunk_size
        int start_row, end_row
        list chunks = []
        list filters = []
        int i
        
    # Create filter - now using cached version
    sos = cached_butter_filter(framerate, start, end).astype(np.float32)
    
    # Create empty output array with same memory layout as input
    filtered_series = np.zeros_like(series, dtype=np.float32)
    
    # Split the image into horizontal chunks
    chunk_size = max(1, height // num_workers)
    
    # Prepare chunks of data for parallel processing
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        # Each chunk contains (chunk_data, real_time, synthetic_time)
        chunks.append(series[:, start_row:end_row, :])
        filters.append(sos)
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(filter_chunk, chunks, filters))
    
    # Combine results
    current_idx = 0
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        filtered_series[:, start_row:end_row, :] = results[current_idx]
        current_idx += 1
    
    return filtered_series


cdef ndarray filter_chunk(ndarray[float, ndim=3] chunk, ndarray[float, ndim=2] sos):
    """Process filter on a single chunk"""
    # Use direct memory view for better performance
    return sosfiltfilt(sos, chunk, axis=0).astype(np.float32)


cdef tuple figure_limits(ndarray[float, ndim=3] series, float framerate, float frequency, int start, int end):
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
        ndarray[float, ndim=1] time
        ndarray[float, ndim=1] meanseries
        ndarray[long long, ndim=1] defaults
        ndarray[float, ndim=1] diff
        ndarray[long long, ndim=1] outliers
        ndarray[long long, ndim=1] outlierdiffs
        ndarray[long long, ndim=1] selectedrange
        float nperiods
        int i
        
    # Optimized computation of mean series - collapse both spatial dimensions at once
    meanseries = np.mean(np.mean(series, axis=1), axis=1)
    
    # Pre-allocate arrays
    time = np.linspace(0, series.shape[0]/framerate, series.shape[0],dtype=np.float32)
    defaults = np.array([start, end], dtype=np.int64)
    
    # Calculate differences and find outliers
    diff = np.diff(meanseries)
    outliers = np.argwhere(abs(diff) > 3*np.std(meanseries)).flatten().astype(np.int64)
    outliers = np.append(outliers, defaults)
    outliers = np.sort(outliers)
    outlierdiffs = np.diff(outliers).astype(np.int64)
    
    # Find the maximum difference between outliers
    cdef long max_idx = np.argmax(outlierdiffs)
    selectedrange = np.array([outliers[max_idx]+1, outliers[max_idx+1]-1], dtype=np.int64)
    
    start = int(selectedrange[0])
    end = int(selectedrange[1])
    
    nperiods = (end-start)/framerate*frequency
    nperiods = floor(nperiods)
    
    return start, end, nperiods


# Optimized interpolation chunk processor
cdef interpolate_chunk(tuple chunk_data):
    """Process a single chunk for interpolation"""
    cdef:
        ndarray[float, ndim=3] series_chunk = chunk_data[0]
        ndarray[float, ndim=1] real_t = chunk_data[1]
        ndarray[float, ndim=1] synthetic_t = chunk_data[2]
    
    
    return interpolate.pchip_interpolate(real_t,series_chunk,synthetic_t,axis = 0)


cdef parallel_interpolate3D(ndarray[float, ndim=3] series, float framerate, int num_workers=8):
    """Parallelize 3D interpolation by splitting the spatial dimensions
    
    Args:
        series: Input 3D array with dimensions (time, height, width)
        framerate: Original sampling rate in Hz
        num_workers: Number of parallel workers
        
    Returns:
        tuple: (interpolated_series, new_framerate)
    """
    cdef:
        float currentinterval_s = 1/framerate
        float desiredinterval_s = 1/np.round(framerate)
        int N = series.shape[0]
        int newN = int(round(N*currentinterval_s/desiredinterval_s))
        ndarray[float, ndim=3] interpseries
        ndarray[float, ndim=1] synthetic_t
        ndarray[float, ndim=1] real_t
        int height = series.shape[1]
        int chunk_size
        int start_row, end_row
        list chunks = []
        int i
    
    # Pre-compute time arrays
    synthetic_t = np.linspace(0, (newN)*desiredinterval_s, newN,dtype=np.float32)
    real_t = np.linspace(0, N*currentinterval_s, N,dtype=np.float32)
    
    # Create empty output array with proper memory layout
    interpseries = np.zeros((newN, series.shape[1], series.shape[2]), dtype=np.float32)
    
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


# Optimized FFT chunk processor
cdef process_fft_chunk(tuple chunk_data):
    """Process FFT on a single spatial chunk"""
    cdef:
        ndarray[float, ndim=3] series_chunk = chunk_data[0]
        float framerate = chunk_data[1]
        float frequency = chunk_data[2]
        int N = chunk_data[3]
        ndarray[complex, ndim=3] series_fft
        ndarray[float, ndim=3] series_fft_abs
        ndarray[float, ndim=1] series_frequencies
        int good_freq
        int nfft
        ndarray[float, ndim=2] ACarray_chunk
    
    # Calculate RFFT on this chunk (more efficient for real-valued signals)
    series_fft = np.fft.rfft(series_chunk, axis=0)
    nfft = series_chunk.shape[0]  # Original signal length, not FFT length

    # RFFT already gives us only positive frequencies, no need to slice
    series_fft_abs = 2 * np.abs(series_fft).astype(np.float32)
    series_fft_abs /= nfft

    # Handle DC component (should not be doubled)
    series_fft_abs[0] /= 2

    # Compute frequencies for RFFT
    series_frequencies = np.fft.rfftfreq(nfft, 1/framerate).astype(np.float32)
    
    # Find the frequency index closest to target - use more efficient approach
    good_freq = np.abs(series_frequencies - frequency).argmin()
    
    # Extract the AC image for this chunk
    ACarray_chunk = series_fft_abs[good_freq, :, :]
    return ACarray_chunk


cdef parallel_getACimage(ndarray[float, ndim=3] series, float framerate, float frequency, int num_workers=8):
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
        ndarray[float, ndim=2] full_ACarray
        list chunks = []
        int i
    
    # Create output array with correct memory layout
    full_ACarray = np.zeros((height, width), dtype=np.float32)
    
    # Calculate optimal chunk size based on number of workers
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
    current_idx = 0
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        full_ACarray[start_row:end_row, :] = results[current_idx]
        current_idx += 1
    
    return full_ACarray


# Optimized main function
cpdef get_AC_data(ndarray[float, ndim=3] images, float framerate=16.7, float tolerance=0.2, 
                float frequency=1, float periods=10, int start=150, int end=399, 
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
        ndarray[float, ndim=3] series
        ndarray[float, ndim=2] DCarray
        ndarray[float, ndim=2] ACarray
        ndarray[float, ndim=1] time
        ndarray[float, ndim=1] meanseries
        int newlastindex
        float tic, toc
    
    # Use optimal number of workers
    num_workers = min(32, (os.cpu_count() or 1))
    
    # Copy the input array only once
    series = np.ascontiguousarray(images)
    
    # Get optimal start and end indices
    if not hardlimits:
        start, end, nperiods = figure_limits(series, framerate, frequency, start, end)
    else:
        nperiods = periods
    # Adjust end index based on number of periods
    newlastindex = int(round(nperiods/frequency * framerate))
    while (newlastindex + start > end):
        nperiods -= 1
        newlastindex = int(round(nperiods/frequency * framerate))
    
    end = newlastindex + start
    
    # Slice the original series to reduce memory usage
    series = series[start:end, :, :]
    
    # Calculate DC array (mean over time)
    DCarray = np.mean(series, axis=0, dtype=np.float32)
    
    # Apply interpolation if requested
    if interpolation:
        series, framerate = parallel_interpolate3D(series, framerate, num_workers)
    
    # Detrend the time series if requested
    #if detr:
       #series = detrend(series, axis=0)
    
    # Apply bandpass filtering if requested
    if filt:
        series = parallel_bandpass3D(series, framerate, start=0.1, end=6, num_workers=num_workers)
    else:
        series = series - DCarray
    # Calculate AC image
    ACarray = parallel_getACimage(series, framerate, frequency, num_workers)
    
    # Create time array for output
    time = np.linspace(0, series.shape[0]/framerate, series.shape[0], dtype=np.float32)
    
    return ACarray.astype(np.float64), DCarray.astype(np.float64), (time, series), (start, end)