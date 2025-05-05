# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True, profile=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cupy as cp
import numpy as np
from cupyx.scipy import interpolate
from cupyx.scipy.signal import butter, sosfiltfilt
import os





def figure_limits(series, framerate, start, end):
    """Determine optimal start and end indices for analysis.
    
    Args:
        series: Input 3D array with dimensions (time, height, width)
        framerate: Sampling rate in Hz
        start: Initial start index
        end: Initial end index
        
    Returns:
        tuple: (start, end, nperiods)
    """
        
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
    max_idx = np.argmax(outlierdiffs)
    selectedrange = np.array([outliers[max_idx]+1, outliers[max_idx+1]-1], dtype=np.int64)
    
    start = int(selectedrange[0])
    end = int(selectedrange[1])
    
    nperiods = (end-start)/framerate
    nperiods = np.floor(nperiods)
    
    return start, end, nperiods


# Optimized interpolation chunk processor


# Optimized main function
def get_AC_data(images, framerate=16.7,  tolerance=0.2, 
                 frequency=1,  periods=10,  start=150,  end=399, 
                 interpolation=True,   hardlimits=False,  filt=True):
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
    
    
    # Copy the input array only once
    series = np.ascontiguousarray(images)
    
    # Get optimal start and end indices
    if not hardlimits:
        start, end, nperiods = figure_limits(series, framerate, start, end)
    else:
        nperiods = periods
    # Copy the series to GPU
    series = cp.asarray(series, dtype=np.float32)
    # Adjust end index based on number of periods
    newlastindex = int(round(nperiods * framerate))
    while (newlastindex + start > end):
        nperiods -= 1
        newlastindex = int(round(nperiods * framerate))
    
    end = newlastindex + start
    
    # Slice the original series to reduce memory usage
    series = series[start:end, :, :]
    
    # Calculate DC array (mean over time)
    DCarray = cp.mean(series, axis=0, dtype=np.float32)
    
    # Apply interpolation if requested
    if interpolation:
        currentinterval_s = 1/framerate
        desiredinterval_s = 1/round(framerate)
        N = series.shape[0]
        newN = int(round(N*currentinterval_s/desiredinterval_s))
        synthetic_t = cp.linspace(0, (newN)*desiredinterval_s, newN,dtype=np.float32)
        real_t = cp.linspace(0, N*currentinterval_s, N,dtype=np.float32)
        spl = interpolate.PchipInterpolator(real_t,series, axis=0)
        series = spl(synthetic_t)
    
    # Apply bandpass filtering if requested
    if filt:
        butterfilt = butter(4, btype='bandpass', fs=framerate, Wn=cp.array([0.1, 6], dtype=np.float32), output='sos')
        series = sosfiltfilt(butterfilt, series, axis=0)
    else:
        series = series - DCarray
    # Calculate AC image
    series_fft = cp.fft.fft(series, axis=0)
    nfft = series_fft.shape[0]
    
    # Only compute half the spectrum (positive frequencies)
    series_fft_abs = 2*cp.abs(series_fft)[0:nfft//2].astype(np.float32)
    series_fft_abs /= nfft
    
    # Compute frequencies only once
    series_frequencies = cp.fft.fftfreq(nfft, 1/framerate)[0:nfft//2].astype(np.float32)
    
    # Find the frequency index closest to target - use more efficient approach
    good_freq = cp.abs(series_frequencies - frequency).argmin()
    
    # Extract the AC image for this chunk
    ACarray = series_fft_abs[good_freq, :, :]
    
    # Create time array for output
    time = np.linspace(0, series.shape[0]/framerate, series.shape[0], dtype=np.float32)
    
    series = cp.asnumpy(series)
    ACarray = cp.asnumpy(ACarray)
    DCarray = cp.asnumpy(DCarray)
    return ACarray.astype(np.float64), DCarray.astype(np.float64), (time, series), (start, end)