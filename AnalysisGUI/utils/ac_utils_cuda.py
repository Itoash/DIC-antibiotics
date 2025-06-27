# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True, profile=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cupy as cp
import numpy as np
from cupyx.scipy import interpolate
from cupyx.scipy.signal import butter, sosfiltfilt
import os
import gc




def figure_limits(series, framerate, frequency, start, end):
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
    
    nperiods = (end-start)*frequency/framerate
    nperiods = np.floor(nperiods)
    nperiods = int(nperiods)
    
    return start, end, nperiods


def get_AC_chunk(series, framerate=16.7,  tolerance=0.2, 
                 frequency=1,  periods=10,  start=150,  end=399, 
                 interpolation=True,   hardlimits=False,  filt=True):
    
    
    
     # Copy the series to GPU
    series = cp.asarray(series, dtype=np.float32)

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
        series = interpolate.pchip_interpolate(real_t,series,synthetic_t, axis=0)
        del real_t,synthetic_t

    # Apply bandpass filtering if requested
    if filt:
        butterfilt = butter(4, btype='bandpass', fs=framerate, Wn=cp.array([0.1, 6], dtype=np.float32), output='sos')
        series = sosfiltfilt(butterfilt, series, axis=0)
    else:
        series = series - DCarray
    # Calculate AC image
    series_fft_abs = cp.fft.rfft(series, axis=0)
    nfft = series_fft_abs.shape[0]
    
    # Only compute half the spectrum (positive frequencies)
    series_fft_abs = 2/nfft*cp.abs(series_fft_abs).astype(np.float32)
    
    
    # Compute frequencies only once
    series_frequencies = cp.fft.rfftfreq(nfft, 1/framerate).astype(np.float32)
    
    # Find the frequency index closest to target - use more efficient approach
    good_freq = cp.abs(series_frequencies - frequency).argmin()
    
    # Extract the AC image for this chunk
    ACarray = series_fft_abs[good_freq, :, :]
    del series_fft_abs
    # Create time array for output
    time = np.linspace(0, series.shape[0]/framerate, series.shape[0], dtype=np.float32)
    series = cp.asnumpy(series)
    ACarray = cp.asnumpy(ACarray)
    DCarray = cp.asnumpy(DCarray)
    return ACarray.astype(np.float64), DCarray.astype(np.float64), (time, series), (start, end)


# Optimized main function
def get_AC_data(images, framerate=16.7,  tolerance=0.2, 
                frequency=1,  periods=10,  start=150, end=399, 
                interpolation=True, hardlimits=False, filt=True):

    series = np.ascontiguousarray(images)

    # Determine best temporal segment
    if not hardlimits:
        start, end, nperiods = figure_limits(series, framerate, frequency, start, end)
    else:
        nperiods = periods

    newlastindex = int(round(nperiods / frequency * framerate))
    while (newlastindex + start > end):
        nperiods -= 1
        newlastindex = int(round(nperiods / frequency * framerate))

    end = newlastindex + start
    series = series[start:end, :, :]  # Slice time

    t, H, W = series.shape

    if max(series.shape[1],series.shape[2]) < 512:
        return get_AC_chunk(series, framerate=framerate, tolerance=tolerance,
                            frequency=frequency, periods=periods, start=start, end=end,
                            interpolation=interpolation, hardlimits=hardlimits, filt=filt)

    chunk_size = 512

    ACarray = np.empty((H, W), dtype=np.float64)
    DCarray = np.empty((H, W), dtype=np.float64)

    # Placeholder for series reconstruction
    # We'll infer output time length from first chunk
    time_out = None
    series_out = None

    for i in range(0, H, chunk_size):
        i_end = min(i + chunk_size, H)
        for j in range(0, W, chunk_size):
            j_end = min(j + chunk_size, W)

            chunk = series[:, i:i_end, j:j_end]

            AC_chunk, DC_chunk, (time_chunk, processed_chunk), _ = get_AC_chunk(
                chunk, framerate=framerate, tolerance=tolerance,
                frequency=frequency, periods=periods, start=start, end=end,
                interpolation=interpolation, hardlimits=hardlimits, filt=filt
            )

            # Store AC and DC
            ACarray[i:i_end, j:j_end] = AC_chunk
            DCarray[i:i_end, j:j_end] = DC_chunk

            # Initialize output series container once
            if series_out is None:
                T_out = processed_chunk.shape[0]
                series_out = np.empty((T_out, H, W), dtype=np.float32)
                time_out = time_chunk  # Only need to save once

            # Insert processed chunk into output array
            series_out[:, i:i_end, j:j_end] = processed_chunk
            del AC_chunk, DC_chunk, processed_chunk, chunk
            cp._default_memory_pool.free_all_blocks()
            gc.collect()
    return ACarray, DCarray, (time_out, series_out), (start, end)