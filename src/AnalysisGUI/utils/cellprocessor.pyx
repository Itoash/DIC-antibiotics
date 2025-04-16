# distutils: language=c++
# cython: boundscheck=False, nonecheck=False, cdivision=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from libcpp cimport bool as cpp_bool
import numpy as np
cimport numpy as np
import concurrent.futures
import cv2
import os
from libc.stdlib cimport malloc, free
from numpy cimport ndarray
import time as tm
from typing import Dict, List, Tuple, Any
import dill as pickle
# Define constants for feature names to avoid string lookups
FEATURE_NAMES = ['idx', 'times', 'position', 'area', 'Ellipse angle', 'Ellipse major',
               'Ellipse minor', 'DC mean', 'DC min', 'DC max', 'AC mean', 'AC min', 'AC max',
               'AC interior area', 'AC interior mean', 'AC contour mean',
               'AC interior/back contrast', 'AC interior/contour contrast', 'AC solidity', 
               'Interior contour', 'Total contour']

# Add proper boolean type for numpy arrays
ctypedef np.npy_bool BOOL_t
ctypedef np.int32_t INT32_t
ctypedef np.float64_t FLOAT64_t

def process_frame_py(frame_idx, AC, DC, labels):
    return process_frame(frame_idx, AC, DC, labels)
cpdef dict process_cells(dict celldict, list ACs, list DCs, list labels, np.ndarray times):
    """
    Process all cells in parallel using multiprocessing.
    
    Args:
        celldict: Dictionary mapping cell names to lineage information
        ACs: List of AC images
        DCs: List of DC images
        labels: List of label images
        times: Array of time points
        
    Returns:
        Dictionary containing processed cell data
    """
    cdef dict cells = {}
    cdef str cellname
    cdef list cell_list = []  # Fixed: Initialize the list
    cdef dict frame_labels_dict = {}
    cdef int num_workers = max(1, os.cpu_count() - 1)  # Leave one core free
    
    # Process frames first and collect results
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:  # Fixed: Typo in ProcessPoolExecutor
        # Process each frame in parallel
        futures = [executor.submit(process_frame_py, frame_idx, ACs[frame_idx], DCs[frame_idx], labels[frame_idx]) 
                 for frame_idx in range(len(ACs))]
        
        # Collect results and convert to dictionary
        for future in concurrent.futures.as_completed(futures):
            try:
                frame_results = future.result()
                frame_labels_dict.update(frame_results)  # Merge dictionaries
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
    
    # Now process cell lineages with the complete frame_labels_dict
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for cellname, frame_labels in celldict.items():
            cell_list.append(executor.submit(
                process_cell_lineage, cellname, frame_labels, times, frame_labels_dict))
        
        for future in concurrent.futures.as_completed(cell_list):
            try:
                cellname, cell_data = future.result()
                cells[cellname] = cell_data
                # Print progress every 10 cells
                if len(cells) % 10 == 0:
                    print(f"Processed {len(cells)}/{len(celldict)} cells")
            except Exception as e:
                print(f"Error processing cell lineage: {str(e)}")
    
    return cells


cdef tuple process_cell_lineage(str cellname, dict lineage_dict, np.ndarray times, dict label_results):
    """
    Process a single cell lineage by aggregating features from different frames.
    
    Args:
        cellname: Name of the cell
        lineage_dict: Dictionary mapping frames to cell labels
        times: Array of time points
        label_results: Dictionary of precomputed features for each (frame, label) pair
        
    Returns:
        Tuple of (cellname, cell_data)
    """
    cdef dict cell_data = {feature: [] for feature in FEATURE_NAMES}
    cdef float tic = tm.time()
    cdef int frame_idx, label_val
    cdef list features
    cdef int i
    
    # Initialize times array to match lineage entries
    cell_data['times'] = np.array([times[frame] for frame, _ in lineage_dict.items()])
    cell_data['idx'] = list(lineage_dict.keys())
    
    # Link up the lineage dict with the feature dict
    for frame, label_val in lineage_dict.items():
        key = (int(frame), int(label_val))
        
        # Check if this (frame, label) pair exists in our results
        if key not in label_results:
            print(f"Warning: Missing data for frame {frame}, label {label_val}")
            continue
            
        features = label_results[key]
        
        # Append each feature to the corresponding list
        cell_data['position'].append(features[0])
        
        # For features 3 through n-2, use enumerate with starting index
        for i, feature_name in enumerate(FEATURE_NAMES[3:-2]):
            cell_data[feature_name].append(features[i+1])
        
        # Last two features are contours
        cell_data['Interior contour'].append(features[-2])
        cell_data['Total contour'].append(features[-1])
    
    # Convert lists to numpy arrays for numerical features
    for feature in FEATURE_NAMES[1:-2]:
        if feature in cell_data and len(cell_data[feature]) > 0:
            cell_data[feature] = np.array(cell_data[feature])
    return (cellname, cell_data)


cdef dict process_frame(int frame_idx, np.ndarray AC, np.ndarray DC, np.ndarray labels):
    """
    Process all labels in a single frame.
    
    Args:
        frame_idx: Index of the frame
        AC: AC image
        DC: DC image
        labels: Label image
        
    Returns:
        Dictionary mapping (frame, label) pairs to features
    """
    cdef dict result = {}
    cdef float col
    cdef list features
    cdef np.ndarray unique_labels
    
    try:
        unique_labels = np.unique(labels.flatten())
        
        # Skip background (label 0)
        for col in unique_labels:
            if col == 0:
                continue
                
            features = process_label(AC, DC, labels, col)
            result[(int(frame_idx), int(col))] = features
            
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        
    return result


cdef list process_label(np.ndarray AC, np.ndarray DC, np.ndarray label, float col):
    """
    Process a single label in an image to extract features.
    
    Args:
        AC: AC image
        DC: DC image
        label: Label image
        col: Label value to process
        
    Returns:
        List of computed features
    """
    # Create mask for this label
    cdef np.ndarray[BOOL_t, ndim=2] mask = (label == float(col))
    cdef np.ndarray[FLOAT64_t, ndim=2] filtered = np.zeros_like(label, dtype=np.float64)
    cdef np.ndarray[BOOL_t, ndim=2] background_mask = (label == 0)
    
    # Initialize feature variables
    cdef np.ndarray[FLOAT64_t, ndim=1] com = np.array([0.0, 0.0])
    cdef double area = 0.0
    cdef double angle = 0.0
    cdef double major = 0.0
    cdef double minor = 0.0
    cdef double DCmean = 0.0
    cdef double DCmin = 0.0
    cdef double DCmax = 0.0
    cdef double ACmean = 0.0
    cdef double ACmin = 0.0
    cdef double ACmax = 0.0
    cdef double interiorarea = 0.0
    cdef double interiormean = 0.0
    cdef double contourmean = 0.0
    cdef double intbackcontrast = 0.0
    cdef double intcontcontrast = 0.0
    cdef double solidity = 0.0
    cdef np.ndarray intcontour = np.array([[0, 0]])
    cdef np.ndarray contour = np.array([[0, 0]])
    
    # Local variables for processing
    cdef tuple contour_list
    cdef np.ndarray hierarchy
    cdef np.ndarray coords
    cdef object ellipse
    cdef np.ndarray DC_values, AC_values
    cdef list intcontours
    cdef np.ndarray interior_mask
    cdef float denom1, denom2
    cdef np.ndarray background_values, interior_values, contour_values
    cdef np.ndarray contour_mask
    cdef double background

    # Create binary mask for contour detection
    filtered[mask] = 1
    coords = np.argwhere(mask)

    if len(coords) > 0:
        # Calculate center of mass and area
        com = np.mean(coords, axis=0)
        area = len(coords)

        if area > 1:
            # Find contours
            contour_list, hierarchy = cv2.findContours(
                filtered.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if contour_list and len(contour_list) > 0:
                contour = contour_list[0].reshape(-1, 2)
                
                # Fit ellipse if enough points
                if len(contour) > 4:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                    major = max(ellipse[1])
                    minor = min(ellipse[1])
                    
                # Close the contour
                if len(contour) > 0:
                    start = contour[0, :].reshape(1, 2)
                    contour = np.append(contour, start, axis=0)
        else:
            # Default values for small regions
            angle = 0.0
            major = 1.0
            minor = 1.0

        # Process DC values
        DC_values = DC[mask]
        if len(DC_values) > 0:
            DCmean = np.mean(DC_values)
            DCmin = np.min(DC_values)
            DCmax = np.max(DC_values)

        # Process AC values
        AC_values = AC[mask]
        if len(AC_values) > 0:
            ACmean = np.mean(AC_values)
            ACmin = np.min(AC_values)
            ACmax = np.max(AC_values)

            # Find interior regions
            intcontours, interior_mask = findInterior(AC, mask)
            interiorarea = np.count_nonzero(interior_mask)
            
            if interiorarea > 0:
                # Calculate interior statistics
                interior_values = AC[interior_mask]
                interiormean = np.mean(interior_values) if len(interior_values) > 0 else 0

                # Calculate background statistics
                background_values = AC[background_mask]
                background = np.mean(background_values) if len(background_values) > 0 else 0

                # Calculate contour statistics
                contour_mask = mask & ~interior_mask
                contour_values = AC[contour_mask]
                contourmean = np.mean(contour_values) if len(contour_values) > 0 else 0

                # Calculate contrast metrics
                denom1 = interiormean + background
                if denom1 != 0:
                    intbackcontrast = (interiormean - background) / denom1

                denom2 = interiormean + contourmean
                if denom2 != 0:
                    intcontcontrast = (interiormean - contourmean) / denom2

                solidity = interiorarea / area if area > 0 else 0
    
    return [com, area, angle, major, minor, DCmean, DCmin, DCmax,
            ACmean, ACmin, ACmax, interiorarea, interiormean, contourmean,
            intbackcontrast, intcontcontrast, solidity, intcontours, contour]


cdef tuple findInterior(np.ndarray[FLOAT64_t, ndim=2] image, np.ndarray[BOOL_t, ndim=2] cell_mask, 
                      float min_area_ratio=0.05, float max_area_ratio=1.0):
    """
    Find interior regions within a cell mask based on intensity thresholding.
    
    Args:
        image: Intensity image
        cell_mask: Binary mask of the cell region
        min_area_ratio: Minimum area ratio threshold for interior regions
        max_area_ratio: Maximum area ratio threshold for interior regions
        
    Returns:
        Tuple of (interior_contours, interior_mask)
    """
    # Compute areas and thresholds once
    cdef int cell_area = np.count_nonzero(cell_mask)
    cdef int min_area = <int>(cell_area * min_area_ratio)
    cdef int max_area = <int>(cell_area * max_area_ratio)
    
    # Pre-allocate arrays
    cdef np.ndarray[INT32_t, ndim=2] interior_mask = np.zeros_like(cell_mask, dtype=np.int32)
    cdef np.ndarray[FLOAT64_t, ndim=2] masked_image = np.zeros_like(image)
    
    # Create masked image with vectorized operations (faster than iterating)
    masked_image = np.where(cell_mask, image, 0)
    
    # Calculate mean intensity more efficiently using numpy directly
    cdef double mean_intensity = 0.0
    cdef np.ndarray cell_values = image[cell_mask]
    if len(cell_values) > 0:
        mean_intensity = np.mean(cell_values)
    
    # Try adaptive threshold finding with binary search
    cdef double high_thresh = np.max(masked_image) if np.max(masked_image) > 0 else mean_intensity
    cdef double low_thresh = mean_intensity/1.5
    cdef double mid_thresh = mean_intensity
    cdef tuple contours
    cdef np.ndarray thresholded
    cdef int max_iterations = 15  # More iterations for better convergence
    cdef int iteration = 0
    cdef double target_area = cell_area * 0.5  # Target ~50% of cell area for interior
    cdef double current_area = 0
    cdef double prev_thresh = -1
    
    # First try with mean intensity
    _, thresholded = cv2.threshold(masked_image, mean_intensity, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found or too small/large, use binary search to find appropriate threshold
    if not contours or cv2.contourArea(contours[0]) < min_area or cv2.contourArea(contours[0]) > max_area:
        while iteration < max_iterations and abs(high_thresh - low_thresh) > 0.01:
            iteration += 1
            
            # Avoid repeat calculations
            if mid_thresh == prev_thresh:
                break
                
            prev_thresh = mid_thresh
            mid_thresh = (high_thresh + low_thresh) / 2.0
            
            _, thresholded = cv2.threshold(masked_image, mid_thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                high_thresh = mid_thresh
                continue
                
            current_area = sum(cv2.contourArea(c) for c in contours)
            
            # Adjust threshold based on area
            if current_area < target_area:
                high_thresh = mid_thresh
            else:
                low_thresh = mid_thresh
    
    # Process contours efficiently
    cdef list interior_contours = []
    cdef double area
    cdef np.ndarray cont, start, cont_reshaped
    
    for cont in contours:
        area = cv2.contourArea(cont)
        if min_area <= area <= max_area:
            interior_contours.append(cont)
            cv2.drawContours(interior_mask, [cont], -1, 255, -1)
    
    # Pre-allocate results list and process contours
    cdef list results = []
    
    for cont in interior_contours:
        cont_reshaped = cont.reshape(cont.shape[0], 2)
        if len(cont_reshaped) > 0:
            start = cont_reshaped[0].reshape(1, 2)
            results.append(np.vstack((cont_reshaped, start)))
    
    return results, interior_mask.astype(bool)