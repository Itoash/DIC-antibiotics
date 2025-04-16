# distutils: language=c++
# cython: boundscheck=False, nonecheck=False, cdivision=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from libcpp cimport bool
import numpy as np
cimport numpy as np
import concurrent.futures
import cv2
import os
from libc.stdlib cimport malloc, free
from numpy cimport ndarray
import time as tm

cpdef dict process_cells(dict celldict, list ACs,list DCs, list labels, np.ndarray times):
    cdef dict cells = {}
    cdef str cellname
    cdef list futures = []
    cdef dict frame_labels

    with concurrent.futures.ThreadPoolExecutor(max_workers = int(np.sqrt(os.cpu_count()))) as executor:
        for cellname, frame_labels in celldict.items():
            futures.append(executor.submit(
                process_cell_lineage, cellname, frame_labels, ACs, DCs, labels, times))

        for future in concurrent.futures.as_completed(futures):
            cellname, cell_data = future.result()
            cells[cellname] = cell_data

    return cells


cdef tuple process_cell_lineage(str cellname, dict lineage_dict, list ACs, list DCs, list labels, np.ndarray times):
    cdef list feature_names = ['idx', 'times', 'position', 'area', 'Ellipse angle', 'Ellipse major',
                               'Ellipse minor', 'DC mean', 'DC min', 'DC max', 'AC mean', 'AC min', 'AC max',
                               'AC interior area', 'AC interior mean', 'AC contour mean',
                               'AC interior/back contrast', 'AC interior/contour contrast', 'AC solidity', 'Interior contour', 'Total contour']
    cdef dict cell_data = {feature: [] for feature in feature_names}
    cdef list frame_indexes = list(lineage_dict.keys())
    cdef list results = []
    cdef list future_results
    cdef int frame_idx
    cdef dict result
    cdef list features
    cdef int i
    tic = tm.time()
    def process_frame(int frame_idx):
        cdef int label_val = lineage_dict[frame_idx]
        cdef dict result = {
            'frame_idx': frame_idx,
            'time': times[frame_idx] if 0 <= frame_idx < len(times) else np.nan
        }

        if 0 <= frame_idx < len(ACs):
            features = process_label(
                ACs[frame_idx], DCs[frame_idx], labels[frame_idx], label_val)
            result['features'] = features
        else:
            result['features'] = [np.array([np.nan, np.nan])] + [np.nan] * 17 + [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])]

        return result
        
    with concurrent.futures.ThreadPoolExecutor(max_workers = int(np.sqrt(os.cpu_count()))) as executor:
        future_results = list(executor.map(process_frame, frame_indexes))
        results.extend(future_results)

    results.sort(key=lambda x: x['frame_idx'])

    for result in results:
        cell_data['times'].append(result['time'])
        cell_data['idx'].append(result['frame_idx'])
        features = result['features']
        cell_data['position'].append(features[0])
        cell_data['Interior contour'].append(features[-2])
        cell_data['Total contour'].append(features[-1])

        for i, feature in enumerate(feature_names[3:-2], 1): 
            cell_data[feature].append(features[i])

    for feature in feature_names[1:-2]:
        cell_data[feature] = np.array(cell_data[feature])
    print('Processed '+cellname+' in'+str(tm.time()-tic)+'s')
    return (cellname, cell_data)


cdef list process_label(np.ndarray AC, np.ndarray DC, np.ndarray label, float col):
    cdef np.ndarray[bool, ndim=2] mask = (label == float(col))
    cdef np.ndarray[double, ndim=2] filtered = np.zeros_like(label, dtype=np.float64)
    cdef np.ndarray[bool, ndim=2] background_mask = (label == 0)
    cdef np.ndarray[double, ndim=1] com = np.array([0.0, 0.0])
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
    cdef tuple contour_list
    cdef np.ndarray hierarchy
    cdef np.ndarray[long,ndim=2] coords
    cdef object ellipse, DC_values, AC_values, intcontours, interior_mask
    cdef float denom1, denom2

    filtered[mask] = 1
    coords = np.argwhere(mask)

    if len(coords) > 0:
        com = np.mean(coords, axis=0)
        area = len(coords)

        if area > 1:
            contour_list, hierarchy = cv2.findContours(
                filtered.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if contour_list:
                contour = contour_list[0].reshape(contour_list[0].shape[0], contour_list[0].shape[2])
                if len(contour) > 4:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                    major = max(ellipse[1])
                    minor = min(ellipse[1])
                start = contour[0, :].reshape(1, 2)
                contour = np.append(contour, start, axis=0)
        else:
            angle = 0.0
            major = 1.0
            minor = 1.0

        DC_values = DC[mask]
        if len(DC_values) > 0:
            DCmean = np.mean(DC_values)
            DCmin = np.min(DC_values)
            DCmax = np.max(DC_values)

        AC_values = AC[mask]
        if len(AC_values) > 0:
            ACmean = np.mean(AC_values)
            ACmin = np.min(AC_values)
            ACmax = np.max(AC_values)

            intcontours, interior_mask = findInterior(AC, mask)
            interiorarea = np.count_nonzero(interior_mask)
            if interiorarea > 0:
                interior_values = AC[interior_mask]
                interiormean = np.mean(interior_values) if len(interior_values) > 0 else 0

                background_values = AC[background_mask]
                background = np.mean(background_values) if len(background_values) > 0 else 0

                contour_mask = mask & ~interior_mask
                contour_values = AC[contour_mask]
                contourmean = np.mean(contour_values) if len(contour_values) > 0 else 0

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


cdef tuple findInterior(np.ndarray[double, ndim=2] image, np.ndarray[bool, ndim=2] cell_mask, 
                      float min_area_ratio=0.05, float max_area_ratio=1.0):
    # Get typed views for faster access
    cdef double[:, :] image_view = image
    cdef bool[:, :] cell_mask_view = cell_mask
    
    # Compute areas and thresholds once
    cdef int cell_area = np.count_nonzero(cell_mask)
    cdef int min_area = <int>(cell_area * min_area_ratio)
    cdef int max_area = <int>(cell_area * max_area_ratio)
    
    # Pre-allocate arrays (just once)
    cdef np.ndarray[int, ndim=2] interior_mask = np.zeros_like(cell_mask, dtype=np.int32)
    cdef np.ndarray[double, ndim=2] masked_image = np.zeros_like(image)
    cdef double[:, :] masked_image_view = masked_image
    
    # Create masked image with vectorized operations
    masked_image = np.where(cell_mask > 0, image, 0)
    
    # Calculate mean intensity more efficiently
    cdef double sum_intensity = 0.0
    cdef int count = 0
    cdef int i, j
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    
    for i in range(height):
        for j in range(width):
            if cell_mask_view[i, j] > 0:
                sum_intensity += image_view[i, j]
                count += 1
    
    cdef double mean_intensity = sum_intensity / count if count > 0 else 0.0
    
    # Binary search for threshold instead of linear search
    cdef double high_thresh = mean_intensity
    cdef double low_thresh = 0.0
    cdef double mid_thresh
    cdef tuple contours
    cdef np.ndarray thresholded
    cdef int max_iterations = 10  # Usually converges faster than linear search
    cdef int iteration = 0
    
    # First try with mean intensity
    _, thresholded = cv2.threshold(masked_image, mean_intensity, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, use binary search to find appropriate threshold
    while len(contours) == 0 and iteration < max_iterations:
        iteration += 1
        mid_thresh = (high_thresh + low_thresh) / 2.0
        _, thresholded = cv2.threshold(masked_image, mid_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            high_thresh = mid_thresh
        else:
            break
    
    # Process contours
    cdef list interior_contours = []
    cdef double area
    cdef np.ndarray cont, start
    cdef list results = []
    
    
    for cont in contours:
        area = cv2.contourArea(cont)
        if min_area <= area <= max_area:
            interior_contours.append(cont)
            cv2.drawContours(interior_mask, [cont], -1, 255, -1)
    
    # Pre-allocate results list if you know the approximate size
    results = [None] * len(interior_contours)
    
    cdef int idx = 0
    for cont in interior_contours:
        # Use more efficient reshape operation
        cont_reshaped = cont.reshape(cont.shape[0], 2)
        start = cont_reshaped[0].reshape(1, 2)
        results[idx] = np.vstack((cont_reshaped, start))
        idx += 1
    
    # In case we overallocated, trim the list
    if idx < len(results):
        results = results[:idx]
    
    return results, interior_mask