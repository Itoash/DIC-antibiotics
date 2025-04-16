#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:46:32 2025

@author: victorionescu
"""
import numpy as np
import concurrent.futures
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import skimage as ski
from numba import jit
import os
import time as tm
from cellprocessor import process_cells

def process_cellsx(celldict, ACs, DCs, labels, times):
    """
    Process multiple cells across frames and organize results by cell.

    Parameters
    ----------
    dict(dict) celldict : dict of dicts with format:
        {<cellname>:{<frameindex>:<label>}}
        that gives position and name to cells
    list(np.ndarray[double,ndim=2]) ACs : list of AC images to analyze
    list(np.ndarray[double,ndim=2]) DCs : list of DC images to analyze
    list(np.ndarray[double,ndim=2]) labels : list of label images to provide masks
    list(int) times: associated times to be added to cell props (start from 0)

    Returns
    -------
    dict(dict) cells: dict of dicts with format:
        {<cellname>:{<propertyname>:<values>}}
    """
    tic = tm.time()
    # Initialize result dictionary
    cells = {}

    # Process each cell using multiprocessing for efficiency
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a list of tasks to process in parallel
        futures = []
        for cellname, frame_labels in celldict.items():
            futures.append(executor.submit(
                process_cell_lineage, cellname, frame_labels, ACs, DCs, labels,times))

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            cellname, cell_data = future.result()
            cells[cellname] = cell_data
    # futures = []
    # for cellname,frame_labels in celldict.items():
    #     for AC,DC,lab,tim in zip(ACs,DCs,labels,times):
    #         res = process_cell_lineage(cellname, frame_labels, AC, DC, lab, times)
    #         futures.append(res)
    # for el in futures:       
    #     cellname,cell_data =el
    #     cells[cellname] = cell_data
    print(tm.time()-tic)
    return cells


def process_cell_lineage(cellname, lineage_dict, ACs, DCs, labels, times):
    """
    Process a single cell's lineage across multiple frames using dictionary approach
    with ThreadPoolExecutor to parallelize frame processing

    Parameters
    ----------
    str cellname : Name of the cell
    dict lineage_dict : Dictionary mapping frame indexes to label values
    list ACs : List of AC images
    list DCs : List of DC images
    list labels : List of label images
    list times : List of timepoints

    Returns
    -------
    tuple: (cellname, cell_data_dict)
    """
    # Define feature names
    feature_names = ['idx', 'times', 'position', 'area', 'Ellipse angle', 'Ellipse major',
                     'Ellipse minor', 'DC mean', 'DC min', 'DC max', 'AC mean', 'AC min', 'AC max',
                     'AC interior area', 'AC interior mean', 'AC contour mean',
                     'AC interior/back contrast', 'AC interior/contour contrast', 'AC solidity','Interior contour','Total contour']

    # Get the frame indexes and label values
    frame_indexes = list(lineage_dict.keys())

    # Initialize the cell data dictionary with empty lists
    cell_data = {feature: [] for feature in feature_names}
    # Function to process a single frame

    def process_frame(frame_idx):
        label_val = lineage_dict[frame_idx]

        # Initialize results
        result = {
            'frame_idx': frame_idx,
            'time': times[frame_idx] if 0 <= frame_idx < len(times) else np.nan
        }

        # Process the frame if it's within bounds
        if 0 <= frame_idx < len(ACs):
            features = process_label(
                ACs[frame_idx], DCs[frame_idx], labels[frame_idx], label_val)
            result['features'] = features
        else:
            # Default values for out of bounds
            result['features'] = [
                np.array([np.nan, np.nan]),  # position
                np.nan,  # area
                np.nan,  # angle
                np.nan,  # major
                np.nan,  # minor
                np.nan,  # DCmean
                np.nan,  # DCmin
                np.nan,  # DCmax
                np.nan,  # ACmean
                np.nan,  # ACmin
                np.nan,  # ACmax
                np.nan,  # interiorarea
                np.nan,  # interiormean
                np.nan,  # contourmean
                np.nan,  # intbackcontrast
                np.nan,  # intcontcontrast
                np.nan,  # solidity
                np.nan,  # contour of interior
                np.nan   # total contour
                
            ]

        return result

    # Process frames in parallel using ThreadPoolExecutor
    # We use ThreadPoolExecutor instead of ProcessPoolExecutor here because:
    # 1. The operations are I/O bound rather than CPU bound
    # 2. We avoid the overhead of serializing/deserializing large image data
    # 3. We're already using ProcessPoolExecutor at the cell level
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map the processing function to all frame indexes
        future_results = list(executor.map(process_frame, frame_indexes))
        results.extend(future_results)

    # Sort results by frame index to maintain temporal order
    results.sort(key=lambda x: x['frame_idx'])

    # Populate the cell_data dictionary
    for result in results:
        cell_data['times'].append(result['time'])
        cell_data['idx'].append(result['frame_idx'])
        features = result['features']

        # Add position (features[0])
        cell_data['position'].append(features[0])
        cell_data['Interior contour'].append(features[-2])
        cell_data['Total contour'].append(features[-1])
        # Add all other features
        for i, feature in enumerate(feature_names[3:-2], 1):
            cell_data[feature].append(features[i])

    # Convert lists to numpy arrays for easier manipulation
    for feature in feature_names[1:-2]:
        cell_data[feature] = np.array(cell_data[feature])

    return cellname, cell_data


def process_label(AC, DC, label, col):
    """
    Process one label at one timepoint fully

    Parameters
    ----------
    np.ndarray[double,ndim=2] AC : AC image at timepoint
    np.ndarray[double,ndim=2] DC : DC image at timepoint
    np.ndarray[double,ndim=2] label : labelimage at timepoint
    double col: label color to select from image

    Returns
    -------
    list: Features for the specified label
    """
    # Create mask for the specified label
    mask = (label == float(col))
    filtered = np.zeros_like(label, dtype=float)
    filtered[mask] = 1
    background_mask = (label == 0)
    # Default values
    com = np.array([0.0, 0.0])
    area = 0.0
    angle = 0.0
    major = 0.0
    minor = 0.0
    DCmean = 0.0
    DCmin = 0.0
    DCmax = 0.0
    ACmean = 0.0
    ACmin = 0.0
    ACmax = 0.0
    interiorarea = 0.0
    interiormean = 0.0
    contourmean = 0.0
    intbackcontrast = 0.0
    intcontcontrast = 0.0
    solidity = 0.0
    intcontour = np.array([0.0, 0.0])
    contour = np.array([0.0, 0.0])
    
    # Get coordinates of the mask
    coords = np.argwhere(mask)

    # Only process if mask contains pixels
    if len(coords) > 0:
        # Centroid coordinates
        com = np.mean(coords, axis=0)
        # Area of mask
        area = len(coords)

        if area > 1:  # Mask has more than one pixel
            # Find contours
            contour, hierarchy = cv2.findContours(
                filtered.astype('uint8'),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE
            )

            if contour:  # Check if contours were found
                contour = contour[0]
                cdims = np.shape(contour)
                contour = np.reshape(contour, (cdims[0], cdims[2]))
                if len(contour) > 4:  # Enough points for ellipse fitting
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                    # Ensure major axis is the larger one
                    major = max(ellipse[1])
                    # Ensure minor axis is the smaller one
                    minor = min(ellipse[1])
                start = contour[0,:].reshape(1,2)
                contour = np.append(contour,start,axis=0)
        else:  # Single pixel mask
            angle = 0.0
            major = 1.0
            minor = 1.0

        # DC statistics
        DC_values = DC[mask]
        if len(DC_values) > 0:
            DCmean = np.mean(DC_values)
            DCmin = np.min(DC_values)
            DCmax = np.max(DC_values)

        # AC statistics
        AC_values = AC[mask]
        if len(AC_values) > 0:
            ACmean = np.mean(AC_values)
            ACmin = np.min(AC_values)
            ACmax = np.max(AC_values)

            # Interior region processing
            intcontours,interior_mask = findInterior(AC, mask)
            
            # Calculate interior and contour metrics
            interiorarea = np.count_nonzero(interior_mask)
            if interiorarea > 0:
                interior_values = AC[interior_mask]
                interiormean = np.mean(interior_values) if len(
                    interior_values) > 0 else 0

                # Background is everything outside the masks
                background_values = AC[background_mask]
                background = np.mean(background_values) if len(
                    background_values) > 0 else 0

                # Contour is mask minus interior
                contour_mask = mask & ~interior_mask
                contour_values = AC[contour_mask]
                contourmean = np.mean(contour_values) if len(
                    contour_values) > 0 else 0

                # Calculate contrast metrics (avoid division by zero)
                denom1 = interiormean + background
                if denom1 != 0:
                    intbackcontrast = (interiormean - background) / denom1

                denom2 = interiormean + contourmean
                if denom2 != 0:
                    intcontcontrast = (interiormean - contourmean) / denom2

                # Calculate solidity
                solidity = interiorarea / area if area > 0 else 0

    # Return all features as a list
    
    return [com, area, angle, major, minor, DCmean, DCmin, DCmax, ACmean, ACmin, ACmax,
            interiorarea, interiormean, contourmean, intbackcontrast, intcontcontrast, solidity, intcontours,contour]

def findInterior(image, cell_mask, min_area_ratio=0.05, max_area_ratio=1):
    """
    Find interior contours in an image based on a provided cell mask.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The input image (e.g., AC image)
    cell_mask : numpy.ndarray
        Binary mask specifying the region of interest (cell boundary)
    min_area_ratio : float, optional
        Minimum contour area as a fraction of the cell mask area
    max_area_ratio : float, optional
        Maximum contour area as a fraction of the cell mask area
    
    Returns:
    --------
    interior_contours : list
        List of interior contours that meet the size criteria
    interior_mask : numpy.ndarray
        Binary mask highlighting the interior regions
    """
    # Ensure the mask is binary
    if cell_mask.dtype != np.uint8:
        cell_mask = cell_mask.astype(np.uint8)
    
    # Calculate mask statistics
    cell_area = np.count_nonzero(cell_mask)
    mean_intensity = np.mean(image[cell_mask > 0])
    
    # Calculate area thresholds
    min_area = int(cell_area * min_area_ratio)
    max_area = int(cell_area * max_area_ratio)
    
    # Create interior mask initialized as empty
    interior_mask = np.zeros_like(cell_mask)
    
    # Find interior regions using thresholding
    # Only search within the cell mask
    masked_image = np.zeros_like(image)
    masked_image[cell_mask > 0] = image[cell_mask > 0]
    
    # Apply initial threshold at mean intensity
    _, thresholded = cv2.threshold(masked_image, mean_intensity, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresholded.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found with initial threshold, try lowering the threshold
    if len(contours) == 0:
        safety = 0
        while len(contours) == 0 and safety < 100:
            safety += 1
            threshold = mean_intensity * (1 - safety/100)
            _, thresholded = cv2.threshold(masked_image, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
    
    # Filter contours by size
    interior_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            interior_contours.append(contour)
            # Add this contour to the interior mask
            cv2.drawContours(interior_mask, [contour], -1, 255, -1)
    results = []
    for cont in interior_contours:
        cdims = cont.shape
        cont = np.reshape(cont,(cdims[0],cdims[2]))
        start = cont[0,:].reshape((1,2))
        cont = np.append(cont,start,axis = 0)
        results.append(cont)
    return results, interior_mask
    

def plot_cell_features(cells, feature_name):
    """
    Plot a specific feature for all cells over time

    Parameters
    ----------
    dict cells : Cell data dictionary
    str feature_name : Name of the feature to plot
    """
    plt.figure(figsize=(10, 6))

    for cellname, cell_data in cells.items():
        # Skip if the feature doesn't exist or has no values
        if feature_name not in cell_data or len(cell_data[feature_name]) == 0:
            continue

        # For position, we need to handle it differently (it's a 2D array)
        if feature_name == 'position':
            positions = cell_data[feature_name]
            plt.plot(positions[:, 1], positions[:, 0], 'o-', label=cellname)
            plt.xlabel('X position')
            plt.ylabel('Y position')
        else:
            # For all other features, plot against time
            plt.plot(cell_data['times'],
                     cell_data[feature_name], 'o-', label=cellname)
            plt.xlabel('Time')
            plt.ylabel(feature_name)

    plt.title(f'{feature_name} over time')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_cells_at_frame(ACs, labels, cells, frame_idx):
    """
    Visualize cells at a specific frame

    Parameters
    ----------
    list ACs : List of AC images
    list labels : List of label images
    dict cells : Cell data dictionary
    int frame_idx : Frame index to visualize
    """
    # Get the AC image and label at the specified frame
    AC = ACs[frame_idx]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the AC image
    ax.imshow(AC, cmap='gray')

    # For each cell, add annotations
    for cellname, cell_data in cells.items():
        # Find the index in the cell's data corresponding to this frame
        times = cell_data['times']
        time_idx = np.where(times == frame_idx)[0]

        if len(time_idx) == 0:
            # This cell doesn't exist at this frame
            continue

        time_idx = time_idx[0]

        # Get the cell's position, angle, and axes
        position = cell_data['position'][time_idx]
        angle = cell_data['Ellipse angle'][time_idx]
        major = cell_data['Ellipse major'][time_idx]
        minor = cell_data['Ellipse minor'][time_idx]

        # Skip if any value is NaN
        if (np.isnan(position).any() or np.isnan(angle) or
                np.isnan(major) or np.isnan(minor)):
            continue

        # Add an ellipse
        ellipse = Ellipse((position[1], position[0]),
                          width=major, height=minor,
                          angle=angle,
                          edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(ellipse)

        # Add the cell name
        ax.text(position[1], position[0], cellname,
                color='white', fontsize=12, ha='center', va='center')
        
        
        contour_interior = cell_data['Interior contour'][time_idx] 
        for cont in contour_interior:
            ax.plot(cont[:,0],cont[:,1],c = 'g')
    plt.title(f'Cells at Frame {frame_idx}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Create synthetic data for demonstration
    # Image dimensions
    width, height = 400, 400
    num_frames = 10

    # Generate random AC and DC images
    np.random.seed(42)  # For reproducibility
    ACs = [np.random.rand(width,height) * 255 for _ in range(num_frames)]
    DCs = [np.random.rand(width,height) * 255 for _ in range(num_frames)]
    
    # Create label images with two cells
    labels = [np.zeros((height, width)) for _ in range(num_frames)]

    # Cell 1: A circular cell that moves diagonally
    for i in range(num_frames):
        # Center position moves from (20,20) to (40,40)
        center_y = int(100 +i* 5)
        center_x = int(100 +i* 5)
        radius = 15

        # Create a circular mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (radius)**2
        mask2 = (x - center_x)**2 + 2*(y - center_y)**2 <= (radius/2)**2
        ACs[i][mask2] = 300
        labels[i][mask] = 2
        

    # Cell 2: An elliptical cell that changes shape
    for i in range(num_frames):
        # Center is fixed at (70,70)
        center_y, center_x = 200, 200
        # Major and minor axes vary with time
        major = 30 + 5 * np.sin(i * np.pi / 5)
        minor = 15 + 3 * np.cos(i * np.pi / 5)

        # Create an elliptical mask
        y, x = np.ogrid[:height, :width]
        # Rotate the ellipse
        angle = i * 10  # degrees
        angle_rad = angle * np.pi / 180

        # Ellipse equation with rotation
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        x_centered = x - center_x
        y_centered = y - center_y
        x_rot = x_centered * cos_angle + y_centered * sin_angle
        y_rot = -x_centered * sin_angle + y_centered * cos_angle
        mask = (x_rot / major)**2 + (y_rot / minor)**2 <= 1
        mask2 = (x_rot/major)**2 + (y_rot/minor)**2 <= 0.5
        ACs[i][mask2] = 300
        labels[i][mask] = 1

    # Create cell dictionary
    celldict = {
        # Cell 1 appears in all frames with label 1
        "cell1": {i: 2 for i in range(num_frames)},
        # Cell 2 appears in all frames with label 2
        "cell2": {i: 1 for i in range(num_frames)}
    }

    # Define times
    times = list(range(num_frames))

    print("Processing cells...")
    start_time = time.time()

    # Process the cells
    results = process_cells(celldict, ACs, DCs, labels, times)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

    # Print some results
    print("\nSummary of results:")
    for cellname, cell_data in results.items():
        print(f"\n{cellname}:")
        print(f"  Times: {cell_data['times']}")
        print(f"  Average area: {np.mean(cell_data['area']):.2f}")
        print(f"  Average AC mean: {np.mean(cell_data['AC mean']):.2f}")
        print(f"  Average DC mean: {np.mean(cell_data['DC mean']):.2f}")
        print(f"  Average AC int mean: {np.mean(cell_data['AC interior mean']):.2f}")

    # Examples of visualization
    print("\nGenerating visualizations...")

    # Plot area over time
    plot_cell_features(results, 'AC interior area')

    # Plot cell positions
    plot_cell_features(results, 'area')

    # Visualize cells at frame 5
    for f in range(num_frames):
        visualize_cells_at_frame(ACs, labels, results, f)
                    
    print("Done!")
