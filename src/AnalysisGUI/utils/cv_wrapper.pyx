# distutils: language = c++
# cython: language_level=3
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint8_t, int32_t

# Define OpenCV's Point2i structure and Point typedef correctly
cdef extern from "opencv2/core/types.hpp" namespace "cv":
    cdef struct Point2i:
        int x
        int y
    
    ctypedef Point2i Point

# Include your C++ header with the correct function signature
cdef extern from "cv_helpers.h":
    void find_interior(
        const double* img_data,
        const unsigned char* mask_data,
        int rows, int cols,
        double min_area_ratio,
        double max_area_ratio,
        vector[vector[Point]]& contours_out,
        vector[uint8_t]& interior_mask_out
    )

cpdef tuple call_find_interior(np.ndarray[np.float64_t, ndim=2] img,
                              np.ndarray[np.uint8_t, ndim=2] mask,
                              double min_area_ratio,
                              double max_area_ratio):
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    
    # Declare all variables at the beginning of the function
    cdef vector[vector[Point]] contours
    cdef vector[uint8_t] interior_mask
    cdef np.ndarray[np.uint8_t, ndim=2] interior_mask_np
    cdef np.ndarray[np.uint8_t, ndim=1] temp
    cdef int mask_size, num_contours, contour_size
    cdef int i, j
    cdef np.ndarray[np.int32_t, ndim=2] py_contour
    
    # Call the C++ function
    find_interior(
        <const double*> img.data,
        <const unsigned char*> mask.data,
        rows,
        cols,
        min_area_ratio,
        max_area_ratio,
        contours,
        interior_mask
    )
    
    # Convert interior_mask to numpy array
    interior_mask_np = np.zeros((rows, cols), dtype=np.uint8)
    mask_size = interior_mask.size()
    
    # Only copy if the sizes match
    if mask_size == rows * cols:
        # Create a temporary buffer and copy data
        temp = np.zeros(mask_size, dtype=np.uint8)
        for i in range(mask_size):
            temp[i] = interior_mask[i]
        # Reshape to 2D array
        interior_mask_np = temp.reshape((rows, cols)).copy()
    
    # Convert contours to list of numpy arrays
    py_contours = []
    num_contours = contours.size()
    
    for i in range(num_contours):
        contour_size = contours[i].size()
        py_contour = np.zeros((contour_size, 2), dtype=np.int32)
        
        for j in range(contour_size):
            py_contour[j, 0] = contours[i][j].x
            py_contour[j, 1] = contours[i][j].y
        
        py_contours.append(py_contour)
    
    return py_contours, interior_mask_np