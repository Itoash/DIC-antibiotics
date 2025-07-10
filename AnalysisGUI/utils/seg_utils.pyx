# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: language=c++
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
import numpy as np
cimport numpy as np
from cellpose_omni import models
import time
import omnipose

def segmentDComni(list imgs, dict params=None):
    """
    Segment images using Cellpose Omni model
    
    Parameters:
    -----------
    imgs : list
        List of 2D numpy arrays representing images
    params : dict, optional
        Segmentation parameters. If None, uses default parameters.
    Returns:
    --------
    list
        List of masks for each image
    """
    cdef int nimg = len(imgs)
    cdef str model_name = 'bact_phase_omni'
    cdef object use_GPU = omnipose.gpu.use_gpu()
    cdef object model = models.CellposeModel(gpu=use_GPU,model_type=model_name)
    cdef list chans = [0, 0]  # segment based on first channel, no second channel
    cdef np.ndarray n = np.arange(nimg, dtype=np.int32)  # segment all images in list

    # define default parameters
    cdef dict default_params = {
        'channels': chans,
        'rescale': None,
        'mask_threshold': -2,
        'flow_threshold': 0.99,
        'transparency': False,
        'omni': True,
        'cluster': False,
        'resample': True,
        'verbose': False,
        'tile': True,
        'niter': None,
        'augment': True,
        'affinity_seg': True,
        'invert': True
    }
    if params is None:
        params = default_params

    cdef double tic = time.time()
    cdef list image_list = [imgs[i] for i in n]
    cdef list masks, flows, styles
    masks, flows, styles = model.eval(image_list, **params)
    cdef double net_time = time.time() - tic
    print('total segmentation time: {}s'.format(net_time))
    return masks

