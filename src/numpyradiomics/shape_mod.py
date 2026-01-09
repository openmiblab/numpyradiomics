import numpy as np
from numpyradiomics.shape_3d_mod import shape_3d
from numpyradiomics.shape_2d_mod import shape_2d

def shape(input_mask, spacing=None):
    """
    Compute shape features for a binary mask (similar to Pyradiomics shape_2D).

    Parameters
    ----------
    input_mask : np.ndarray
        2D or 3D binary mask (non-zero = ROI)
    spacing : tuple of float
        Pixel or voxel spacing (row_spacing, col_spacing, slice_spacing)

    Returns
    -------
    """
    if np.ndim(input_mask) == 2:
        return shape_2d(input_mask, spacing)
    if np.ndim(input_mask) == 3:
        return shape_3d(input_mask, spacing)