import numpy as np
from numpyradiomics.shape_3d_mod import shape_3d, shape_3d_units
from numpyradiomics.shape_2d_mod import shape_2d, shape_2d_units
from numpyradiomics.ski_shape_3d_mod import ski_shape_3d, ski_shape_3d_units

def shape(input_mask:np.ndarray, spacing=None):
    """
    Compute shape features for a binary mask (similar to Pyradiomics shape_2D).

    Parameters
    ----------
    input_mask : np.ndarray
        2D or 3D binary mask (non-zero = ROI)
    spacing : tuple of float
        Pixel or voxel spacing in units of mm (row_spacing, col_spacing, slice_spacing)

    Returns
    -------
    """
    if np.ndim(input_mask) == 2:
        return shape_2d(input_mask, spacing)
    if np.ndim(input_mask) == 3:
        rad_shape = shape_3d(input_mask, spacing)
        ski_shape = ski_shape_3d(input_mask, spacing)
        return rad_shape | ski_shape
    

def shape_units(dim:int):
    """Units of returned shape metrics

    Args:
        dim (int): nr of dimensions (2 or 3)

    Returns:
        dict: variables and their units
    """
    if dim == 2:
        return shape_2d_units
    if dim == 3:
        return shape_3d_units | ski_shape_3d_units