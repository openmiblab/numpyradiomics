import numpy as np
from scipy.ndimage import label

def gldm_features(input_image, input_mask, binWidth=25, levels=None, distance=1, connectivity=None):
    """
    Compute 14 Pyradiomics-style GLDM features for 2D or 3D images.

    Parameters
    ----------
    input_image : np.ndarray
        2D or 3D image array.
    input_mask : np.ndarray
        Same shape as input_image. Non-zero = ROI.
    binWidth : float
        Width of intensity bins for quantization.
    levels : int, optional
        Number of gray levels. Computed automatically if None.
    distance : int
        Maximum distance for voxel dependencies (not used explicitly here, included for compatibility).
    connectivity : int, optional
        Neighbor connectivity:
            - 2D: 4 or 8 (default 8)
            - 3D: 6, 18, or 26 (default 26)

    Returns
    -------
    dict
        14 Pyradiomics GLDM features.
    """
    roi_mask = input_mask > 0
    if not np.any(roi_mask):
        raise ValueError("Mask contains no voxels.")

    roi_image = input_image.copy()
    roi_image[~roi_mask] = 0

    # Quantize intensities
    min_val = np.min(roi_image[roi_mask])
    max_val = np.max(roi_image[roi_mask])
    if max_val == min_val:
        raise ValueError("ROI has constant intensity; GLDM cannot be computed.")

    if levels is None:
        levels = int(np.ceil((max_val - min_val)/binWidth)) + 1

    img_quant = np.floor((roi_image - min_val)/binWidth).astype(int)
    img_quant = np.clip(img_quant, 0, levels-1)

    # Determine dimensionality and default connectivity
    dims = input_image.ndim
    if connectivity is None:
        connectivity = 8 if dims == 2 else 26

    structure = _get_connectivity_structure_nd(dims, connectivity)

    # Maximum possible dependence size
    max_dependence_size = np.sum(roi_mask)
    gldm = np.zeros((levels, max_dependence_size), dtype=np.int64)

    # Compute dependencies for each gray level
    for g in range(levels):
        mask_g = (img_quant == g) & roi_mask
        labeled, num_features = label(mask_g, structure=structure)
        if num_features == 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]  # skip background
        for size in sizes:
            gldm[g, size-1] += 1

    # Trim empty columns
    gldm = gldm[:, np.any(gldm, axis=0)]

    # Compute GLDM features
    features = _compute_gldm_features(gldm)
    return features


# ----------------- Helper Functions ----------------- #

def _compute_gldm_features(P):
    """Compute 14 Pyradiomics GLDM features from matrix P."""
    P = P.astype(np.float64)
    Ns = P.shape[1]  # dependence sizes
    Ng = P.shape[0]  # gray levels
    Nd = P.sum()
    P_norm = P / (Nd + 1e-12)

    i = np.arange(1, Ng+1).reshape(-1,1)
    j = np.arange(1, Ns+1).reshape(1,-1)

    Ps = P_norm.sum(axis=0)  # dependence size probabilities
    Pg = P_norm.sum(axis=1)  # gray level probabilities

    mean_gray = np.sum(i * Pg)
    mean_dep = np.sum(j * Ps)

    features = dict()
    features['SmallDependenceEmphasis'] = np.sum(Ps / (j**2))
    features['LargeDependenceEmphasis'] = np.sum(Ps * (j**2))
    features['GrayLevelNonUniformity'] = np.sum(Pg**2)
    features['GrayLevelNonUniformityNormalized'] = np.sum(Pg**2) / (Nd**2 + 1e-12)
    features['DependenceNonUniformity'] = np.sum(Ps**2)
    features['DependenceNonUniformityNormalized'] = np.sum(Ps**2) / (Nd**2 + 1e-12)
    features['DependencePercentage'] = Nd / P.size
    features['LowGrayLevelEmphasis'] = np.sum(P_norm / (i**2))
    features['HighGrayLevelEmphasis'] = np.sum(P_norm * (i**2))
    features['SmallDependenceLowGrayLevelEmphasis'] = np.sum(P_norm / (i**2 * j**2))
    features['SmallDependenceHighGrayLevelEmphasis'] = np.sum(P_norm * (i**2 / j**2))
    features['LargeDependenceLowGrayLevelEmphasis'] = np.sum(P_norm * (j**2 / i**2))
    features['LargeDependenceHighGrayLevelEmphasis'] = np.sum(P_norm * (i**2 * j**2))
    features['GrayLevelVariance'] = np.sum(Pg * (i - mean_gray)**2)

    return features


def _get_connectivity_structure_nd(dims, connectivity):
    """Return 2D or 3D connectivity structure for labeling."""
    if dims == 2:
        if connectivity == 4:
            structure = np.array([[0,1,0],
                                  [1,1,1],
                                  [0,1,0]], dtype=int)
        else:  # default 8
            structure = np.ones((3,3), dtype=int)
    elif dims == 3:
        if connectivity == 6:
            structure = np.zeros((3,3,3), dtype=int)
            structure[1,1,0] = structure[1,1,2] = 1
            structure[1,0,1] = structure[1,2,1] = 1
            structure[0,1,1] = structure[2,1,1] = 1
        elif connectivity == 18:
            structure = np.ones((3,3,3), dtype=int)
            structure[0,0,0] = structure[0,0,2] = structure[0,2,0] = structure[0,2,2] = 0
            structure[2,0,0] = structure[2,0,2] = structure[2,2,0] = structure[2,2,2] = 0
        else:  # 26-connectivity
            structure = np.ones((3,3,3), dtype=int)
    else:
        raise ValueError("Input must be 2D or 3D.")
    return structure
