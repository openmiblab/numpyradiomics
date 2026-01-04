import numpy as np
from scipy.ndimage import label

def glszm_features(input_image, input_mask, binWidth=25, levels=None, connectivity=None):
    """
    Compute 16 Pyradiomics-style GLSZM features for 2D or 3D images.

    Parameters
    ----------
    input_image : np.ndarray
        2D or 3D image array.
    input_mask : np.ndarray
        2D or 3D mask array (non-zero = ROI).
    binWidth : float
        Width of intensity bins for quantization.
    levels : int, optional
        Number of gray levels after quantization.
    connectivity : int, optional
        Connectivity for zone labeling:
            - 2D: 4 or 8 (default 8)
            - 3D: 6, 18, or 26 (default 26)
    Returns
    -------
    dict
        Dictionary of 16 GLSZM features.
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
        raise ValueError("ROI has constant intensity; GLSZM cannot be computed.")

    if levels is None:
        levels = int(np.ceil((max_val - min_val) / binWidth)) + 1

    img_quant = np.floor((roi_image - min_val) / binWidth).astype(int)
    img_quant = np.clip(img_quant, 0, levels - 1)

    # Determine image dimensionality and default connectivity
    dims = input_image.ndim
    if connectivity is None:
        connectivity = 8 if dims == 2 else 26

    structure = _get_connectivity_structure_nd(dims, connectivity)

    # Initialize GLSZM matrix
    max_zone_size = np.sum(roi_mask)
    glszm = np.zeros((levels, max_zone_size), dtype=np.int64)

    # Compute zones for each gray level
    for g in range(levels):
        mask_g = (img_quant == g) & roi_mask
        labeled, num_features = label(mask_g, structure=structure)
        if num_features == 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]  # skip background
        for size in sizes:
            glszm[g, size - 1] += 1

    # Trim empty columns
    glszm = glszm[:, np.any(glszm, axis=0)]

    # Compute GLSZM features
    features = _compute_glszm_features_16(glszm)
    return features


def _get_connectivity_structure_nd(dims, connectivity):
    """
    Returns a connectivity structure for 2D or 3D images.
    2D: connectivity = 4 or 8
    3D: connectivity = 6, 18, 26
    """
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



def _compute_glszm_features_16(P):
    """
    Compute the 16 Pyradiomics GLSZM features from GLSZM matrix P.
    """
    P = P.astype(np.float64)
    Ns = P.shape[1]  # zone sizes
    Ng = P.shape[0]  # gray levels
    Nz = P.sum()
    P_norm = P / (Nz + 1e-12)

    i = np.arange(1, Ng + 1).reshape(-1, 1)  # gray levels
    j = np.arange(1, Ns + 1).reshape(1, -1)  # zone sizes

    Ps = P_norm.sum(axis=0)  # zone size probabilities
    Pg = P_norm.sum(axis=1)  # gray level probabilities

    mean_gray = np.sum(i * Pg)
    mean_size = np.sum(j * Ps)

    features = dict()

    # 16 Pyradiomics features
    features['SmallAreaEmphasis'] = np.sum(Ps / (j**2))
    features['LargeAreaEmphasis'] = np.sum(Ps * (j**2))
    features['GrayLevelNonUniformity'] = np.sum(Pg**2)
    features['GrayLevelNonUniformityNormalized'] = np.sum(Pg**2) / (Nz**2 + 1e-12)
    features['ZoneSizeNonUniformity'] = np.sum(Ps**2)
    features['ZoneSizeNonUniformityNormalized'] = np.sum(Ps**2) / (Nz**2 + 1e-12)
    features['ZonePercentage'] = Nz / P.size
    features['LowGrayLevelZoneEmphasis'] = np.sum(P_norm / (i**2))
    features['HighGrayLevelZoneEmphasis'] = np.sum(P_norm * (i**2))
    features['SmallAreaLowGrayLevelEmphasis'] = np.sum(P_norm / (i**2 * j**2))
    features['SmallAreaHighGrayLevelEmphasis'] = np.sum(P_norm * (i**2 / j**2))
    features['LargeAreaLowGrayLevelEmphasis'] = np.sum(P_norm * (j**2 / i**2))
    features['LargeAreaHighGrayLevelEmphasis'] = np.sum(P_norm * (i**2 * j**2))
    features['GrayLevelVariance'] = np.sum(Pg * (i - mean_gray)**2)
    features['ZoneSizeVariance'] = np.sum(Ps * (j - mean_size)**2)
    # For Pyradiomics consistency, keep ZoneSizeNonUniformity as last
    features['ZoneSizeNonUniformity'] = np.sum(Ps**2)

    return features
