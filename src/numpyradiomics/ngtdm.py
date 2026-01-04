import numpy as np
from scipy.ndimage import convolve


def ngtdm_features(input_image, input_mask, binWidth=25, levels=None, distance=1):
    """
    Compute Pyradiomics-style NGTDM (Neighborhood Gray Tone Difference Matrix) features for a 3D image.

    NGTDM captures how voxel intensities differ from their local neighborhood. Features include
    coarseness, contrast, busyness, complexity, and strength.

    Parameters
    ----------
    input_image : np.ndarray
        3D image array.
    input_mask : np.ndarray
        3D mask array (non-zero = ROI).
    binWidth : float, optional
        Width of intensity bins for quantization. Default 1.0.
    levels : int, optional
        Number of gray levels. If None, determined from binWidth automatically.
    distance : int, optional
        Neighborhood radius. Default is 1 (3x3x3 neighborhood).

    Returns
    -------
    dict
        Dictionary with 5 NGTDM features:
        - Coarseness
        - Contrast
        - Busyness
        - Complexity
        - Strength
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
        raise ValueError("ROI has constant intensity; NGTDM cannot be computed.")

    if levels is None:
        levels = int(np.ceil((max_val - min_val) / binWidth)) + 1

    img_quant = np.floor((roi_image - min_val) / binWidth).astype(int)
    img_quant = np.clip(img_quant, 0, levels - 1)

    # Neighborhood kernel (3D cube)
    kernel = np.ones((2*distance+1, 2*distance+1, 2*distance+1), dtype=np.float64)
    kernel[distance, distance, distance] = 0  # exclude center voxel
    neighbor_count = np.sum(kernel)

    # Compute local mean of neighbors
    local_sum = convolve(img_quant * roi_mask, kernel, mode='constant', cval=0)
    local_count = convolve(roi_mask.astype(np.float64), kernel, mode='constant', cval=0)
    local_mean = np.zeros_like(img_quant, dtype=np.float64)
    mask_valid = local_count > 0
    local_mean[mask_valid] = local_sum[mask_valid] / local_count[mask_valid]

    # Compute S_i: sum of absolute differences for each gray level
    S_i = np.zeros(levels, dtype=np.float64)
    N_i = np.zeros(levels, dtype=np.float64)

    for g in range(levels):
        mask_g = (img_quant == g) & roi_mask & mask_valid
        N_i[g] = np.sum(mask_g)
        if N_i[g] > 0:
            S_i[g] = np.sum(np.abs(img_quant[mask_g] - local_mean[mask_g]))

    # Total number of voxels with neighbors
    N_total = np.sum(N_i)

    # Avoid division by zero
    S_i[S_i == 0] = 1e-12
    N_i[N_i == 0] = 1e-12

    # Compute features (Pyradiomics definitions)
    coarseness = N_total / np.sum(N_i / S_i)
    contrast = np.sum((np.arange(levels).reshape(-1,1) - np.arange(levels).reshape(1,-1))**2 *
                      S_i[:, None] * S_i[None, :]) / N_total
    busyness = np.sum(S_i) / np.sum(np.abs(np.arange(levels) * N_i - np.sum(np.arange(levels) * N_i)))
    complexity = np.sum(np.abs(np.arange(levels) * N_i - np.arange(levels).reshape(-1,1) * N_i[:, None]) *
                        S_i[:, None] / N_total)
    strength = np.sum(S_i * (np.arange(levels)**2)) / np.sum(S_i + 1e-12)

    features = {
        "Coarseness": coarseness,
        "Contrast": contrast,
        "Busyness": busyness,
        "Complexity": complexity,
        "Strength": strength
    }

    return features
