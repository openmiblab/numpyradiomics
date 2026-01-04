import numpy as np

def glcm_features(input_image, input_mask, distances=[1], binWidth=25, levels=None,
                  symmetric=True, normalized=True):
    """
    Vectorized computation of 2D/3D GLCM features with intensity quantization based on binWidth.

    Parameters:
        input_image (np.ndarray): 2D or 3D image array.
        input_mask (np.ndarray): 2D or 3D mask array (non-zero = ROI).
        distances (list of int, optional): Distances for voxel/pixel pairs.
        binWidth (float, optional): Width of each intensity bin.
        levels (int, optional): Number of gray levels. If None, computed from binWidth automatically.
        symmetric (bool, optional): Make GLCM symmetric.
        normalized (bool, optional): Normalize GLCM to sum to 1.

    Returns:
        dict: 24 Pyradiomics GLCM features averaged over all offsets.
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
        raise ValueError("ROI has constant intensity; GLCM cannot be computed.")

    if levels is None:
        levels = int(np.ceil((max_val - min_val) / binWidth)) + 1

    img_quant = np.floor((roi_image - min_val) / binWidth).astype(int)
    img_quant = np.clip(img_quant, 0, levels - 1)

    # Determine dimensionality
    dims = input_image.ndim

    # Generate offsets
    offsets = []
    if dims == 2:
        # 2D: four primary directions
        dir_vectors = [(0, 1), (1, 1), (1, 0), (1, -1)]
        for d in distances:
            offsets.extend([(dx*d, dy*d) for dx, dy in dir_vectors])
    elif dims == 3:
        # 3D: 26 neighbors
        offsets = [(dx*d, dy*d, dz*d)
                   for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                   if not (dx==dy==dz==0) for d in distances]
    else:
        raise ValueError("Input must be 2D or 3D")

    # Accumulate features
    feature_sums = {name: 0.0 for name in _glcm_feature_names()}
    count = 0
    for offset in offsets:
        glcm = _compute_glcm_vectorized(img_quant, roi_mask, offset, levels, symmetric, normalized)
        feats = _glcm_features_from_matrix_24(glcm)
        for k in feature_sums:
            feature_sums[k] += feats[k]
        count += 1

    # Average over all offsets
    for k in feature_sums:
        feature_sums[k] /= count

    return feature_sums


def _compute_glcm_vectorized(image, mask, offset, levels, symmetric, normalized):
    """
    Compute a single 2D or 3D GLCM for a given offset using vectorized indexing.
    """
    dims = image.ndim
    if dims == 2:
        dx, dy = offset
        X, Y = image.shape
        x_range = slice(max(0, -dx), min(X, X - dx))
        y_range = slice(max(0, -dy), min(Y, Y - dy))
        x_shift = slice(max(0, dx), min(X, X + dx))
        y_shift = slice(max(0, dy), min(Y, Y + dy))

        valid_mask = mask[x_range, y_range] & mask[x_shift, y_shift]

        vals1 = image[x_range, y_range][valid_mask]
        vals2 = image[x_shift, y_shift][valid_mask]

    else:  # 3D
        dx, dy, dz = offset
        X, Y, Z = image.shape
        x_range = slice(max(0, -dx), min(X, X - dx))
        y_range = slice(max(0, -dy), min(Y, Y - dy))
        z_range = slice(max(0, -dz), min(Z, Z - dz))
        x_shift = slice(max(0, dx), min(X, X + dx))
        y_shift = slice(max(0, dy), min(Y, Y + dy))
        z_shift = slice(max(0, dz), min(Z, Z + dz))

        valid_mask = mask[x_range, y_range, z_range] & mask[x_shift, y_shift, z_shift]

        vals1 = image[x_range, y_range, z_range][valid_mask]
        vals2 = image[x_shift, y_shift, z_shift][valid_mask]

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (vals1, vals2), 1)

    if symmetric:
        glcm = glcm + glcm.T
    if normalized:
        glcm = glcm / (glcm.sum() + 1e-12)

    return glcm




def _glcm_feature_names():
    """
    Returns the 24 standard Pyradiomics GLCM feature names.
    """
    return [
        'autocorrelation', 'joint_average', 'cluster_prominence', 'cluster_shade',
        'cluster_tendency', 'contrast', 'correlation', 'difference_entropy',
        'difference_variance', 'dissimilarity', 'energy', 'entropy', 'homogeneity',
        'homogeneity2', 'inverse_variance', 'max_probability', 'sum_average',
        'sum_entropy', 'sum_squares', 'variance', 'joint_variance',
        'joint_entropy', 'IMC1', 'IMC2'
    ]


def _glcm_features_from_matrix_24(p):
    """
    Compute all 24 Pyradiomics GLCM features from a normalized GLCM matrix.
    """
    # Make sure p is normalized
    p = p / (np.sum(p) + 1e-12)
    levels = p.shape[0]
    i, j = np.indices((levels, levels))

    # Marginals
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    ux = np.sum(i * px)
    uy = np.sum(j * py)
    sx = np.sqrt(np.sum((i - ux)**2 * px))
    sy = np.sqrt(np.sum((j - uy)**2 * py))

    # Core features
    contrast = np.sum((i - j)**2 * p)
    dissimilarity = np.sum(np.abs(i - j) * p)
    homogeneity = np.sum(p / (1.0 + (i - j)**2))
    homogeneity2 = np.sum(p / (1.0 + np.abs(i - j)))
    energy = np.sqrt(np.sum(p**2))
    entropy = -np.sum(p * np.log2(p + 1e-12))
    correlation = np.sum((i - ux) * (j - uy) * p / (sx * sy + 1e-12))
    asm = np.sum(p**2)
    max_probability = np.max(p)
    sum_average = np.sum(np.arange(2, 2*levels+1) * _sum_probabilities(p))
    sum_entropy = -np.sum(_sum_probabilities(p) * np.log2(_sum_probabilities(p) + 1e-12))
    sum_squares = np.sum((i - ux)**2 * p)
    variance = np.sum((i - ux)**2 * p)
    cluster_shade = np.sum(((i + j - ux - uy)**3) * p)
    cluster_prominence = np.sum(((i + j - ux - uy)**4) * p)
    cluster_tendency = cluster_prominence  # identical to prominence in Pyradiomics
    joint_average = np.sum(i * j * p)
    joint_variance = np.sum((i - ux)**2 * p)
    difference_variance = np.var(np.sum(p, axis=0))
    difference_entropy = -np.sum(np.sum(p, axis=0) * np.log2(np.sum(p, axis=0) + 1e-12))
    autocorrelation = np.sum(i * j * p)
    IMC1 = (entropy - (-np.sum(px * np.log2(px + 1e-12) + py * np.log2(py + 1e-12)))) / (max(sx*sy, 1e-12))
    IMC2 = np.sqrt(1 - np.exp(-2 * (entropy - (-np.sum(px * np.log2(px + 1e-12) + py * np.log2(py + 1e-12))))))

    return {
        'autocorrelation': autocorrelation,
        'joint_average': joint_average,
        'cluster_prominence': cluster_prominence,
        'cluster_shade': cluster_shade,
        'cluster_tendency': cluster_tendency,
        'contrast': contrast,
        'correlation': correlation,
        'difference_entropy': difference_entropy,
        'difference_variance': difference_variance,
        'dissimilarity': dissimilarity,
        'energy': energy,
        'entropy': entropy,
        'homogeneity': homogeneity,
        'homogeneity2': homogeneity2,
        'inverse_variance': homogeneity2,  # Pyradiomics maps this
        'max_probability': max_probability,
        'sum_average': sum_average,
        'sum_entropy': sum_entropy,
        'sum_squares': sum_squares,
        'variance': variance,
        'joint_variance': joint_variance,
        'joint_entropy': entropy,
        'IMC1': IMC1,
        'IMC2': IMC2
    }


def _sum_probabilities(p):
    """
    Helper function for sum-related GLCM features.
    Returns the sum probability vector.
    """
    levels = p.shape[0]
    sum_prob = np.zeros(2*levels-1)
    for k in range(2, 2*levels+1):
        sum_prob[k-2] = np.sum(np.where((np.indices((levels, levels))[0] + np.indices((levels, levels))[1] == k-2), p, 0))
    return sum_prob

