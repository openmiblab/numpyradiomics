import numpy as np
from skimage.measure import regionprops, label, perimeter
from skimage.morphology import convex_hull_image

def shape_2d(input_mask, spacing=(1.0, 1.0)):
    """
    Compute 2D shape features for a binary mask (similar to Pyradiomics shape_2D).

    Parameters
    ----------
    input_mask : np.ndarray
        2D binary mask (non-zero = ROI)
    pixel_spacing : tuple of float
        Pixel spacing in mm (row_spacing, col_spacing)

    Returns
    -------
    dict
        Dictionary of 2D shape features:
        - Area
        - Perimeter
        - MajorAxisLength
        - MinorAxisLength
        - Eccentricity
        - Solidity
        - Extent
        - ConvexArea
        - EquivalentDiameter
        - Circularity
    """

    mask = input_mask > 0
    if not np.any(mask):
        raise ValueError("Mask contains no pixels.")

    # Label connected components
    lbl = label(mask)
    props = regionprops(lbl, spacing=spacing)

    # Use largest region if multiple
    region = max(props, key=lambda r: r.area)

    # Area and convex area
    area = region.area
    convex_area = region.convex_area
    solidity = convex_area / area if area > 0 else 0

    # Perimeter
    peri = perimeter(mask, neighbourhood=8)
    # Equivalent diameter
    equiv_diam = np.sqrt(4 * area / np.pi)

    # Major / minor axes
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length

    # Eccentricity
    ecc = region.eccentricity

    # Extent: ratio of area to bounding box
    extent = region.extent

    # Circularity: 4*pi*area / perimeter^2
    circ = 4 * np.pi * area / (peri**2 + 1e-12)

    features = {
        "Area": 0.01 * area,
        "Perimeter": 0.1 * peri,
        "MajorAxisLength": 0.1 * major_axis,
        "MinorAxisLength": 0.1 * minor_axis,
        "Eccentricity": 100 * ecc,
        "Solidity": 100 * solidity,
        "Extent": 100 * extent,
        "ConvexArea": 0.01 * convex_area,
        "EquivalentDiameter": 0.1 * equiv_diam,
        "Circularity": 100 * circ,
    }

    return features


shape_2d_units = {
    "Area": 'cm^2',
    "Perimeter": 'cm',
    "MajorAxisLength": 'cm',
    "MinorAxisLength": 'cm',
    "Eccentricity": '%',
    "Solidity": '%',
    "Extent": '%',
    "ConvexArea": 'cm^2',
    "EquivalentDiameter": 'cm',
    "Circularity": '%',
}
