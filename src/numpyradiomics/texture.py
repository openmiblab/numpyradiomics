
# Import the vectorized feature functions you defined previously
from numpyradiomics.first_order import first_order_features
from numpyradiomics.glcm import glcm_features
from numpyradiomics.glszm import glszm_features
from numpyradiomics.glrlm import glrlm_features
from numpyradiomics.ngtdm import ngtdm_features
from numpyradiomics.gldm import gldm_features


def texture_features(input_image, input_mask,
                     binWidth=1.0, glcm_distance=1, glcm_angles=None,
                     gldm_distance=1, ngtdm_distance=1,
                     connectivity=None, voxelVolume=1, voxelArrayShift=0):
    """
    Compute full Pyradiomics-style features for 2D or 3D images.

    Features include:
        - First-Order (19)
        - GLCM (24)
        - GLSZM (16)
        - GLRLM (16)
        - NGTDM (5)
        - GLDM (14)

    Parameters
    ----------
    input_image : np.ndarray
        2D or 3D image array.
    input_mask : np.ndarray
        Same shape as input_image. Non-zero = ROI.
    binWidth : float
        Intensity bin width for quantization.
    glcm_distance : int
        Distance for GLCM computation.
    glcm_angles : list of tuples or None
        Angles for GLCM directions (3D vectors). Default None.
    gldm_distance : int
        Distance for GLDM dependencies.
    ngtdm_distance : int
        Neighborhood radius for NGTDM.
    connectivity : int or None
        Neighborhood connectivity:
            - 2D: 4 or 8
            - 3D: 6, 18, 26
    voxelVolume : float
        Volume per voxel.
    voxelArrayShift : float
        Value added to voxel intensities before some computations.

    Returns
    -------
    dict
        Combined dictionary of all features.
    """

    dims = input_image.ndim
    if connectivity is None:
        connectivity = 8 if dims == 2 else 26

    # ---------------- FIRST-ORDER ---------------- #
    first_order_dict = first_order_features(input_image, input_mask,
                                   voxelVolume=voxelVolume,
                                   binWidth=binWidth,
                                   voxelArrayShift=voxelArrayShift)

    # ---------------- GLCM ---------------- #
    glcm_dict = glcm_features(input_image, input_mask,
                              binWidth=binWidth,
                              distances=[glcm_distance],
                              symmetric=True,
                              normalized=True)

    # ---------------- GLSZM ---------------- #
    glszm_dict = glszm_features(input_image, input_mask,
                                binWidth=binWidth,
                                connectivity=connectivity)

    # ---------------- GLRLM ---------------- #
    glrlm_dict = glrlm_features(input_image, input_mask,
                                binWidth=binWidth,
                                connectivity=connectivity)

    # ---------------- NGTDM ---------------- #
    ngtdm_dict = ngtdm_features(input_image, input_mask,
                                distance=ngtdm_distance)

    # ---------------- GLDM ---------------- #
    gldm_dict = gldm_features(input_image, input_mask,
                              binWidth=binWidth,
                              distance=gldm_distance,
                              connectivity=connectivity)

    # ---------------- COMBINE ---------------- #
    all_features = {}
    all_features.update(first_order_dict)
    all_features.update(glcm_dict)
    all_features.update(glszm_dict)
    all_features.update(glrlm_dict)
    all_features.update(ngtdm_dict)
    all_features.update(gldm_dict)

    return all_features
