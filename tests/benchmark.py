
import numpy as np
import SimpleITK as sitk

# Pyradiomics imports
import radiomics

# Custom implementations
from numpyradiomics import firstorder, glcm, glszm, glrlm, gldm, ngtdm, shape

# Tolerance for numerical comparison
RTOL = 1e-5



# ------------------- TEST TEXTURE FEATURES ------------------- #
def test_first_order_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)

    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

    # Explicit PyRadiomics settings
    settings = {
        "binWidth": 1.0
    }

    # PyRadiomics
    result_py = radiomics.firstorder.RadiomicsFirstOrder(
        itk_img,
        itk_mask,
        **settings
    ).execute()

    # Custom
    result_custom = firstorder(
        img,
        mask,
        **settings
    )

    for key in result_py:
        assert np.isclose(
            result_py[key],
            result_custom[key],
            rtol=RTOL
        ), f"Mismatch in {key}"



def test_glcm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = radiomics.glcm.RadiomicsGLCM(itk_img, itk_mask).execute()
    result_custom = glcm(img, mask)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_glszm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)

    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

    result_py = radiomics.glszm.RadiomicsGLSZM(itk_img, itk_mask).execute()
    result_custom = glszm(img, mask)

    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=1e-6), f"Mismatch in {key}"



def test_glrlm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = radiomics.glrlm.RadiomicsGLRLM(itk_img, itk_mask).execute()
    result_custom = glrlm(img, mask, binWidth=1)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_gldm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = radiomics.gldm.RadiomicsGLDM(itk_img, itk_mask).execute()
    result_custom = gldm(img, mask, binWidth=1)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_ngtdm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = radiomics.ngtdm.RadiomicsNGTDM(itk_img, itk_mask).execute()
    result_custom = ngtdm(img, mask, binWidth=1)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


# ------------------- TEST SHAPE FEATURES ------------------- #
def test_shape_features_2d():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 3:7] = 1
    
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    result_py = radiomics.shape.RadiomicsShape2D(itk_mask).execute()
    result_custom = shape(mask)
    
    for key in result_custom:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_shape_features_3d():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:8, 3:7, 1:9] = 1
    
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    voxel_spacing = (1.0, 1.0, 1.0)
    
    result_py = radiomics.shape.RadiomicsShape3D(itk_mask).execute()
    result_custom = shape(mask, voxel_spacing)
    
    for key in result_custom:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


if __name__=='__main__':
    # test_shape_features_3d()
    # test_shape_features_2d()
    # test_ngtdm_features()
    # test_gldm_features()
    # test_glrlm_features()
    # test_glszm_features()
    # test_glcm_features()
    test_first_order_features()
