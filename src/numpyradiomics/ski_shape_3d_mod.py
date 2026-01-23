import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import interpn
import skimage



def ski_shape_3d(arr, spacing=(1.0, 1.0, 1.0)):

    # Add zeropadding at the boundary slices for masks that extend to the edge
    # Motivation: this could have some effect if surfaces are extracted - could create issues
    # if the values extend right up to the boundary of the slab.
    shape = list(arr.shape)
    shape[-1] = shape[-1] + 2*4
    array = np.zeros(shape)
    array[:,:,4:-4] = arr

    # Get voxel dimensions from the affine
    # We are assuming here the voxel dimensions are in mm.
    # If not the units provided with the return values are incorrect.
    voxel_volume = spacing[0]*spacing[1]*spacing[2]
    nr_of_voxels = np.count_nonzero(array > 0.5)
    volume = nr_of_voxels * voxel_volume

    # Surface properties - for now only extracting surface area
    try:
        # Note: this is smoothing the surface first - not tested in depth whether this is necessary or helpful.
        # It does appear to make a big difference on surface area so should be looked at more carefully.
        smooth_array = ndi.gaussian_filter(array, 1.0)
        verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=0.5, step_size=1.0)
    except:
        # If a mask has too few points, smoothing can reduce the max to below 0.5. Use the midpoint in that case
        # Note this may work in general but 0.5 has been used for previous data collection so keep that as default
        smooth_array = ndi.gaussian_filter(array, 1.0)
        verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=np.mean(smooth_array), step_size=1.0)
    surface_area = skimage.measure.mesh_surface_area(verts, faces)

    # Interpolate to isotropic for non-isotropic voxels
    # Motivation: this is required by the region_props function
    spacing = np.array(spacing)
    if np.amin(spacing) != np.amax(spacing):
        array, isotropic_spacing = interpolate3d_isotropic(array, spacing)
        isotropic_voxel_volume = isotropic_spacing**3
    else:
        isotropic_spacing = np.mean(spacing)
        isotropic_voxel_volume = voxel_volume

    # Get volume properties - mostly from region_props, except for compactness and depth
    array = np.round(array).astype(np.int16)
    region_props_3D = skimage.measure.regionprops(array)[0]

    # Calculate 'compactness' (our definition) - define as volume to surface ratio
    # expressed as a percentage of the volume-to-surface ration of an equivalent sphere.
    # The sphere is the most compact of all shapes, i.e. it has the largest volume to surface area ratio,
    # so this is guaranteed to be between 0 and 100%
    radius = region_props_3D['equivalent_diameter_area']*isotropic_spacing/2 # mm
    v2s = volume/surface_area # mm
    v2s_equivalent_sphere = radius/3 # mm
    compactness = 100 * v2s/v2s_equivalent_sphere # %

    # Fractional anisotropy - in analogy with FA in diffusion 
    m0 = region_props_3D['inertia_tensor_eigvals'][0]
    m1 = region_props_3D['inertia_tensor_eigvals'][1]
    m2 = region_props_3D['inertia_tensor_eigvals'][2]
    m = (m0 + m1 + m2)/3 # average moment of inertia (trace of the inertia tensor)
    FA = np.sqrt(3/2) * np.sqrt((m0-m)**2 + (m1-m)**2 + (m2-m)**2) / np.sqrt(m0**2 + m1**2 + m2**2)

    # Measure maximum depth (our definition)
    distance = ndi.distance_transform_edt(array)
    max_depth = np.amax(distance)

    # TODO: some of these fail (math error) for masks with limited non-zero values - test and catch
    return {
        'surface_area': surface_area/100,
        'volume': volume/1000,
        'bounding_box_volume': region_props_3D['area_bbox']*isotropic_voxel_volume/1000,
        'convex_hull_volume': region_props_3D['area_convex']*isotropic_voxel_volume/1000,
        'extent': region_props_3D['extent']*100,    # Percentage of bounding box filled
        'solidity': region_props_3D['solidity']*100,   # Percentage of convex hull filled
        'compactness': compactness,
        'long_axis_length': region_props_3D['axis_major_length']*isotropic_spacing/10,
        'short_axis_length': region_props_3D['axis_minor_length']*isotropic_spacing/10,
        'equivalent_diameter': region_props_3D['equivalent_diameter_area']*isotropic_spacing/10,
        'maximum_depth': max_depth*isotropic_spacing/10,
        'primary_moment_of_inertia': region_props_3D['inertia_tensor_eigvals'][0]*isotropic_spacing**2/100,
        'second_moment_of_inertia': region_props_3D['inertia_tensor_eigvals'][1]*isotropic_spacing**2/100,
        'third_moment_of_inertia': region_props_3D['inertia_tensor_eigvals'][2]*isotropic_spacing**2/100,
        'mean_moment_of_inertia': m*isotropic_spacing**2/100,
        'fractional_anisotropy_of_inertia': 100*FA,
        'volume_qc': region_props_3D['area']*isotropic_voxel_volume/1000,
        # Taking this out for now - computation uses > 32GB of memory for large masks
        # 'ski_longest_caliper_diameter': region_props_3D['feret_diameter_max']*isotropic_spacing/10, 
    }


ski_shape_3d_units = {
    'surface_area': 'cm^2',
    'volume': 'mL',
    'bounding_box_volume': 'mL',
    'convex_hull_volume': 'mL',
    'extent': '%',    # Percentage of bounding box filled
    'solidity': '%',   # Percentage of convex hull filled
    'compactness': '%',
    'long_axis_length': 'cm',
    'short_axis_length': 'cm',
    'equivalent_diameter': 'cm',
    'maximum_depth': 'cm',
    'primary_moment_of_inertia': 'cm^2',
    'second_moment_of_inertia': 'cm^2',
    'third_moment_of_inertia': 'cm^2',
    'mean_moment_of_inertia': 'cm^2',
    'fractional_anisotropy_of_inertia': '%',
    'volume_qc': 'mL',
    'ski_longest_caliper_diameter': 'cm',
}



def interpolate3d_isotropic(array, spacing, isotropic_spacing=None):

    if isotropic_spacing is None:
        isotropic_spacing = np.amin(spacing)

    # Get x, y, z coordinates for array
    nx = array.shape[0]
    ny = array.shape[1]
    nz = array.shape[2]
    Lx = (nx-1)*spacing[0]
    Ly = (ny-1)*spacing[1]
    Lz = (nz-1)*spacing[2]
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    # Get x, y, z coordinates for isotropic array
    nxi = 1 + np.floor(Lx/isotropic_spacing)
    nyi = 1 + np.floor(Ly/isotropic_spacing)
    nzi = 1 + np.floor(Lz/isotropic_spacing)
    Lxi = (nxi-1)*isotropic_spacing
    Lyi = (nyi-1)*isotropic_spacing
    Lzi = (nzi-1)*isotropic_spacing
    xi = np.linspace(0, Lxi, nxi.astype(int))
    yi = np.linspace(0, Lyi, nyi.astype(int))
    zi = np.linspace(0, Lzi, nzi.astype(int))

    # Interpolate to isotropic
    ri = np.meshgrid(xi,yi,zi, indexing='ij')
    array = interpn((x,y,z), array, np.stack(ri, axis=-1))
    return array, isotropic_spacing