def ndvi_xarray(img, red, nir):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) from a given image.

    NDVI is calculated using the formula: (NIR - Red) / (NIR + Red + 1E-10).
    The input image bands are implicitly converted to Float32 for the calculation.

    Parameters:
    img (xarray.DataArray): The input image as an xarray DataArray.
    red (int or str): The band index or name corresponding to the red band.
    nir (int or str): The band index or name corresponding to the near-infrared (NIR) band.

    Returns:
    xarray.DataArray: The NDVI values as an xarray DataArray.
    """
    """Calculates the NDVI from a given image. Implicitly converts to Float32."""
    redl = img.sel(band=red).astype('float32')
    nirl = img.sel(band=nir).astype('float32')
    return (nirl - redl) / (nirl + redl + 1E-10)


def ndvi(img, red, nir, axis=-1):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) from a given image.

    NDVI is calculated using the formula: (NIR - Red) / (NIR + Red + 1E-10)
    The function implicitly converts the input image bands to Float32 to ensure precision.

    Parameters:
    img (numpy.ndarray): The input image array.
    red (int): The index of the red band in the image array.
    nir (int): The index of the near-infrared (NIR) band in the image array.
    axis (int, optional): The axis along which the bands are indexed. Default is -1.
                          -1 or 2 indicates that the bands are in the last dimension.
                          0 indicates that the bands are in the first dimension.

    Returns:
    numpy.ndarray: The NDVI values as a float32 array.

    Raises:
    ValueError: If the specified axis is not supported.
    """
    if axis == -1 or axis == 2:
        redl = img[:, :, red].astype('float32')
        nirl = img[:, :, nir].astype('float32')
    elif axis == 0:
        redl = img[red].astype('float32')
        nirl = img[nir].astype('float32')
    else:
        raise ValueError("Calculating NDVI along axis {} not supported.".format(axis))
    return (nirl - redl) / (nirl + redl + 1E-10)


def gci(img, red, green, nir, axis=-1):
    """
    Calculates the Green Chlorophyll Index (GCI) from a given image. Implicitly converts to Float32.

    Parameters:
    img (numpy.ndarray): The input image array.
    red (int): The index of the red band in the image.
    green (int): The index of the green band in the image.
    nir (int): The index of the near-infrared (NIR) band in the image.
    axis (int, optional): The axis along which the bands are indexed. Default is -1.

    Returns:
    numpy.ndarray: The calculated Green Chlorophyll Index (GCI) as a float32 array.

    Raises:
    ValueError: If the specified axis is not supported.
    """
    if axis == -1 or axis == 2:
        redl = img[:, :, red].astype('float32')
        greenl = img[:, :, green].astype('float32')
        nirl = img[:, :, nir].astype('float32')
    elif axis == 0:
        redl = img[red].astype('float32')
        greenl = img[green].astype('float32')
        nirl = img[nir].astype('float32')
    else:
        raise ValueError("Calculating GCI along axis {} not supported.".format(axis))
    return (nirl - greenl) / (nirl + greenl + 1E-10)

def hue(img, red, green, blue, axis=-1):
    """
    Calculates the Hue from a given image. Implicitly converts to Float32.

    Parameters:
    img (numpy.ndarray): The input image array.
    red (int): The index of the red channel.
    green (int): The index of the green channel.
    blue (int): The index of the blue channel.
    axis (int, optional): The axis along which to calculate the hue. Default is -1 (last axis).

    Returns:
    numpy.ndarray: The calculated hue values.

    Raises:
    ValueError: If the specified axis is not supported.
    """
    if axis == -1 or axis == 2:
        redl = img[:, :, red].astype('float32')
        greenl = img[:, :, green].astype('float32')
        bluel = img[:, :, blue].astype('float32')
    elif axis == 0:
        redl = img[red].astype('float32')
        greenl = img[green].astype('float32')
        bluel = img[blue].astype('float32')
    else:
        raise ValueError("Calculating Hue along axis {} not supported.".format(axis))
    hue = 0.5 * (2 * redl - greenl - bluel) / np.sqrt((redl - greenl) ** 2 + (redl - bluel) * (greenl - bluel))
    return hue