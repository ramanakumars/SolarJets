import numpy
import json

import astropy.units as u

from panoptes_client import Subject
from sunpy.map import Map


def solar_conversion(sub, x, y, metadata):
    '''
    Convert from pixel coordinates in the Zooniverse subjects to solar coordinates 
        Inputs
        ------
        sub : int
            subject id of the Zooniverse subject
        x : str or int or float 
            x pixel location 
        y : str or int or float
            y pixel location

        Output
        ------
        x_sun : float 
            Solar X loaction in arcsec
        y_sun : float 
            Solar Y loaction in arcsec
    '''

    # Change de Y pixels to Height-Y since the pixel frame is defined inverted from the Zooniverse processor calculation
    y = float(metadata['#height'])-y

    # Convert coordinates using sunpy
    wc = world_from_pixel(sub, float(x), float(y), metadata)

    # Extract desired values from sunpy map
    solar_x, solar_y = wc.Tx, wc.Ty

    # Reduce values to just the numerical values we want (Heliographic Projection X + Y)
    x_sun = str(solar_x).split('arcsec')[0]
    y_sun = str(solar_y).split('arcsec')[0]
    return float(x_sun), float(y_sun)


def world_from_pixel(subject_id: int, x, y, metadata):
    '''
    Gets the solar coordinates of point (x,y) on selected image

    Parameters
    ----------
    subject_id : int
        The subject id # for the image that we are extracting coordinates from
    x : float
        The x value to be converted
    y : float
        The y value to be converted
    '''

    width = float(metadata["#width"])

    height = float(metadata["#height"])
    # y= height-y # To account for the inverted y axis in pixel coordinates

    # Normalize x and y if needed
    if x > 1 and y > 1:
        x = x / width
        y = y / height

    # Try to collect the fits_headers directly from the metadata
    fits_headers = {
        'naxis1': metadata['#naxis1'],  # Pixels along axis 1
        'naxis2': metadata['#naxis2'],  # Pixels along axis 2
        # Units of the coordinate increments along naxis1 e.g. arcsec
        'cunit1': metadata['#cunit1'],
        # Units of the coordinate increments along naxis2 e.g. arcsec
        'cunit2': metadata['#cunit2'],
        'crval1': metadata['#crval1'],  # Coordinate value at reference point on naxis1
        'crval2': metadata['#crval2'],  # Coordinate value at reference point on naxis2
        # Spatial scale of pixels for naxis1, i.e. coordinate increment at reference point
        'cdelt1': metadata['#cdelt1'],
        # Spatial scale of pixels for naxis2, i.e. coordinate increment at reference point
        'cdelt2': metadata['#cdelt2'],
        'crpix1': metadata['#crpix1'],  # Pixel coordinate at reference point naxis1
        'crpix2': metadata['#crpix2'],  # Pixel coordinate at reference point naxis2
        'crota2': metadata['#crota2'],  # Rotation of the horizontal and vertical axes in degrees
    }

    # Create an empty map
    map = Map(numpy.zeros((1, 1)), fits_headers)

    # extract important pieces of metadata
    fits_width = float(map.meta["naxis1"])
    fits_height = float(map.meta["naxis2"])
    im_ll_x = float(metadata['#im_ll_x'])
    im_ll_y = float(metadata['#im_ll_y'])
    im_ur_x = float(metadata['#im_ur_x'])
    im_ur_y = float(metadata['#im_ur_y'])

    # Fit normalized pixel coordinates within the correct image box
    axis_x_normalized = (x - im_ll_x) / (im_ur_x - im_ll_x)
    axis_y_normalized = (y - im_ll_y) / (im_ur_y - im_ll_y)

    # Get correctly scaled pixel coordinates
    pix_x = axis_x_normalized * fits_width
    pix_y = axis_y_normalized * fits_height

    # Call on Sunpy to finish the pixel_to_world conversion
    return map.pixel_to_world(pix_x * u.pix, pix_y * u.pix)
