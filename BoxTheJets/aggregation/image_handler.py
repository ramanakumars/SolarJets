import numpy

import astropy.units as u

from panoptes_client import Subject
from sunpy.map import Map

def world_from_pixel(subject_id, x, y):
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

    subject = Subject(subject_id)
    metadata = subject.metadata

    #Normalize x and y if needed
    if x > 1 and y > 1:
        x = x / metadata['#width']
        y = y / metadata['#height']

    #Create an empty map
    '''
    Note:
    At the moment, this will only for subjects with .fts headers included in their metadata.
    Subject sets without this metadata information will not work with this version of the code.
    '''
    map = Map(numpy.zeros((1,1)), metadata['#fits_header_0'])


    #extract important pieces of metadata
    fits_width = map.meta["naxis1"]
    fits_height = map.meta["naxis2"]
    im_ll_x = metadata['#im_ll_x']
    im_ll_y = metadata['#im_ll_y']
    im_ur_x = metadata['#im_ur_x']
    im_ur_y = metadata['#im_ur_y']

    #Fit normalized pixel coordinates within the correct image box
    axis_x_normalized = (x - im_ll_x) / (im_ur_x - im_ll_x)
    axis_y_normalized = (y - im_ll_y) / (im_ur_y - im_ll_y)

    #Get correctly scaled pixel coordinates
    pix_x = axis_x_normalized * fits_width
    pix_y = axis_y_normalized * fits_height

    #Call on Sunpy to finish the pixel_to_world conversion
    return map.pixel_to_world(pix_x * u.pix, pix_y * u.pix)