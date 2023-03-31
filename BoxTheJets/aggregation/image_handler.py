import csv
import numpy
import os

import solar.common.mapproc as mapproc

from solar.database.tables import Fits_File, Visual_File
from sunpy.map import Map
from pathlib import Path

def world_from_pixel(image_file, x, y):
    '''
    Gets the solar coordinates of point (x,y) on selected image
    !!! Requires access to the database in order to work !!!

    Parameters
    ----------
    image_file : solar.database.tables.Visual_File
        The Visual File that we are analyzing
    x : float
        The x value to be converted
    y : float
        The y value to be converted
    '''

    #Derive the name of the .fts file from the given image
    image_name = str(image_file.file_name)
    fits_name = image_name[:image_name.rfind('_')] + '_.fts'

    #Get the correct .fts file from the database
    fits_file = Fits_File.select().where(Fits_File.file_name == fits_name)[0]

    #Collect the fits headers as a dictionary for creating sunpy map
    fits_headers = fits_file.get_header_as_dict()

    #Normalize x and y if needed
    if x > 1 and y > 1:
        x = x / image_file.width
        y = y / image_file.height

    try: #Check to see if fits headers are enough to create the temporary sunpy map
        #Create an empty map
        fake_map = Map(numpy.zeros((1,1)), fits_headers)
    except: #Fits headers do not have all of the necessary data
        fits_path = str(fits_file.file_path)
        
        #Find .fts file path
        if not os.path.isfile(fits_path):
            harddrive = '/Volumes/LaCie/Solarjet_Zooniversedata/'
            if os.path.isfile(harddrive + fits_path):
                fits_path = harddrive + fits_path
            else:
                raise FileNotFoundError('Fits file not found at either location:\n' + fits_path + '\n' + harddrive + fits_path)
        
        #Create a sunpy map to extract header data
        fake_map = Map(Path(fits_path))
    
    #Call on mapproc to retrieve the coordinates
    return mapproc.world_from_pixel_norm(fake_map, image_file, x, y)
    
def get_metadata(file, metadata = 'files/exports/meta.csv'):
    """
    Gets the first row of metadata which contains an image with name 'file'

    Parameters
    ----------
    file : str
        Name of the image whose metadata we want
    metadata : str
        Location of the metadata file
        Default is: 'files/exports/meta.csv'

    Returns
    -------
    meta_dict : dict
        A dictionary containing the metadata values 
    """

    #Load .csv file as an array
    with open(metadata, newline='') as metadata_file:
        raw_data = csv.reader(metadata_file)

    #Read header and extract column #s containing image names
    header_row = raw_data[0]
    header_cols = []
    for col, header in enumerate(header_row):
        if header.startswith("#file_name_"):
            header_cols.append(col)
    
    #Search each row's selected columns
    wanted_row = None
    for index, row in enumerate(raw_data):
        file_names = [row[i] for i in header_cols]
        #Find the first row that contains 'file' in one of the selected columns
        for name in file_names:
            if name == file:
                wanted_row = index
                break
        
        #Ends search once row is found
        if wanted_row is not None:
            break

    #Organize row into a dictionary using the headers as keys
    meta_dict = {}
    for index in range(raw_data[0]):
        meta_dict[raw_data[0][index]] = raw_data[wanted_row][index]

    return meta_dict

def extract_from_header(fake_key, fake_dict):
    """
    Extracts from the fake dictionary the value matching the key of attribute

    Parameters
    ----------
    fake_key : str
        The key that we are searching for
    fake_dict : str
        The fake dictionary/json input of the fits header
        Expects the raw string extracted from the metadata

    Returns
    -------
    value : str
        The value associated with fake_key in fake_dict
    """

    start_index = fake_dict.find(fake_key) + len(fake_key) + len('\"\": ')
    value = fake_dict[start_index:]
    value = value[:value.find(',')]

    return value