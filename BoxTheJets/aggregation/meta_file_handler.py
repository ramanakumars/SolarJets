import json
import tqdm
import datetime
import numpy as np
from dateutil.parser import parse

# First three functions are to read out the Zooniverse subjects file and make the metafile
# The class MetaFile is to read out the produced metafile


def convert_fileName_to_datetime(fileName: str):
    '''
        Takes a Zooniverse filename and converts it to the datetime format

        Inputs
        ------
        filename: str
            Filename for the imaga in zooniverse format is ssw_cutout_YYYYMMdd_hhmmss_aia_304_####.png
        Outputs
        ------
            datetime format in UTC
    '''

    try:
        dateTimeComponents = fileName.split('_')[2:4]
        return parse(f'{dateTimeComponents[0]}T{dateTimeComponents[1]}')
    except BaseException:
        print('dateTime could not be extracted')
        return ''


def create_subjectinfo(subject, subjectsdata):
    '''
        Takes a Zooniverse subjectsdata and makes a reduced meta dictionary containing chosen keys only
        Inputs
        ------
        subject: int
            Zooniverse subject ID
        subjectsdata: astropy.table.table.Table
            data Table with Zooniverse metadata keys ['subject_id','metadata']
        Outputs
        ------
        wantedDict: dict
            dictionary of the wanted metadata for the given subject
    '''

    # Select keys we want to write to json file
    keysToImport = [
        '#file_name_0',  # First image filename in Zooniverse subject
        '#file_name_14',  # Last image filename in Zooniverse subject
        '#sol_standard',  # HEK coronal jet event name
        '#width',  # Width of the image in pixel
        '#height',  # Height of the image in pixel
        '#naxis1',  # Pixels along axis 1
        '#naxis2',  # Pixels along axis 2
        '#cunit1',  # Units of the coordinate increments along naxis1 e.g. arcsec
        '#cunit2',  # Units of the coordinate increments along naxis2 e.g. arcsec
        '#crval1',  # Coordinate value at reference point on naxis1
        '#crval2',  # Coordinate value at reference point on naxis2
        '#cdelt1',  # Spatial scale of pixels for naxis1, i.e. coordinate increment at reference point
        '#cdelt2',  # Spatial scale of pixels for naxis2, i.e. coordinate increment at reference point
        '#crpix1',  # Pixel coordinate at reference point naxis1
        '#crpix2',  # Pixel coordinate at reference point naxis2
        '#crota2',  # Rotation of the horizontal and vertical axes in degrees
        '#im_ll_x',  # Vertical distance in pixels between bottom left corner and start solar image
        '#im_ll_y',  # Horizontal distance in pixels between bottom left corner and start solar image
        '#im_ur_x',  # Vertical distance in pixels between bottom left corner and end solar image
        '#im_ur_y'  # Horizontal distance in pixels between bottom left corner and end solar image
    ]
    try:
        allData = json.loads(subjectsdata['metadata'][subjectsdata['subject_id'] == subject][-1])
        wantedDict = {key: allData[key] for key in keysToImport}
        wantedDict['startDate'] = str(convert_fileName_to_datetime(wantedDict['#file_name_0']))
        wantedDict['endDate'] = str(convert_fileName_to_datetime(wantedDict['#file_name_14']))
        wantedDict['#width'] = float(wantedDict['#width'])
        wantedDict['#height'] = float(wantedDict['#height'])
        del allData  # does this help memory management?
        return wantedDict
    except BaseException:
        print('')
        print(
            f"Not all metadata available for subject {subject} atempting to gather minimal information")
        try:
            allData = json.loads(
                subjectsdata['metadata'][subjectsdata['subject_id'] == subject][-1])
            reducedwantedDict = {key: allData[key] for key in [
                "#file_name_0", "#file_name_14", "#sol_standard"]}
            reducedwantedDict['startDate'] = str(
                convert_fileName_to_datetime(reducedwantedDict['#file_name_0']))
            reducedwantedDict['endDate'] = str(
                convert_fileName_to_datetime(reducedwantedDict['#file_name_14']))
            return reducedwantedDict
        except BaseException:
            print(f"something went wrong while writing subject {subject}")
            return {}


def create_allmetadata_subjectinfo(subject, subjectsdata):
    '''
        Takes a Zooniverse subjectsdata and makes a meta dictionary containing all keys
        'width', 'cdelt1', 'cdelt2', 'crota2', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cunit1', 'cunit2', height', 'naxis1', 'naxis2'
        'ssw_id', 'im_ll_x', 'im_ll_y', 'im_ur_x', 'im_ur_y', 'description', 'event_db_id',  'sol_standard', 'frame_per_sub', 'stddev_crota2'
        'vis_db_0', 'vis_db_1', 'vis_db_2', 'vis_db_3', 'vis_db_4', 'vis_db_5', 'vis_db_6', 'vis_db_7', 'vis_db_8', 'vis_db_9', 'vis_db_10', 'vis_db_11', 'vis_db_12', 'vis_db_13', 'vis_db_14'
        'fits_db_0', 'fits_db_1', 'fits_db_2', 'fits_db_3', 'fits_db_4', 'fits_db_5', 'fits_db_6', 'fits_db_7', 'fits_db_8', 'fits_db_9', 'fits_db_10', 'fits_db_11', 'fits_db_12', 'fits_db_13', 'fits_db_14'
        'file_name_0', 'file_name_1', 'file_name_2', 'file_name_3', 'file_name_4', 'file_name_5', 'file_name_6', 'file_name_7', 'file_name_8', 'file_name_9', 'visual_type', 'file_name_10', 'file_name_11', 'file_name_12', 'file_name_13', 'file_name_14'
        Inputs
        ------
        subject: int
            Zooniverse subject ID
        subjectsdata: astropy.table.table.Table
            data Table with Zooniverse metadata keys ['subject_id','metadata']
        Outputs
        ------
        wantedDict: dict
            dictionary of the renamed keys metadata for the given subject
    '''

    try:
        allData = json.loads(subjectsdata['metadata'][subjectsdata['subject_id'] == subject][-1])
        wantedDict = wantedDict = {key[1:]: allData[key] for key in allData.keys()}
        wantedDict['startDate'] = str(convert_fileName_to_datetime(wantedDict['#file_name_0']))
        wantedDict['endDate'] = str(convert_fileName_to_datetime(wantedDict['#file_name_14']))
        wantedDict['#width'] = float(wantedDict['#width'])
        wantedDict['#height'] = float(wantedDict['#height'])
        del allData  # does this help memory management?
        return wantedDict
    except BaseException:
        print('')
        print(
            f"Not all metadata available for subject {subject} atempting to gather minimal information")
        try:
            allData = json.loads(
                subjectsdata[subjectsdata['subject_id'] == subject][-1]['metadata'])
            reducedwantedDict = {key: allData[key] for key in [
                "#file_name_0", "#file_name_14", "#sol_standard"]}
            reducedwantedDict['startDate'] = str(
                convert_fileName_to_datetime(reducedwantedDict['#file_name_0']))
            reducedwantedDict['endDate'] = str(
                convert_fileName_to_datetime(reducedwantedDict['#file_name_14']))
            return reducedwantedDict
        except BaseException:
            print(f"something went wrong while writing subject {subject}")
            return {}


def create_metadata_jsonfile(filename: str, subjectstoloop: np.array, subjectsdata):
    '''
        Write out the metadata file for a given set of subjectstoloop to filename
        Inputs
        ------
        filename: str
            Filename to where the metadata should be written to
        subjectstoloop: np.array
            List of subjects for which the metadata should be gathered
        subjectsdata: astropy.table.table.Table
            data Table with Zooniverse metadata keys ['subject_id','metadata']

    '''
    file = open(filename, 'w')
    file.write('[')
    for i, subject in enumerate(tqdm.tqdm(subjectstoloop, ascii=True, desc='Writing subjects to JSON')):
        subjectDict = {}
        subjectDict['subjectId'] = int(subject)
        subjectDict['data'] = create_subjectinfo(subject, subjectsdata)
        if i != len(subjectstoloop) - 1:
            file.write(json.dumps(subjectDict, indent=3) + ',')
        else:
            file.write(json.dumps(subjectDict, indent=3) + ']')
    file.close()
    print(' ')
    print("succesfully wrote subject information to file " + filename)


class MetaFile:
    '''
        Data class to read out the meta data for each of the subjects given in Zooniverse
    '''

    def __init__(self, file_name: str):
        '''
            Inputs
            ------
            file_name : meta data json file
                Contains for each subject a set of meta data
                keys {'#file_name_0','#file_name_14', '#sol_standard', '#width','#height',
                     '#naxis1', '#naxis2', '#cunit1', '#cunit2','#crval1','#crval2', '#cdelt1', '#cdelt2',
                     '#crpix1', '#crpix2', '#crota2', '#im_ll_x', '#im_ll_y','#im_ur_x', '#im_ur_y'}
        '''
        try:
            data = json.load(open(file_name))
        except FileNotFoundError:
            print(f'{file_name} was not found')
            return
        except BaseException:
            print('This file could not be read out as a json, please check the format')
            return

        self.file_name = file_name
        self.data = data
        self.subjects = np.asarray([x['subjectId'] for x in data])
        self.SOL_unique = np.unique([x['data']['#sol_standard'] for x in data])

    def get_subjectid_by_solstandard(self, sol_standard: str):
        '''
        Get an array of subject id in the sol_standard HEK event
            Inputs
            ------
            sol_standard : str
                Date and time of a read in event
                Solar Object Locator of HEK database
                format: 'SOLyyyy-mm-ddThh:mm:ssL000C000'
            Outputs
            ------
            np.array
                Array with all subjects id's in the HEK event
        '''
        try:
            return np.asarray([x['subjectId'] for x in self.data if x['data']['#sol_standard'] == sol_standard])
        except BaseException:
            print('ERROR: sol_standard ' + str(sol_standard) +
                  ' could not be read from ' + self.file_name)
            return np.asarray([])

    def get_subjectdata_by_solstandard(self, sol_standard: str):
        '''
        Get an array of metadata for the subjects in the sol_standard HEK event
            Inputs
            ------
            sol_standard : str
                Date and time of a read in event
                Solar Object Locator of HEK database
                format: 'SOLyyyy-mm-ddThh:mm:ssL000C000'
            Outputs
            ------
            np.array
                Array with dict metadata for all subjects in the HEK event
        '''
        try:
            return np.asarray([x['data'] for x in self.data if x['data']['#sol_standard'] == sol_standard])
        except BaseException:
            print('ERROR: sol_standard ' + str(sol_standard) +
                  ' could not be read from ' + self.file_name)
            return np.asarray([])

    def get_subjectkeyvalue_by_solstandard(self, sol_standard: str, key: str):
        '''
            Get an array of key values of the subjects in the sol_standard HEK event
            Inputs
            ------
            sol_standard : str
                Date and time of a read in event
                Solar Object Locator of HEK database
                format: 'SOLyyyy-mm-ddThh:mm:ssL000C000'
            key : str
                Dict key names
                keys {'#file_name_0','#file_name_14', '#sol_standard', '#width','#height',
                    '#naxis1', '#naxis2', '#cunit1', '#cunit2','#crval1','#crval2', '#cdelt1', '#cdelt2',
                    '#crpix1', '#crpix2', '#crota2', '#im_ll_x', '#im_ll_y','#im_ur_x', '#im_ur_y'}
            Outputs
            ------
            np.array
                Array with key value of the subjects in the HEK event
        '''
        try:
            if key == 'startDate' or key == 'endDate':
                return np.asarray([string_to_datetime(x['data'][key]) for x in self.data if x['data']['#sol_standard'] == sol_standard], dtype='datetime64')
            else:
                return np.asarray([x['data'][key] for x in self.data if x['data']['#sol_standard'] == sol_standard])
        except KeyError:
            print('ERROR: key ' + key + ' not found, please check your spelling')
        except BaseException:
            print('ERROR: sol_standard ' + str(sol_standard) +
                  ' could not be read from ' + self.file_name)
            return np.asarray([])

    def get_subjectid_by_dates(self, start_date: str, end_date: str):
        '''
            Get an array of subject id in a given timeframe
            Inputs
            ------
            start_date : str
                start of wanted time frame format 'YYYY-MM-dd'
            end_date : str
                end of wanted time frame format 'YYYY-MM-dd'

            Outputs
            ------
            np.array
                Array with all subjects id's occuring in given timeframe
        '''

        try:
            S, E = string_to_datetime(start_date), string_to_datetime(end_date)
            return np.asarray([x['subjectId'] for x in self.data if S < string_to_datetime(x['data']['startDate']) < E])
        except ValueError:
            print('ERROR: the start_date and end_date should be in format \'YYY-MM-dd\' or \'YYYY-MM-dd\' hh:mm:ss')
        except BaseException:
            print('ERROR: no data can be found between ' +
                  str(start_date) + str(end_date) + ' in ' + self.file_name)
            return np.asarray([])

    def get_subjectdata_by_id(self, subject: int):
        '''
        Get an array of metadata for the subject
            Inputs
            ------
            subject: int
                Zooniverse subject ID
            Outputs
            ------
            np.array
                Array with dict metadata for the subject
        '''
        try:
            response = np.asarray([x['data'] for x in self.data if x['subjectId'] == subject])
            if len(response) == 1:
                return response[0]
            else:
                print('ERROR: subjectId ' + str(subject) +
                      ' is occuring more than once in ' + self.file_name)
                return np.asarray([])
        except BaseException:
            print("ERROR: could not load data from file: " + self.file_name)

    def get_subjectkeyvalue_by_id(self, subject: int, key: str):
        '''
            Get an array of key values of a subject
            Inputs
            ------
            subject: int
                Zooniverse subject ID
            key : str
                Dict key names
                keys {'#file_name_0','#file_name_14', '#sol_standard', '#width','#height',
                    '#naxis1', '#naxis2', '#cunit1', '#cunit2','#crval1','#crval2', '#cdelt1', '#cdelt2',
                    '#crpix1', '#crpix2', '#crota2', '#im_ll_x', '#im_ll_y','#im_ur_x', '#im_ur_y'}
            Outputs
            ------
            value
                key value of the subject
        '''
        try:
            if key == 'startDate' or key == 'endDate':
                return np.asarray([string_to_datetime(x['data'][key]) for x in self.data if x['subjectId'] == subject], dtype='datetime64')[0]
            else:
                return np.asarray([x['data'][key] for x in self.data if x['subjectId'] == subject])[0]
        except KeyError:
            print('ERROR: key ' + key + ' not found, please check your spelling')
        except BaseException:
            print('ERROR: subjectId ' + str(subject) + ' could not be read from ' + self.file_name)
            return np.asarray([])

    def get_subjectkeyvalue_by_list(self, subjectidlist: np.array, key: str):
        '''
            Get an array of key values of the subjects in the sol_standard HEK event
            Inputs
            ------
            subjectidlist : np.array
                list with Zooniverse subject id's
            key : str
                Dict key names
                keys {'#file_name_0','#file_name_14', '#sol_standard', '#width','#height',
                    '#naxis1', '#naxis2', '#cunit1', '#cunit2','#crval1','#crval2', '#cdelt1', '#cdelt2',
                    '#crpix1', '#crpix2', '#crota2', '#im_ll_x', '#im_ll_y','#im_ur_x', '#im_ur_y'}
            Outputs
            ------
            np.array
                Array with key value of the subjects in the subjectidlist
        '''
        try:
            if key == 'startDate' or key == 'endDate':
                return np.asarray([[string_to_datetime(x['data'][key]) for x in self.data if x['subjectId'] == subjectId][0] for subjectId in subjectidlist], dtype='datetime64')
            else:
                return np.asarray([[x['data'][key] for x in self.data if x['subjectId'] == subjectId][0] for subjectId in subjectidlist])
        except KeyError:
            print('ERROR: key ' + key + ' not found, please check your spelling')
        except BaseException:
            print('ERROR: subjectId ' + str(subjectidlist) +
                  ' could not be read from ' + self.file_name)
            return np.asarray([])


def string_to_datetime(datetimestring: str):
    '''
        Construct a date from a string in ISO 8601 format.
        Inputs
        ------
        datetimestring: str
            datetime str value to be converted from json format: 'YYYY-MM-dd hh:mm:ss'
            to datetime(YYYY,MM,dd,hh,mm,ss)
        Outputs
        ------
        datetime(YYYY,MM,dd,hh,mm,ss)
    '''
    return datetime.datetime.fromisoformat(datetimestring)
