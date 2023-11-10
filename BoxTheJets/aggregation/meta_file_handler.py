import json
import datetime
import numpy as np
from dateutil.parser import parse


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
    except Exception:
        print(f'dateTime could not be extracted from {fileName}')
        raise


def create_subjectinfo(subject, subjectsdata, keysToImport=None):
    '''
        Takes a Zooniverse subjectsdata and makes a reduced meta dictionary containing chosen keys only or the full metadata
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
            dictionary of the wanted metadata for the given subject
    '''

    try:
        allData = json.loads(subjectsdata['metadata'][subjectsdata['subject_id'] == subject][-1])
        if keysToImport is None:
            keysToImport = allData.keys()
        wantedDict = {key: allData[key] for key in keysToImport}
        wantedDict['#fits_names'] = json.loads(wantedDict['#fits_names'].replace("'", '"'))
        wantedDict['startDate'] = str(convert_fileName_to_datetime(wantedDict['#fits_names'][0]))
        wantedDict['endDate'] = str(convert_fileName_to_datetime(wantedDict['#fits_names'][-1]))
        wantedDict['#width'] = float(wantedDict['#width'])
        wantedDict['#height'] = float(wantedDict['#height'])
        del allData  # does this help memory management?
        return wantedDict
    except Exception:
        print('')
        print(f"Not all metadata available for subject {subject} atempting to gather minimal information")
        raise


class SubjectMetadata:
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
            with open(file_name, 'r') as infile:
                data = json.load(infile)
        except FileNotFoundError:
            print(f'{file_name} was not found')
            return
        except BaseException:
            print('This file could not be read out as a json, please check the format')
            return

        self.file_name = file_name
        self.data = np.asarray(data)
        self.subject_ids = np.asarray([x['subject_id'] for x in data])
        self.SOL_standard = np.asarray([x['data']['#sol_standard'] for x in data])

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
            return np.unique(self.subject_ids[np.where(self.SOL_standard == sol_standard)[0]])
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
        subjects = self.get_subjectid_by_solstandard(sol_standard)

        return np.asarray([self.get_subjectdata_by_id(subject) for subject in subjects])

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
        subjects = self.get_subjectid_by_solstandard(sol_standard)

        return np.asarray([self.get_subjectkeyvalue_by_id(subject, key) for subject in subjects])

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
            return np.asarray([x['subject_id'] for x in self.data if S < string_to_datetime(x['data']['startDate']) < E])
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
        output = {}
        for key in self.data[0]['data'].keys():
            output[key] = self.get_subjectkeyvalue_by_id(subject, key)

        return output

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
        row = self.data[self.subject_ids == subject][0]
        if key == 'startDate' or key == 'endDate':
            return string_to_datetime(row['data'][key])
        else:
            return row['data'][key]

        
        
    def get_subjectid_by_JetCluster(self, list_jet_clusters, shj_id: str):
        '''
        Get an array of subject id in the Jet_Cluster identified by shj_id
            Inputs
            ------
            list_jet_clusters : list
                List of Jet_cluster objects to search
                can be read from a Jet_cluster.json file

            shj_id : str 
                The Jet_Cluster identifier to get the information for

            Outputs
            ------
            np.array
                Array with all subjects id's in the Jet_cluster with shj_id
        '''
        try:
            Jet = [list_jet_clusters[i] for i in range(len(list_jet_clusters)) if list_jet_clusters[i].ID == shj_id][0]
            return np.asarray([Jet.jets[i].subject for i in range(len(Jet.jets))])
        except:
            print('ERROR: shj_identifier ' + str(shj_id) +
                  ' could not be read from the list_jet_clusters input')
            return np.asarray([])

    def get_subjectdata_by_JetCluster(self, list_jet_clusters, shj_id: str):
        '''
        Get an array of metadata for the subjects in the Jet_Cluster identified by shj_id
            Inputs
            ------
            list_jet_clusters : list
                List of Jet_cluster objects to search
                can be read from a Jet_cluster.json file

            shj_id : str 
                The Jet_Cluster identifier to get the information for

            Outputs
            ------
            np.array
                Array with dict metadata for all subjects in the Jet_cluster with shj_id
        '''
        subjects_list = self.get_subjectid_by_JetCluster(list_jet_clusters, shj_id)
        return np.asarray([self.get_subjectdata_by_id(subject) for subject in subjects_list])
    

    def get_subjectkeyvalue_by_JetCluster(self, list_jet_clusters, shj_id: str, key: str):
        '''
            Get an array of key values of the subjects in the Jet_Cluster identified by shj_id
            Inputs
            ------
            list_jet_clusters : list
                List of Jet_cluster objects to search
                can be read from a Jet_cluster.json file

            shj_id : str 
                The Jet_Cluster identifier to get the information for
            key : str 
                Dict key names 
                keys {'#file_name_0','#file_name_14', '#sol_standard', '#width','#height',
                    '#naxis1', '#naxis2', '#cunit1', '#cunit2','#crval1','#crval2', '#cdelt1', '#cdelt2', 
                    '#crpix1', '#crpix2', '#crota2', '#im_ll_x', '#im_ll_y','#im_ur_x', '#im_ur_y'}
            Outputs
            ------
            np.array
                Array with key value of the subjects in the Jet_cluster with shj_id
        '''
        subjects_list = self.get_subjectid_by_JetCluster(list_jet_clusters, shj_id)
        return self.get_subjectkeyvalue_by_list(subjects_list, key)
    



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
    return np.asarray(datetime.datetime.fromisoformat(datetimestring), dtype=np.datetime64)
