import json
import datetime
import numpy as np

class MetaFile:
    '''
        Data class to read out the meta data for each of the subjects given in Zooniverse
    '''
    def __init__(self, file_name : str):
        '''
            Inputs
            ------
            file_name : meta data json file 
                    Contains for each subject a set of meta data e.g. start_date, file_name

        '''
        try:
            data = json.load(open(file_name))
        except FileNotFoundError: 
            print(f'{file_name} was not found')
        except:
            print('This file could not be read out as a json, please check the format')

        self.file_name = file_name
        self.data = data
        self.subjects = np.asarray([x['subjectId'] for x in data])
        self.SOL_unique = np.unique([x['data']['#sol_standard'] for x in data])

    def getSubjectIdBySolStandard(self,sol_standard: str):
        try:
            return np.asarray([x['subjectId'] for x in self.data if x['data']['#sol_standard'] == sol_standard])
        except:
            print('ERROR: sol_standard '+ str(sol_standard) +' could not be read from '+ self.file_name)
            return np.asarray([])

    def getSubjectdataBySolStandard(self, sol_standard: str):
        try:
            return np.asarray([x['data'] for x in self.data if x['data']['#sol_standard'] == sol_standard])
        except:
            print('ERROR: sol_standard '+ str(sol_standard) +' could not be read from '+ self.file_name)
            return np.asarray([])   

    def getSubjectByKeyBySolStandard(self,sol_standard: str, key: str):
        try:
            if key == 'startDate' or key == 'endDate':
                return np.asarray([stringToDateTime(x['data'][key]) for x in self.data if x['data']['#sol_standard'] == sol_standard])
            else:
                return np.asarray([x['data'][key] for x in self.data if x['data']['#sol_standard'] == sol_standard])
        except KeyError:
            print('ERROR: key '+ key +' not found, please check your spelling')
        except:
            print('ERROR: sol_standard '+ str(sol_standard) +' could not be read from '+ self.file_name)
            return np.asarray([])  
        
    def getSubjectByDates(self,start_date: str,end_date: str):
        try:
            S,E= stringToDateTime(start_date),stringToDateTime(end_date)
            return np.asarray([x['subjectId'] for x in self.data if S<stringToDateTime(x['data']['startDate'])<E])
        except ValueError:
            print('ERROR: the start_date and end_date should be in format \'YYY-MM-dd\' or \'YYYY-MM-dd\' hh:mm:ss')
        except:
            print('ERROR: no data can be found between '+ str(start_date)+ str(end_date) +' in '+ FILE_NAME)
            return np.asarray([])

    def getSubjectdatabyId(self,subjectId: int):
        try:
            response = np.asarray([x['data'] for x in self.data if x['subjectId']==subjectId])
            if len(response) == 1:
                return response[0]
            else:
                print('ERROR: subjectId '+ str(subjectId) +' is occuring more than once in '+ self.file_name)
                return np.asarray([])
        except:
            print("ERROR: could not load data from file: "+ self.file_name)

def stringToDateTime(dateTimeString: str):
    return datetime.datetime.fromisoformat(dateTimeString)