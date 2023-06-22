import numpy as np
from panoptes_client import Subject
import tqdm
import json


class QuestionResult:
    '''
        Data class to handle all binary question answers given by the volunteers
    '''

    def __init__(self, reduction_file, subject_loader=None):
        '''
            Inputs
            ------
            reduction_data : str
                Path to the CSV file (question_reducer_jet_or_not.csv)
                with columns data.yes and data.no for each subject
            subject_loader : SubjectLoader
                An instance of the SubjectLoader object which contains
                the subject metadata database from Zooniverse
        '''
        data = ascii.read(reduction_file, format='csv')
        data['data.yes'].fill_value = 0
        data['data.no'].fill_value = 0
        self.data = data.filled()

        self.subject_ids = np.asarray(data['subject_id'][:])

        self.subject_loader = subject_loader

        # populate the local data structures
        self.get_agreement()
        self.get_obs_time()

    def get_agreement(self):
        '''
            Find the agreement between the volunteers for the Jet or Not question
        '''
        num_votes = np.asarray(self.data['data.no']) + np.asarray(self.data['data.yes'])
        counts = np.dstack([self.data['data.yes'], self.data['data.no']])[0]

        # set the agreement score and consensus value for each subject
        self.most_likely = np.argmax(counts, axis=1)
        self.agreement = np.max(counts, axis=1) / num_votes

    def get_SOL(self):
        '''
            Pull the Zooniverse subject metadata and build a list of SOL event IDs
            for all the subjects loaded
        '''
        SOL = []

        for i, subject in enumerate(tqdm.tqdm(self.data['subject_id'])):
            if self.subject_loader is not None:
                # get this from the local subject loader
                SOL.append(self.subject_loader.get_meta(subject)['#sol_standard'])
            else:
                # if there is no subject loader, pull from panoptes
                panoptes_subject = Subject(subject)
                SOL.append(panoptes_subject.metadata['#sol_standard'])

        self.SOL = np.asarray(SOL)

    def export(self):
        '''
            Exports the subject info and corresponding SOL data and agreement values
            into a JSON output
        '''

        SOL_unique, SOL_inds = np.unique(self.SOL)

        SOL_data = []

        for i, (SOL, inds) in enumerate(tqdm.tqdm(zip(SOL_unique, SOL_inds), total=len(SOL_unique))):
            # fetch the corresponding data
            subjects = self.subject_ids[inds]
            agreements = self.agreement[inds]

            SOL_data.append({
                'SOL': SOL,
                'subject_ids': subjects,
                'agreements': agreements,
            })

        with open('SOL_data.json', 'w') as outfile:
            json.dump(SOL_data, outfile)
