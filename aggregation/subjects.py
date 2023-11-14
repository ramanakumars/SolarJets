import numpy as np
from astropy.io import ascii
import json


class SubjectLoader:
    '''
        Data loader for the Zooniverse subject metadata. This pulls from the
        subject export CSV file from Zooniverse' project builder lab.
    '''
    def __init__(self, subjects_file):
        self.subject_data = ascii.read(subjects_file, format='csv')

        self.subject_ids = np.asarray(self.subject_data['subject_ids'][:])

    def get_meta(self, subject_id):
        '''
            Retrieve the metadata for a given subject.

            Input
            -----
            subject_id : int
                The Zooniverse subject ID for a given subject

            Outputs
            -------
            metadata : dict
                Dictionary containing the metadata uploaded to Zooniverse
        '''
        subject_subset = np.where(self.subject_ids == subject_id)[0]

        if len(subject_subset) < 1:
            raise ValueError(f"No subject associated with {subject_id}")

        return json.loads(subject_subset[0]['metadata'])
