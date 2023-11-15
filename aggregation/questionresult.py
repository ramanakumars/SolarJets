import numpy as np
import datetime
from astropy.table import Table
from astropy.io import ascii
from .meta_file_handler import SubjectMetadata


class QuestionResult:
    '''
        Data class to handle all binary question answers given by the volunteers
    '''

    def __init__(self, reducer_csv: str, subject_metadata_json: str, task: str):
        '''
            Inputs
            ------
            reducer_csv : str
                    Rows contain the question_reducer_jet_or_not.csv
                    with columns data.yes and data.no for each subject
            subject_metadata_json : str
                    Path to the generated subject metadata JSON file
            task : str
                    The task corresponding to the reduction
        '''
        data = ascii.read(reducer_csv, format='csv')
        data = data[data['task'] == task]

        if 'data.yes' not in data.colnames:
            data = ascii.read(reducer_csv, format='csv')
            data = data[data['task'] == task]
            data.rename_column('data.no-there-is-no-solar-jet-in-this-video', 'data.no')
            data.rename_column('data.yes-there-is-at-least-one-jet-in-this-video', 'data.yes')

        data['data.yes'].fill_value = 0
        data['data.no'].fill_value = 0

        if 'data.None' in data.colnames:
            data['data.None'].fill_value = 0

        # add the None entry (blank) to no
        self.data: Table = data.filled()
        if 'data.None' in data.colnames:
            self.data['data.no'][:] = np.asarray(self.data['data.no'] + self.data['data.None'])
            del self.data['data.None']

        self.subject_metadata = SubjectMetadata(subject_metadata_json)
        self.subjects = np.asarray(data['subject_id'])

        assert len(np.unique(self.subjects)) == len(self.subjects), "There are duplicate subject data!"
        self.get_agreement()

    def get_data_by_id(self, subject_id: int) -> Table:
        subject_data = self.data[self.subjects == subject_id]
        return subject_data

    def get_data_by_idlist(self, list_subjects: list[int]) -> Table:
        index = [np.where(self.subjects == x)[0][0] for x in list_subjects]
        subjects_data = self.data[index]
        return subjects_data

    def get_agreement(self):
        '''
        Find the agreement between the volunteers for the Jet or Not question
        '''
        num_votes = np.asarray(self.data['data.no'][:]) + np.asarray(self.data['data.yes'][:])
        counts = np.asarray(np.dstack([self.data['data.yes'], self.data['data.no']])[0])
        most_likely = np.argmax(counts, axis=1)

        value_yes = most_likely == 0
        value_no = most_likely == 1

        agreement = np.zeros_like(num_votes)

        agreement[value_yes] = counts[value_yes, 0] / (num_votes[value_yes])
        agreement[value_no] = counts[value_no, 1] / (num_votes[value_no])

        self.agreement = np.asarray(agreement)  # The order of agreement is in the order of the subjects

        jet_mask = most_likely == 0
        non_jet_mask = most_likely == 1
        self.is_jet = np.zeros(len(self.data), dtype=bool)
        self.is_jet[jet_mask] = True
        self.is_jet[non_jet_mask] = False

    def count_jets(self, A, t):
        '''
        Get properties of the SOL event by looping over the various subjects
        Inputs
        ------
        A : np.array
            list of answers of the subjects
        t : np.array
            starting times of the subjects

        Output
        ------
            tel : int
                count of how many jet events (sequential jet subjects)
            L : str
                string of flagging per jet event seperated by ' '
            start : str
                string of starting times jet event seperated by ' '
            end : str
                string of end times jet event seperated by ' '
        '''

        L = ''
        start = ''
        end = ''
        tel = 0
        prev = 'n'
        switch = 0
        for i in range(len(A)):
            f1 = '0'
            f2 = '0'
            a = A[i]
            if prev != a:
                if a == 'y':
                    tel += 1
                    start += str(t[i]) + ' '
                    s = i  # start index
                else:
                    end += str(t[i]) + ' '
                    if np.abs(t[i] - t[s]) < datetime.timedelta(minutes=6):
                        f1 = '1'
                switch += 1
                prev = a
            else:
                switch = 0

            if switch > 1:  # Flagging jets where we have ynyn or nyny so jets inclosed in
                f2 = '1'
                switch = 0
            if f1 + f2 != '00':
                L = L + ' ' + f1 + f2 + ' ' + str(t[i])

        return tel, L, start, end

    def csv_SOL(self, SOL, obs_time, Ans, agreement, jet_mask, non_jet_mask, task, filenames, end_time):
        '''
        Make the subject and SOL csv-files

        Input
        -----
        SOL, obs_time, Ans, agreement, jet_mask, non_jet_mask, task,filenames, end_time : str format
        Properties of the subjects to be saved in a csv file
        '''

        open('subjects_{}.csv'.format(task), 'w')
        start_i = 0
        f = open('SOL_{}_stats.csv'.format(task), 'w')  # PUT in Box the jet
        writer = csv.writer(f)
        writer.writerow(['#SOL-event', 'subjects in event', 'filename0', 'times subjects', 'number of jet clusters', 'start event', 'end event', 'Flagged jets'])
        SOL_small = np.array([])
        Num = np.array([])
        # total 122 figures SOL events
        while start_i < len(SOL):
            I = SOL == SOL[start_i]
            C, N, start, end = self.count_jets(Ans[I], obs_time[I])
            SOL_small = np.append(SOL_small, SOL[start_i])
            Num = np.append(Num, C)
            S = np.array(self.data['subject_id'][I], dtype=str)
            S2 = ' '.join(S)
            T = obs_time[I]
            T2 = ' '.join(np.asarray(T, dtype=str))
            F = filenames[I]
            F2 = ' '.join(np.asarray(F, dtype=str))
            E = end_time[I]
            A = Ans[I]
            Ag = agreement[I]
            sol_event = SOL[I]
            # Make list with all subjects
            with open('subjects_{}.csv'.format(task), 'a') as csvfile:
                np.savetxt(csvfile, np.column_stack((S, T, E, A, Ag, F, sol_event)), delimiter=",", newline='\n', fmt='%s')
            ##
            writer.writerow([SOL[start_i], S2, F2, T2, C, start, end, N])
            start_i = np.max(np.where(I == True)) + 1
        f.close()
