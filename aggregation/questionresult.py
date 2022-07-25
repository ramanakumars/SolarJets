import numpy as np
import matplotlib.pyplot as plt
import datetime
from panoptes_client import Panoptes, Subject, Workflow
from dateutil.parser import parse
import csv

class QuestionResult:
    '''
        Data class to handle all binary question answers given by the volunteers
    '''
    def __init__(self, data):
        '''
            Inputs
            ------
            data : csv-file
                    Rows contain the question_reducer_jet_or_not.csv
                    with columns data.yes and data.no for each subject 

        '''
        data['data.yes'].fill_value = 0
        data['data.no'].fill_value = 0
        self.data = data.filled()
        
    def Agr_mask(self):
        '''
        Find the agreement between the volunteers for the Jet or Not question
        '''
        num_votes = np.asarray(self.data['data.no']) + np.asarray(self.data['data.yes'])
        counts    = np.asarray(np.dstack([self.data['data.yes'], self.data['data.no']])[0])
        most_likely  = np.argmax(counts, axis=1)

        value_yes = most_likely==0
        value_no  = most_likely==1

        agreement = np.zeros_like(num_votes)

        agreement[value_yes] = counts[value_yes,0]/(num_votes[value_yes])
        agreement[value_no]  = counts[value_no,1]/(num_votes[value_no])

        agreement = np.asarray(agreement) #The order of agreement is in the order of the subjects

        jet_mask = most_likely==0
        jet_subjects = self.data['subject_id'][jet_mask]
        non_jet_mask = most_likely==1
        non_jet_subjects = self.data['subject_id'][non_jet_mask]
        Ans=np.zeros(len(self.data['subject_id']),dtype=str)
        Ans[jet_mask]='y'
        Ans[non_jet_mask]='n'

        return agreement,jet_mask,non_jet_mask,Ans
        
        
    def obs_time(self):
        '''
        Connect to the Zooniverse to get the observation starting time, SOL event, 
        filenames of 1st image of the subject and the end_time of the subjects.
        '''
        
        obs_time = np.array([])
        SOL = np.array([])
        filenames= np.array([])
        end_time = np.array([])

        for i, subject in enumerate(self.data['subject_id']):
            print("\r [%-40s] %d/%d"%(int(i/len(self.data['subject_id'])*40)*'=', i+1, len(self.data['subject_id'])), end='')
            panoptes_subject = Subject(subject)

            # get the obsdate from the filename (format ssw_cutout_YYYYMMDD_HHMMSS_*.png). we'll strip out the 
            # extras and just get the date in ISO format and parse it into a datetime array
            filenames= np.append(filenames,panoptes_subject.metadata['#file_name_0'])
            obs_datestring = panoptes_subject.metadata['#file_name_0'].split('_')[2:4]
            obs_time=np.append(obs_time,parse(f'{obs_datestring[0]}T{obs_datestring[1]}'))
            end_datestring = panoptes_subject.metadata['#file_name_14'].split('_')[2:4]
            end_time=np.append(end_time,parse(f'{end_datestring[0]}T{end_datestring[1]}'))
            SOL=np.append(SOL,panoptes_subject.metadata['#sol_standard'])

        return obs_time,SOL,filenames,end_time
        
        
        
        
    def count_jets(self,A,t):
        '''
        Get properties of the SOL event by looping over the various subjects
        '''
    
        L=''
        start=''
        end=''
        tel=0
        prev='n'
        switch=0
        for i in range(len(A)):  
            f1='0'
            f2='0'
            a=A[i]
            if prev != a:
                if a=='y':
                    tel+=1 #
                    start+=str(t[i])+' '#
                    s=i #start index
                else:
                    end+=str(t[i])+' '#
                    if np.abs(t[i]-t[s])<datetime.timedelta(minutes = 6):
                        f1='1'
                switch+=1
                prev = a
            else:
                switch=0

            if switch>1: #Flagging jets where we have ynyn or nyny so jets inclosed in 
                f2='1'
                switch=0
            if f1+f2!='00':
                L=L+' '+f1+f2+' '+str(t[i])

        return tel,L,start,end
        
        
        
        
    def csv_SOL(self, SOL, obs_time, Ans, agreement, jet_mask, non_jet_mask, task,filenames, end_time):
        '''
        Make the subject and SOL csv-files
        '''
    
        open('subjects_{}.csv'.format(task),'w')
        start_i=0
        f = open('SOL_{}_stats.csv'.format(task), 'w') #PUT in Box the jet
        writer = csv.writer(f)
        writer.writerow(['#SOL-event','subjects in event','filename0','times subjects','number of jet clusters','start event', 'end event','Flagged jets'])
        SOL_small=np.array([])
        Num=np.array([])
        #total 122 figures SOL events
        while start_i<len(SOL):
            I=SOL==SOL[start_i]
            C,N,start,end=self.count_jets(Ans[I],obs_time[I])
            SOL_small=np.append(SOL_small,SOL[start_i])
            Num=np.append(Num,C)
            S=np.array(self.data['subject_id'][I],dtype=str)
            S2=' '.join(S)
            T=obs_time[I]
            T2=' '.join(np.asarray(T,dtype=str))
            F=filenames[I]
            F2=' '.join(np.asarray(F,dtype=str))
            E=end_time[I]
            A=Ans[I]
            Ag=agreement[I]
            sol_event=SOL[I]
            ## Make list with all subjects
            with open('subjects_{}.csv'.format(task),'a') as csvfile:
                np.savetxt(csvfile, np.column_stack((S,T,E,A,Ag,F,sol_event)), delimiter=",",newline='\n',fmt='%s')
            ##
            writer.writerow([SOL[start_i],S2,F2,T2,C,start,end,N])
            start_i=np.max(np.where(I==True))+1
        f.close()