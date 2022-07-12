import numpy as np 
import matplotlib.pyplot as plt
import datetime
from astropy.io import ascii
from panoptes_client import Panoptes, Subject, Workflow
from dateutil.parser import parse
import csv

data_T0 = ascii.read('JetorNot/question_reducer_jet_or_not.csv',format='csv')

# change the columns to make it easier to work with
data_T0.rename_column('data.no-there-are-no-jets-in-this-image-sequence', 'data.no')
data_T0.rename_column('data.yes-there-is-at-least-one-jet-in-this-image-sequence', 'data.yes')

# fill in missing data
data_T0['data.yes'].fill_value = 0
data_T0['data.no'].fill_value = 0
data_T0 = data_T0.filled()

data_T3T4 = ascii.read('BoxTheJets/reductions/question_reducer_box_the_jets_merged.csv', format='csv')

# fill in missing data
data_T3T4['data.yes'].fill_value = 0
data_T3T4['data.no'].fill_value = 0
data_T3T4 = data_T3T4.filled()

data_T3=data_T3T4[data_T3T4['task']=='T3'] #Only use the first Yes/No Jet question, T4 is the question is there a second jet

data_combined=data_T0.copy()
for i in range(len(data_T3)): #Print which subjects were not in first workflow but are in the second (should not happen)
    s=data_T3['subject_id'][i]
    if s not in data_T0['subject_id']:
        print('The following subject was not in the first workflow and thus will be ignored')
        print(data_T3[data_T3['subject_id']==s] ) #A test example, only one vote
    else: 
        j=np.argwhere(data_T0['subject_id']==s)[0][0]
        #print(s,j)
        data_combined['data.yes'][j]=data_T0['data.yes'][j]+data_T3['data.yes'][i]
        data_combined['data.no'][j]=data_T0['data.no'][j]+data_T3['data.no'][i]
        data_combined['task'][j]='Tc'

def Agr_mask(data):
    '''
    Find the agreement between the volunteers for the Jet or Not question
    '''
    num_votes = np.asarray(data['data.no']) + np.asarray(data['data.yes'])
    counts    = np.asarray(np.dstack([data['data.yes'], data['data.no']])[0])
    most_likely  = np.argmax(counts, axis=1)

    value_yes = most_likely==0
    value_no  = most_likely==1

    agreement = np.zeros_like(num_votes)

    agreement[value_yes] = counts[value_yes,0]/(num_votes[value_yes])
    agreement[value_no]  = counts[value_no,1]/(num_votes[value_no])

    agreement = np.asarray(agreement) #The order of agreement is in the order of the subjects
    
    jet_mask = most_likely==0
    jet_subjects = data['subject_id'][jet_mask]
    non_jet_mask = most_likely==1
    non_jet_subjects = data['subject_id'][non_jet_mask]
    Ans=np.zeros(len(data['subject_id']),dtype=str)
    Ans[jet_mask]='y'
    Ans[non_jet_mask]='n'
    
    return agreement,jet_mask,non_jet_mask,Ans

##Agreement functie aanroepen
agreement_T0,jet_mask_T0,non_jet_mask_T0,Ans_T0=Agr_mask(data_T0)
agreement_T3,jet_mask_T3,non_jet_mask_T3,Ans_T3=Agr_mask(data_T3)
agreement_Tc,jet_mask_Tc,non_jet_mask_Tc,Ans_Tc=Agr_mask(data_combined)

##Obstime,sol,filenames
def obs_time(data):
    obs_time = np.array([])
    SOL = np.array([])
    filenames= np.array([])

    for i, subject in enumerate(data['subject_id']):
        print("\r [%-40s] %d/%d"%(int(i/len(data['subject_id'])*40)*'=', i+1, len(data['subject_id'])), end='')
        panoptes_subject = Subject(subject)

        # get the obsdate from the filename (format ssw_cutout_YYYYMMDD_HHMMSS_*.png). we'll strip out the 
        # extras and just get the date in ISO format and parse it into a datetime array
        filenames= np.append(filenames,panoptes_subject.metadata['#file_name_0'])
        obs_datestring = panoptes_subject.metadata['#file_name_0'].split('_')[2:4]
        obs_time=np.append(obs_time,parse(f'{obs_datestring[0]}T{obs_datestring[1]}'))
        SOL=np.append(SOL,panoptes_subject.metadata['#sol_standard'])
        
    return obs_time,SOL,filenames

obs_time_T0,SOL_T0,filenames_T0=obs_time(data_T0)
obs_time_T3,SOL_T3,filenames_T3=obs_time(data_T3)
obs_time_Tc,SOL_Tc,filenames_Tc=np.copy(obs_time_T0),np.copy(SOL_T0),np.copy(filenames_T0)


##make the flag and counting
def count_jets(A,t):
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

##make csv list
def plot_bar_SOL(data, SOL, obs_time, Ans, agreement, jet_mask, non_jet_mask, task,filenames):
    open('BoxTheJets/SOL/subjects_{}.csv'.format(task),'w')
    start_i=0
    f = open('BoxTheJets/SOL/SOL_{}_stats.csv'.format(task), 'w') #PUT in Box the jet
    writer = csv.writer(f)
    writer.writerow(['#SOL-event','subjects in event','filename0','times subjects','number of jet clusters','start event', 'end event','Flagged jets'])
    SOL_small=np.array([])
    Num=np.array([])
    #total 122 figures SOL events
    while start_i<len(SOL):
        I=SOL==SOL[start_i]
        C,N,start,end=count_jets(Ans[I],obs_time[I])
        SOL_small=np.append(SOL_small,SOL[start_i])
        Num=np.append(Num,C)
        S=np.array(data['subject_id'][I],dtype=str)
        S2=' '.join(S)
        T=obs_time[I]
        T2=' '.join(np.asarray(T,dtype=str))
        F=filenames[I]
        F2=' '.join(np.asarray(F,dtype=str))
        A=Ans[I]
        Ag=agreement[I]
        ## Make list with all subjects
        with open('BoxTheJets/SOL/subjects_{}.csv'.format(task),'a') as csvfile:
            np.savetxt(csvfile, np.column_stack((S,T,A,Ag,F)), delimiter=",",newline='\n',fmt='%s')
        ##
        writer.writerow([SOL[start_i],S2,F2,T2,C,start,end,N])
        start_i=np.max(np.where(I==True))+1
    f.close()

plot_bar_SOL(data_T0, SOL_T0, obs_time_T0, Ans_T0, agreement_T0,jet_mask_T0,non_jet_mask_T0, 'T0',filenames_T0)
plot_bar_SOL(data_T3, SOL_T3, obs_time_T3, Ans_T3, agreement_T3,jet_mask_T3,non_jet_mask_T3, 'T3',filenames_T3)
plot_bar_SOL(data_combined, SOL_Tc, obs_time_Tc, Ans_Tc, agreement_Tc,jet_mask_Tc,non_jet_mask_Tc, 'Tc',filenames_Tc)
print('')
