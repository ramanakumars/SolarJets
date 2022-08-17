import os 
#os.chdir('JetOrNot/')
#print(os.getcwd())
from aggregation import QuestionResult 
from astropy.io import ascii
import numpy as np

#Make the csv files 
data_T0 = ascii.read('JetOrNot/question_reducer_jet_or_not.csv',format='csv')

#os.chdir('..')

# change the columns to make it easier to work with
# This is only needed for the first workflow the later ones have the shorter name already
data_T0.rename_column('data.no-there-are-no-jets-in-this-image-sequence', 'data.no')
data_T0.rename_column('data.yes-there-is-at-least-one-jet-in-this-image-sequence', 'data.yes')

# fill in missing data
data_T0['data.yes'].fill_value = 0
data_T0['data.no'].fill_value = 0
data_T0 = data_T0.filled()

data_T3T4 = ascii.read('BoxTheJets/reductions/question_reducer_box_the_jets.csv', format='csv')

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


#initiate the filled data set as a class
data_Tc=QuestionResult(data_combined)

#Calculate the agreement 
agreement_Tc,jet_mask_Tc,non_jet_mask_Tc,Ans_Tc=data_Tc.Agr_mask()

#get the observation time, SOL event, filenames and end time of the subjects 
obs_time_Tc,SOL_Tc,filenames_Tc,end_time_Tc=data_Tc.obs_time()

#Compile the csv files SOL_Tc_stats.csv and subjects_Tc.csv
data_Tc.csv_SOL(SOL_Tc, obs_time_Tc, Ans_Tc, agreement_Tc,jet_mask_Tc,non_jet_mask_Tc, 'Tc',filenames_Tc,end_time_Tc)

print(' ')
print('The csv files SOL_Tc_stats.csv and subjects_Tc.csv are created')