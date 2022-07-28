from aggregation import QuestionResult 
from astropy.io import ascii

#Make the csv files 
data_T3T4_file = ascii.read('reductions/question_reducer_box_the_jets_merged.csv', format='csv')

#initiate the filled data set as a class
data_T3_file=data_T3T4_file[data_T3T4_file['task']=='T3']
data_T3=QuestionResult(data_T3_file)

#Calculate the agreement 
agreement_T3,jet_mask_T3,non_jet_mask_T3,Ans_T3=data_T3.Agr_mask()

#get the observation time, SOL event, filenames and end time of the subjects 
obs_time_T3,SOL_T3,filenames_T3,end_time_T3=data_T3.obs_time()

#Compile the csv files SOL_T0_stats.csv and subjects_T0.csv
data_T3.csv_SOL(SOL_T3, obs_time_T3, Ans_T3, agreement_T3,jet_mask_T3,non_jet_mask_T3, 'T3',filenames_T3,end_time_T3)

print(' ')
print('The csv files SOL_T3_stats.csv and subjects_T3.csv are created')