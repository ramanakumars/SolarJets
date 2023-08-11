import os
from JetOrNot.aggregation import QuestionResult
from astropy.io import ascii
import numpy as np

# Make the csv files
data_T0 = QuestionResult('JetOrNot/reductions/question_reducer_jet_or_not.csv')

data_T3T4 = QuestionResult('BoxTheJets/reductions/question_reducer_box_the_jets.csv')


# Only use the first Yes/No Jet question, T4 is the question is there a second jet
data_T3_data = data_T3T4.data[data_T3T4.data['task'] == 'T3']

data_combined = data_T0.data.copy()
# Print which subjects were not in first workflow but are in the second (should not happen)
for i in range(len(data_T3_data)):
    s = data_T3_data['subject_id'][i]
    if s not in data_T0.data['subject_id']:
        print('The following subject was not in the first workflow and thus will be ignored')
        print(data_T3_data[data_T3_data['subject_id'] == s])  # A test example, only one vote
    else:
        j = np.argwhere(data_T0.data['subject_id'] == s)[0][0]
        # print(s,j)
        data_combined['data.yes'][j] = data_T0.data['data.yes'][j]+data_T3_data['data.yes'][i]
        data_combined['data.no'][j] = data_T0.data['data.no'][j]+data_T3_data['data.no'][i]
        data_combined['task'][j] = 'Tc'

ascii.write(data_combined, 'question_reducer_combined_workflows.csv', format='csv', overwrite=True)

# initiate the filled data set as a class
data_Tc = QuestionResult('question_reducer_combined_workflows.csv')

print(' ')
print('The reduction file question_reducer_combined_workflows.csv was created')
