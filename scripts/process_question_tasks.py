import sys
import numpy as np
import json
import tqdm
sys.path.append('.')

try:
    from aggregation.questionresult import QuestionResult
except ModuleNotFoundError:
    raise

# make the combined reducer for the "is there a jet" question
jet_or_not_question = QuestionResult('reductions/question_reducer_jet_or_not.csv', 'solar_jet_hunter_metadata.json', 'T1')
box_the_jet_question = QuestionResult('reductions/question_reducer_box_the_jets.csv', 'solar_jet_hunter_metadata.json', 'T1')

data_combined = box_the_jet_question.data[0:0].copy()
for subject_id in tqdm.tqdm(np.unique(box_the_jet_question.subjects), desc='Combining reducers', ascii=True):
    jet_or_not_data = jet_or_not_question.get_data_by_id(subject_id)
    box_the_jet_data = box_the_jet_question.get_data_by_id(subject_id)

    new_row = box_the_jet_data.copy()
    new_row['data.yes'] = box_the_jet_data['data.yes'] + jet_or_not_data['data.yes']
    new_row['data.no'] = box_the_jet_data['data.no'] + jet_or_not_data['data.no']
    new_row['task'] = 'Tc'

    data_combined.add_row(np.asarray(new_row)[0])

data_combined.write('reductions/question_reducer_combined.csv', overwrite=True)

# then process the multi jet question
multijet_question = QuestionResult('reductions/question_reducer_box_the_jets.csv', 'solar_jet_hunter_metadata.json', 'T5')
multijet_question.data.write('reductions/multi_jet.csv', overwrite=True)
