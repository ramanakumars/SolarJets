from astropy.io import ascii
import sys
try:
    sys.path.append('.')  # assumes you're running this code from JetOrNot/
    from aggregation import QuestionResult
    from aggregation import create_metadata_jsonfile
except ModuleNotFoundError as e:
    raise e

data_T0 = QuestionResult('reductions/question_reducer_jet_or_not.csv')
aggregation_subjects = data_T0.subjects

Zooniverse_subjectsdata = ascii.read(
    'solar-jet-hunter-subjects.csv', format='csv', include_names=['subject_id', 'metadata'])

create_metadata_jsonfile('../Meta_data_subjects.json',
                         aggregation_subjects, Zooniverse_subjectsdata)
