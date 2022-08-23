import sys
try:
    sys.path.append('.')  # assumes you're running this code from JetOrNot/
    from aggregation import QuestionResult
except ModuleNotFoundError as e:
    raise e
from astropy.io import ascii

# Make the csv files
data_T0_file = ascii.read(
    'reductions/question_reducer_jet_or_not.csv', format='csv')

# change the columns to make it easier to work with
# This is only needed for the first workflow the later ones have the shorter name already
data_T0_file.rename_column(
    'data.no-there-are-no-jets-in-this-image-sequence', 'data.no')
data_T0_file.rename_column(
    'data.yes-there-is-at-least-one-jet-in-this-image-sequence', 'data.yes')

# initiate the filled data set as a class
data_T0 = QuestionResult(data_T0_file)

# Calculate the agreement
agreement_T0, jet_mask_T0, non_jet_mask_T0, Ans_T0 = data_T0.Agr_mask()

# get the observation time, SOL event, filenames and end time of the subjects
obs_time_T0, SOL_T0, filenames_T0, end_time_T0 = data_T0.obs_time()

# Compile the csv files SOL_T0_stats.csv and subjects_T0.csv
data_T0.csv_SOL(SOL_T0, obs_time_T0, Ans_T0, agreement_T0,
                jet_mask_T0, non_jet_mask_T0, 'T0', filenames_T0, end_time_T0)

print(' ')
print('The csv files SOL_T0_stats.csv and subjects_T0.csv are created')
