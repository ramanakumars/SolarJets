import numpy as np
from astropy.io import ascii
from astropy.table import MaskedColumn
import ast
import re

## point extractor
file = 'extracts/point_extractor_by_frame_box_the_jets_scaled.csv'

data = ascii.read(file, delimiter=',')

# fixing FixedWidth errors
for col in data.itercols():
    data.replace_column(col.name, col.astype('object'))

colnames = data.colnames
# get the col names for each variable in frame0
col0 = sorted([i for i in colnames if 'frame0' in i])

data['data.frame0.T1_tool0_frame'] = MaskedColumn([None]*len(data), fill_value='None',
                                                  mask=[True]*len(data), dtype='U20')
data['data.frame0.T1_tool1_frame'] = MaskedColumn([None]*len(data), fill_value='None',
                                                  mask=[True]*len(data), dtype='U20')
data['data.frame0.T5_tool0_frame'] = MaskedColumn([None]*len(data), fill_value='None',
                                                  mask=[True]*len(data), dtype='U20')
data['data.frame0.T5_tool1_frame'] = MaskedColumn([None]*len(data), fill_value='None',
                                                  mask=[True]*len(data), dtype='U20')

for k, col0k in enumerate(col0):
    colframe = col0k.replace('_x', '_frame').replace('_y', '_frame')
    
    # add the frame info for classifications on frame0
    data[col0k].fill_value = 'None'
    mask = np.asarray(data[col0k][:].filled()) == 'None'
    data[colframe][~mask] = str([0])

    for j in range(1,15):
        # we can modify the frame0 tag to frame[n]
        coltype = col0k.replace('frame0', 'frame%d'%j)
        data[coltype].fill_value = 'N/A'

        # find the rows where there is data
        mask = np.asarray(data[coltype][:].filled()) != 'N/A'

        # move those rows to frame0
        data[col0[k]][mask] = data[coltype][mask]

        # and delete the other row
        data[coltype][mask] = ''

        data[colframe][mask] = str([j])

# get the col names for each variable in frame0 in T1
colnames = data.colnames
col0 = sorted([i for i in colnames if 'frame0.T1' in i])

data_merged = data.copy()

for k, col0k in enumerate(col0):
    
    # we can modify the frame0 tag to frame[n]
    colT5 = col0k.replace('T1', 'T5')
    data_merged[colT5].fill_value = 'N/A'

    # find the rows where there is data
    mask = np.where(np.asarray(data_merged[colT5][:].filled()) != 'N/A')[0]

    for row in mask:
        classification_id = data_merged['classification_id'][row]
        row_T1 = np.where((data_merged['classification_id'][:]==classification_id)&(data_merged['task'][:]=='T1'))[0][0]
        try:
            dataT1 = ast.literal_eval(data_merged[col0k][row_T1])
        except ValueError:
            dataT1 = []

        dataT5 = ast.literal_eval(data_merged[colT5][row])

        # combine the T5 info with T1
        outdata = []
        outdata.extend(dataT1)
        outdata.extend(dataT5)

        # move those rows to frame0
        data_merged[col0[k]][row_T1] = str(outdata)
        # print(row, outdata)

        # and delete the other row
        data_merged[colT5][row] = ''

ascii.write(data, file.replace('.csv', '_squashed.csv'), overwrite=True, delimiter=',')
ascii.write(data_merged, file.replace('.csv', '_squashed_merged.csv'), overwrite=True, delimiter=',')

## shape extractor
file = 'extracts/shape_extractor_rotateRectangle_box_the_jets_scaled.csv'

data = ascii.read(file, delimiter=',')

# fixing FixedWidth errors
for col in data.itercols():
    data.replace_column(col.name, col.astype('object'))

colnames = data.colnames

col0 = sorted([i for i in colnames if 'frame0' in i])

data['data.frame0.T1_tool2_frame'] = MaskedColumn([None]*len(data), fill_value='None',
                                                  mask=[True]*len(data), dtype='U20')
data['data.frame0.T5_tool2_frame'] = MaskedColumn([None]*len(data), fill_value='None',
                                                  mask=[True]*len(data), dtype='U20')

for k, col0k in enumerate(col0):
    print(col0k)
    colframe = re.sub('_([a-z]+)$', '_frame', col0k)

    # add the frame info for classifications on frame0
    data[col0k].fill_value = 'None'
    mask = np.asarray(data[col0k][:].filled()) == 'None'
    data[colframe][~mask] = str([0])

    for j in range(1,15):
        coltype = col0k.replace('frame0', 'frame%d'%j)
        data[coltype].fill_value = 'None'
        mask = np.asarray(data[coltype][:].filled()) == 'None'
        data[col0[k]][~mask] = data[coltype][~mask]
        data[coltype][~mask] = ''

        # we are replacing this multiple times
        # as the loop goes over the different variables
        # (x, y, w, h and a). This should be fine, but
        # possibly can be fixed in the future
        data[colframe][~mask] = str([j])

# get the col names for each variable in frame0 in T1
colnames = data.colnames
data_merged = data.copy()
col0 = sorted([i for i in colnames if 'frame0.T1' in i])
for k, col0k in enumerate(col0):
    
    # we can modify the frame0 tag to frame[n]
    colT5 = col0k.replace('T1', 'T5')
    data_merged[colT5].fill_value = 'N/A'

    # find the rows where there is data
    mask = np.where(np.asarray(data_merged[colT5][:].filled()) != 'N/A')[0]

    for row in mask:
        classification_id = data_merged['classification_id'][row]
        row_T1 = np.where((data_merged['classification_id'][:]==classification_id)&(data_merged['task'][:]=='T1'))[0][0]
        try:
            dataT1 = ast.literal_eval(data_merged[col0k][row_T1])
        except ValueError:
            dataT1 = []

        dataT5 = ast.literal_eval(data_merged[colT5][row])

        # combine the T5 info with T1
        outdata = []
        outdata.extend(dataT1)
        outdata.extend(dataT5)

        # move those rows to the T1 array
        data_merged[col0k][row_T1] = str(outdata)
        # print(row, outdata)

        # and delete the other row
        data_merged[colT5][row] = ''

ascii.write(data, file.replace('.csv', '_squashed.csv'), overwrite=True, delimiter=',')
ascii.write(data_merged, file.replace('.csv', '_squashed_merged.csv'), overwrite=True, delimiter=',')
