import numpy as np
from astropy.io import ascii
import ast

## point extractor
file = 'point_extractor_by_frame_box_the_jets.csv'

data = ascii.read(file, delimiter=',')

# fixing FixedWidth errors
for col in data.itercols():
    data.replace_column(col.name, col.astype('object'))

colnames = data.colnames
# get the col names for each variable in frame0
col0 = sorted([i for i in colnames if 'frame0' in i])

for k, col0k in enumerate(col0):
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

# get the col names for each variable in frame0 in T1
# col0 = sorted([i for i in colnames if 'frame0.T1' in i])
# for k, col0k in enumerate(col0):
    
#     # we can modify the frame0 tag to frame[n]
#     colT5 = col0k.replace('T1', 'T5')
#     data[colT5].fill_value = 'N/A'

#     print(col0k, colT5)

#     # find the rows where there is data
#     mask = np.where(np.asarray(data[colT5][:].filled()) != 'N/A')[0]

#     for row in mask:
#         classification_id = data['classification_id'][row]
#         row_T1 = np.where((data['classification_id'][:]==classification_id)&(data['task'][:]=='T1'))[0][0]
#         try:
#             dataT1 = ast.literal_eval(data[col0k][row_T1])
#         except ValueError:
#             dataT1 = []

#         dataT5 = ast.literal_eval(data[colT5][row])

#         # combine the T5 info with T1
#         outdata = []
#         outdata.extend(dataT1)
#         outdata.extend(dataT5)

#         # move those rows to frame0
#         data[col0[k]][row_T1] = str(outdata)
#         # print(row, outdata)

#         # and delete the other row
#         data[colT5][row] = ''

ascii.write(data, file.replace('.csv', '_squashed.csv'), overwrite=True, delimiter=',')

## shape extractor
file = 'shape_extractor_rotateRectangle_box_the_jets.csv'

data = ascii.read(file, delimiter=',')

# fixing FixedWidth errors
for col in data.itercols():
    data.replace_column(col.name, col.astype('object'))

colnames = data.colnames

col0 = sorted([i for i in colnames if 'frame0' in i])

for k, col0k in enumerate(col0):
    for j in range(1,15):
        coltype = col0k.replace('frame0', 'frame%d'%j)
        data[coltype].fill_value = 'None'
        mask = np.asarray(data[coltype][:].filled()) == 'None'
        data[col0[k]][~mask] = data[coltype][~mask]
        data[coltype][~mask] = ''

# get the col names for each variable in frame0 in T1
# col0 = sorted([i for i in colnames if 'frame0.T1' in i])
# for k, col0k in enumerate(col0):
    
#     # we can modify the frame0 tag to frame[n]
#     colT5 = col0k.replace('T1', 'T5')
#     data[colT5].fill_value = 'N/A'

#     print(col0k, colT5)

#     # find the rows where there is data
#     mask = np.where(np.asarray(data[colT5][:].filled()) != 'N/A')[0]

#     for row in mask:
#         classification_id = data['classification_id'][row]
#         row_T1 = np.where((data['classification_id'][:]==classification_id)&(data['task'][:]=='T1'))[0][0]
#         try:
#             dataT1 = ast.literal_eval(data[col0k][row_T1])
#         except ValueError:
#             dataT1 = []

#         dataT5 = ast.literal_eval(data[colT5][row])

#         # combine the T5 info with T1
#         outdata = []
#         outdata.extend(dataT1)
#         outdata.extend(dataT5)

#         # move those rows to the T1 array
#         data[col0[k]][row_T1] = str(outdata)
#         # print(row, outdata)

#         # and delete the other row
#         data[colT5][row] = ''

ascii.write(data, file.replace('.csv', '_squashed.csv'), overwrite=True, delimiter=',')