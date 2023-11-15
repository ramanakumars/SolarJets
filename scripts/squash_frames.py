import numpy as np
from astropy.io import ascii
import json


def merge_tasks(data, task1='T1', task2='T3'):
    # fixing FixedWidth errors
    for col in data.itercols():
        data.replace_column(col.name, col.astype('object'))

    # get the col names for each variable in frame0 in T1
    colnames = data.colnames
    col0 = sorted([i for i in colnames if f'frame0.{task1}' in i])

    data_merged = data.copy()

    for k, col0k in enumerate(col0):

        # we can modify the frame0 tag to frame[n]
        col2 = col0k.replace(task1, task2)
        data_merged[col2].fill_value = 'N/A'

        # find the rows where there is data
        mask = np.where(np.asarray(data_merged[col2][:].filled()) != 'N/A')[0]

        for row in mask:
            classification_id = data_merged['classification_id'][row]
            row_col1 = np.where((data_merged['classification_id'][:] == classification_id) & (data_merged['task'][:] == task1))[0][0]
            try:
                dataT1 = json.loads(data_merged[col0k][row_col1])
            except (ValueError,TypeError):
                dataT1 = []

            dataT5 = json.loads(data_merged[col2][row])

            # combine the T5 info with T1
            outdata = []
            outdata.extend(dataT1)
            outdata.extend(dataT5)

            # move those rows to frame0
            data_merged[col0[k]][row_col1] = str(outdata)
            # print(row, outdata)

            # and delete the other row
            data_merged[col2][row] = ''

    return data_merged


# point extractor
points_file = 'extracts/shape_extractor_temporalPoint_box_the_jets.csv'
points_data = ascii.read(points_file, delimiter=',')
data_merged_points = merge_tasks(points_data, 'T0', 'T3')
ascii.write(data_merged_points, points_file.replace('.csv', '_merged.csv'), overwrite=True, delimiter=',')

# shape extractor
box_file = 'extracts/shape_extractor_temporalRotateRectangle_box_the_jets.csv'
box_data = ascii.read(box_file, delimiter=',')
data_merged_box = merge_tasks(box_data, 'T0', 'T3')
ascii.write(data_merged_box, box_file.replace('.csv', '_merged.csv'), overwrite=True, delimiter=',')
