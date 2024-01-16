import numpy as np
import pandas as pd
import sys
import tqdm
import json
sys.path.append('.')

try:
    from astropy.io import ascii
    from aggregation.meta_file_handler import create_subjectinfo
    from aggregation.io import NpEncoder
except ModuleNotFoundError:
    raise

data = ascii.read('reductions/question_reducer_jet_or_not.csv', format='csv')
subjects = np.unique(data['subject_id'])
subject_metadata = pd.read_csv('../solar-jet-hunter-subjects.csv').to_dict(orient='list')
for key in subject_metadata.keys():
    subject_metadata[key] = np.asarray(subject_metadata[key])

filename = 'solar_jet_hunter_metadata.json'

# Select keys we want to write to json file
keysToImport = [
    '#fits_names',  # First image filename in Zooniverse subject
    '#width',  # Width of the image in pixel
    '#height',  # Height of the image in pixel
    '#naxis1',  # Pixels along axis 1
    '#naxis2',  # Pixels along axis 2
    '#cunit1',  # Units of the coordinate increments along naxis1 e.g. arcsec
    '#cunit2',  # Units of the coordinate increments along naxis2 e.g. arcsec
    '#crval1',  # Coordinate value at reference point on naxis1
    '#crval2',  # Coordinate value at reference point on naxis2
    '#cdelt1',  # Spatial scale of pixels for naxis1, i.e. coordinate increment at reference point
    '#cdelt2',  # Spatial scale of pixels for naxis2, i.e. coordinate increment at reference point
    '#crpix1',  # Pixel coordinate at reference point naxis1
    '#crpix2',  # Pixel coordinate at reference point naxis2
    '#crota2',  # Rotation of the horizontal and vertical axes in degrees
    '#im_ll_x',  # Vertical distance in pixels between bottom left corner and start solar image
    '#im_ll_y',  # Horizontal distance in pixels between bottom left corner and start solar image
    '#im_ur_x',  # Vertical distance in pixels between bottom left corner and end solar image
    '#im_ur_y',  # Horizontal distance in pixels between bottom left corner and end solar image,
    '#sol_standard'  # SOL event that this subject corresponds to
]

subject_data = []
for i, subject in enumerate(tqdm.tqdm(subjects, ascii=True, desc='Writing subjects to JSON')):
    subjectDict = {}
    subjectDict['subject_id'] = int(subject)
    subjectDict['data'] = create_subjectinfo(subject, subject_metadata, keysToImport)
    subject_data.append(subjectDict)

with open(filename, 'w') as outfile:
    json.dump(subject_data, outfile, cls=NpEncoder, indent=4)

print(f"Wrote subject information to {filename}")
