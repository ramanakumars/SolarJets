import numpy as np
import pandas as pd
import sys
sys.path.append('.')

try:
    from aggregation.workflow import Aggregator
    from aggregation.meta_file_handler import create_metadata_jsonfile
except ModuleNotFoundError:
    raise

aggregator = Aggregator('reductions/point_reducer_hdbscan_box_the_jets.csv', 'reductions/shape_reducer_dbscan_box_the_jets.csv')
subjects = aggregator.get_subjects()
metadatafile = pd.read_csv('../solar-jet-hunter-subjects.csv').to_dict(orient='list')
for key in metadatafile.keys():
    metadatafile[key] = np.asarray(metadatafile[key])

create_metadata_jsonfile('../Meta_data_subjects.json', subjects, metadatafile)
