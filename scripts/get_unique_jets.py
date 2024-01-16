import sys
import json
import tqdm
sys.path.append('.')

try:
    from aggregation.workflow import Aggregator
    from aggregation.io import NpEncoder
except ModuleNotFoundError:
    raise

aggregator = Aggregator('reductions/temporal_point_reducer_hdbscan_box_the_jets.csv', 'reductions/shape_reducer_dbscan_box_the_jets.csv')
aggregator.load_subject_data('solar_jet_hunter_metadata.json')
jets = []
for subject in tqdm.tqdm(aggregator.subjects, desc='Finding unique jets', ascii=True):
    jets.extend(aggregator.filter_classifications(subject))

with open('reductions/jets.json', 'w') as outfile:
    json.dump([jet.to_dict() for jet in jets], outfile, cls=NpEncoder, indent=4)
