import sys
import numpy as np
import json
import tqdm
import datetime
sys.path.append('.')

try:
    from aggregation.jet import Jet
    from aggregation.jet_cluster import JetCluster
    from aggregation.meta_file_handler import SubjectMetadata
    from aggregation.shape_utils import get_point_distance, get_box_iou
    from aggregation.io import NpEncoder
except ModuleNotFoundError:
    raise

eps = 2.  # distance epsilon (arbitrary normalized units)
time_eps = 10. * 60.  # two jets are in the same cluster if they are 10 minutes apart

metadata = SubjectMetadata('../solar_jet_hunter_subject_metadata.json')

with open('reductions/jets.json', 'r') as infile:
    jets = [Jet.from_dict(jet) for jet in json.load(infile)]


SOL_unique = np.unique(metadata.SOL_standard)
out_data = []

for sol in tqdm.tqdm(SOL_unique, desc='Clustering events', ascii=True):
    subjects = metadata.get_subjectid_by_solstandard(sol)
    start_times = metadata.get_subjectkeyvalue_by_solstandard(sol, 'startDate')
    end_times = metadata.get_subjectkeyvalue_by_solstandard(sol, 'endDate')

    # sort the jets by time
    sort_inds = np.argsort(start_times)

    start_times: list[float] = start_times[sort_inds].tolist()
    end_times: list[float] = end_times[sort_inds].tolist()
    subjects: list[int] = subjects[sort_inds].tolist()
    jets_subset = np.asarray([jet for jet in jets if jet.subject_id in subjects])

    if len(jets_subset) < 1:
        continue

    dts = [(end_time - start_time).total_seconds() for start_time, end_time in zip(start_times, end_times)]

    # get the time of the jet from the box
    inds = [subjects.index(jet.subject_id) for jet in jets_subset]
    jet_times = np.asarray([start_times[k] + datetime.timedelta(seconds=jet.box.displayTime * dts[k]) for jet, k in zip(jets_subset, inds)])

    box_metric = np.zeros((len(jets_subset), len(jets_subset)))
    time_metric = np.zeros((len(jets_subset), len(jets_subset)))
    point_metric = np.zeros((len(jets_subset), len(jets_subset)))

    for j, jetj in enumerate(jets_subset):
        for k, jetk in enumerate(jets_subset):
            if j == k:
                point_metric[k, j] = 0
                box_metric[k, j] = 0
                time_metric[k, j] = 0
            else:
                point_dist = get_point_distance(jetj.start, jetk.start)
                box_ious = get_box_iou(jetj.box, jetk.box)
                point_metric[k, j] = point_dist / np.mean([jetj.start.extract_dists, jetk.start.extract_dists])
                box_metric[k, j] = 1. - box_ious

                # we will limit to 2 frames (each frame is 5 min)
                time_metric[k, j] = np.abs((jet_times[j] - jet_times[k]).total_seconds())

    point_scale = np.nanpercentile(point_metric[np.isfinite(point_metric) & (point_metric > 0)], 90)
    if np.isnan(point_scale):
        point_scale = np.inf
    distance_metric = point_metric / point_scale + box_metric

    distance_metric[~np.isfinite(distance_metric)] = np.inf

    indices = np.arange(len(jets_subset))
    labels = -1. * np.ones(len(jets_subset))

    while len(indices) > 0:
        ind = indices[0]

        # find all the jets that fall within a distance
        # eps for this jet and those that are not
        # already clustered into a jet
        mask = (distance_metric[ind, :] < eps) & (labels == -1)

        # next make sure that there is a reachability in time
        # jets should be connected to each other to within 1-2 frames
        rem_inds = np.where(mask)[0]
        for j, indi in enumerate(rem_inds):
            # if this is the first index we don't
            # have an idea of past reachability
            if j == 0:
                continue

            # get the reachability in time
            time_disti = time_metric[indi, mask]
            # subset it up to the current jet
            # so we get only past reachability
            t0 = np.argmin(time_disti)
            time_disti = time_disti[:t0]

            # if the previous index was deleted
            # we can end up with empty lists
            # ignore these and assign them
            # to a different cluster
            if len(time_disti) == 0:
                mask[indi] = False
                continue

            # make sure this jet starts after the current jet has ended
            if jets_subset[indi].time_info['start'] < jets_subset[ind].time_info['end']:
                mask[indi] = False
                continue

            # find the smallest interval between this jet and any other
            # jet. then remove this if it more than eps frames away
            if time_disti[time_disti > 0.].min() > time_eps:
                mask[indi] = False

        # assign a new value to these
        labels[mask] = labels.max() + 1

        rem_inds = [np.where(indices == maski)[0][0]
                    for maski in np.where(mask)[0]]

        indices = np.delete(indices, rem_inds)

    # get the list of jets found
    njets = len(np.unique(labels[labels > -1]))

    assert njets > 0, "No jet clusters found!"

    jet_clusters = []
    for j in range(njets):
        mask_j = labels == j
        # subset the list of jets that correspond to this label
        jets_j: Jet = jets_subset[mask_j]
        times_j = jet_times[mask_j]

        clusteri = JetCluster(jets_j.tolist())

        jet_clusters.append(clusteri)

    out_data.append({
        'sol': sol,
        'events': [jet_cluster.to_dict() for jet_cluster in jet_clusters]
    })

with open('reductions/jet_cluster.json', 'w') as outfile:
    json.dump(out_data, outfile, cls=NpEncoder)
