from astropy.io import ascii
from multiprocessing import Pool
import argparse
import numpy as np
from collections import defaultdict
import tqdm
from ast import literal_eval
from panoptes_aggregation.reducers.shape_reducer_dbscan import shape_reducer_dbscan
from panoptes_aggregation.reducers.point_reducer_hdbscan import point_reducer_hdbscan
import signal
import json
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _do_reduction(box_data_sub, point_data_sub, subject):
    ext_data = []

    for row in box_data_sub:
        # format this in the dictionary structure that the reducer expects
        exti = {'frame0': {}}

        for key in ['x', 'y', 'width', 'height', 'angle', 'frame']: 
            # load in the extractor data to a list
            exti['frame0'][f'T1_tool2_{key}'] =\
                literal_eval(row[f'data.frame0.T1_tool2_{key}'])

        ext_data.append(exti)
     
    # do the reduction
    box_reduction = shape_reducer_dbscan(ext_data, metric_type='IoU',
                               shape='rotateRectangle', user_id=False, 
                               allow_single_cluster=True, min_samples=2,
                                min_cluster_size=2, eps=0.5)

    if 'frame0' not in box_reduction.keys():
        print(ext_data)
        print(box_reduction)

    box_out = {}
    box_out['clusters'] = []
    labels = np.asarray(box_reduction['frame0'][f'T1_tool2_cluster_labels'])

    # check if clusters were found
    if 'T1_tool2_clusters_x' in box_reduction['frame0']:
        # if they were, then we load each cluster individually
        nclusters = len(box_reduction['frame0']['T1_tool2_clusters_x'])

        box_out['clusters'] = defaultdict(list)

        # load in individual keys for the box properties
        for key in ['x', 'y', 'width', 'height', 'angle']: 
            for cluster in range(nclusters):
                box_out['clusters'][key].append(box_reduction['frame0'][f'T1_tool2_clusters_{key}'][cluster])
            
    
    # also save the extracts
    box_out['extracts'] = []
    ext_new = defaultdict(list)
    for key in ['x', 'y', 'width', 'height', 'angle','frame']: 
        key_data = []
        for exti in ext_data:
            key_data.extend(exti['frame0'][f'T1_tool2_{key}'])
        
        ext_new[key] = key_data
    # and their corresponding cluster properties 
    # (i.e. which cluster they belong to)
    ext_new['cluster_labels'] = labels.tolist()
        
    # revert to a normal dict structure rather than
    # the defaultdict
    box_out['extracts'] = dict(ext_new)
    
    # do the same for the points
    ext_data = []

    for row in point_data_sub:
        # format this in the dictionary structure that the reducer expects
        exti = {'frame0': {}}

        for key in ['x', 'y', 'frame']: 
            for tool in ['tool0', 'tool1']:
            # load in the extractor data to a list
                exti['frame0'][f'T1_{tool}_{key}'] =\
                    literal_eval(row[f'data.frame0.T1_{tool}_{key}'])

        ext_data.append(exti)
     
    # do the reduction
    point_reduction = point_reducer_hdbscan(ext_data, user_id=False, 
                               allow_single_cluster=True, min_samples=2,
                                min_cluster_size=2)
    points_out = {}
    points_out['start'] = {}
    points_out['end'] = {}

    out_key = {'tool0': 'start', 'tool1': 'end'}

    for tool in ['tool0', 'tool1']:
        out_keyi = out_key[tool]
        points_out[out_keyi]['clusters'] = []
        labels = np.asarray(point_reduction['frame0'][f'T1_{tool}_cluster_labels'])
        probs = np.asarray(point_reduction['frame0'][f'T1_{tool}_cluster_probabilities'])

        # check if clusters were found
        if f'T1_{tool}_clusters_x' in point_reduction['frame0']:
            # if they were, then we load each cluster individually
            nclusters = len(point_reduction['frame0'][f'T1_{tool}_clusters_x'])

            points_out[out_keyi]['clusters'] = defaultdict(list)
            for cluster in range(nclusters):
                # load in individual keys for the box properties
                for key in ['x', 'y']: 
                    points_out[out_keyi]\
                        ['clusters'][key].append(\
                            point_reduction['frame0'][f'T1_{tool}_clusters_{key}'][cluster])

        # also save the extracts
        points_out[out_keyi]['extracts'] = []
        ext_new = defaultdict(list)
        for key in ['x', 'y', 'frame']: 
            key_data = []
            for exti in ext_data:
                key_data.extend(exti['frame0'][f'T1_{tool}_{key}'])
            
            ext_new[key] = key_data
        # and their corresponding cluster properties 
        # (i.e. which cluster they belong to)
        ext_new['cluster_labels'] = labels.tolist()
        ext_new['cluster_probabilities'] = probs.tolist()
        
        # revert to a normal dict structure rather than
        # the defaultdict
        points_out[out_keyi]['extracts'] = dict(ext_new)

    # combine all these data sources into one big dict
    out_row = {}
    out_row['box'] = box_out
    out_row['start'] = points_out['start']
    out_row['end'] = points_out['end']
    out_row['subject_id'] = subject

    out_data.append(out_row)

    return out_data


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reduce Solar Jet Hunter data')
    parser.add_argument('-c', '--num_procs', metavar='n', type=int,
                        help='Number of processors to use', default=1)
    args = parser.parse_args()
    num_procs = args.num_procs

    # read in the points extract data
    point_data = ascii.read('extracts/point_extractor_by_frame'
                            '_box_the_jets_scaled_squashed_merged.csv')

    # read in the box extracts data
    box_data = ascii.read('extracts/shape_extractor_rotateRectangle'
                          '_box_the_jets_scaled_squashed_merged.csv')

    # find the list of subjects from both point and box data
    subjects = set(np.unique(box_data['subject_id'])).\
        intersection(set(np.unique(point_data['subject_id'])))

    # build the reduction export list by looping over the subjects
    out_data = []

    with Pool(processes=num_procs, initializer=initializer) as pool:

        # loop through each subject and get the corresponding
        # rows from the point and box extractors
        inpargs = []
        for subject in tqdm.tqdm(subjects, desc='Loading subjects', ascii=True):
            box_data_sub = box_data[box_data['subject_id'] == subject]
            point_data_sub = point_data[point_data['subject_id'] == subject]

            # find the rows where there is data
            # here we check whether the frame info is filled
            data_frame = np.asarray(box_data_sub['data.frame0.T1_tool2_frame'].filled())
            mask = np.where(data_frame != 'N/A')[0]
            box_rows = box_data_sub[mask]

            data_frame0 = np.asarray(point_data_sub['data.frame0.T1_tool0_frame'].filled())
            data_frame1 = np.asarray(point_data_sub['data.frame0.T1_tool1_frame'].filled())
            mask = np.where((data_frame0 != 'N/A')&(data_frame1 != 'N/A'))[0]
            point_rows = point_data_sub[mask]

            if (len(box_rows) > 0)&(len(point_rows) > 0):
                inpargs.append([box_rows, point_rows, subject])

        try:
            # run this through the multiprocessing pipeline
            r = pool.starmap_async(_do_reduction, inpargs, chunksize=10)

            pool.close()
        
            tasks = pool._cache[r._job]
            ninpt = len(inpargs)

            # print a progress bar
            with tqdm.tqdm(total=ninpt, desc='Doing reduction', ascii=True) as pbar:
                while tasks._number_left > 0:
                    pbar.n = max([ninpt - tasks._number_left * tasks._chunksize, 0])
                    pbar.refresh()

                    time.sleep(0.1)
        except Exception as e:
            print(e)
            pool.terminate()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        finally:
            pool.join()


        results = r.get()

        # save out each row to a list
        for jj, out_row in enumerate(results):
            out_data.append(out_row)

    # save the list to a JSON
    with open('reductions/reductions.json', 'w') as outfile:
        json.dump(out_data, outfile, cls=NpEncoder)
