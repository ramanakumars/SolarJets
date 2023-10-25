import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import tqdm
import json
from shapely.geometry import Polygon, Point
from .jet import Jet
from .shape_utils import BasePoint, Box, get_point_distance, get_box_iou
from .zoo_utils import get_subject_image


def get_point_data_from_rows(point_rows, tool):
    points_data = []
    subject_id = np.unique(point_rows['subject_id'][:])[0]

    for row in point_rows:
        try:
            cluster_x = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_clusters_x']))
            cluster_y = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_clusters_y']))
            cluster_time = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_clusters_displayTime']))
        except json.JSONDecodeError:
            continue

        row_x = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_points_x']))
        row_y = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_points_y']))
        row_time = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_points_displayTime']))
        row_labels = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_cluster_labels']))
        row_probs = np.asarray(json.loads(row[f'data.frame0.T0_toolIndex{tool}_cluster_probabilities']))

        for i, (x, y, time) in enumerate(zip(cluster_x, cluster_y, cluster_time)):
            point = BasePoint(x=x, y=y, displayTime=time, subject_id=subject_id)

            extracts_mask = row_labels == i
            extracts_x = row_x[extracts_mask]
            extracts_y = row_y[extracts_mask]
            extracts_time = row_time[extracts_mask]
            extracts_prob = row_probs[extracts_mask]

            extract_points = []
            for j, (x, y, time, prob) in enumerate(zip(extracts_x, extracts_y, extracts_time, extracts_prob)):
                extract_points.append(BasePoint(x=x, y=y, displayTime=time, probability=prob, subject_id=subject_id))

            point.extracts = extract_points
            points_data.append(point)
    return points_data


class Aggregator:
    '''
        Single data class to handle different aggregation requirements
    '''

    def __init__(self, points_file, box_file):
        '''
            Inputs
            ------
            points_file : str
                path to the reduced points (start/end) data
            box_file : str
                path to the reduced box data
        '''
        point_reductions = ascii.read(points_file, format='csv').filled()
        box_reductions = ascii.read(box_file, format='csv').filled()

        self.subjects = np.unique([*point_reductions['subject_id'], *box_reductions['subject_id']])

        self.start_points_data = []
        self.end_points_data = []
        self.box_data = []

        for i, subject_id in enumerate(tqdm.tqdm(self.subjects, desc='Getting data', ascii=True)):
            point_rows = point_reductions[(point_reductions['subject_id'] == subject_id) & (point_reductions['task'] == 'T0')]

            start_points = get_point_data_from_rows(point_rows, 0)
            self.start_points_data.extend(start_points)

            end_points = get_point_data_from_rows(point_rows, 1)
            self.end_points_data.extend(end_points)

            box_rows = box_reductions[(box_reductions['subject_id'] == subject_id) & (box_reductions['task'] == 'T0')]
            for row in box_rows:
                try:
                    cluster_box_x = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_clusters_x_center']))
                    cluster_box_y = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_clusters_y_center']))
                    cluster_box_w = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_clusters_width']))
                    cluster_box_h = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_clusters_height']))
                    cluster_box_a = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_clusters_angle']))
                    cluster_box_time = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_clusters_displayTime']))
                except json.JSONDecodeError:
                    continue

                row_box_x = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_temporalRotateRectangle_x_center']))
                row_box_y = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_temporalRotateRectangle_y_center']))
                row_box_w = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_temporalRotateRectangle_width']))
                row_box_h = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_temporalRotateRectangle_height']))
                row_box_a = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_temporalRotateRectangle_angle']))
                row_box_time = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_temporalRotateRectangle_displayTime']))
                row_box_labels = np.asarray(json.loads(row['data.frame0.T0_toolIndex2_cluster_labels']))

                for i, (x, y, w, h, a, time) in enumerate(zip(cluster_box_x, cluster_box_y, cluster_box_w, cluster_box_h, cluster_box_a, cluster_box_time)):
                    box = Box(xcenter=x, ycenter=y, width=w, height=h, angle=np.radians(a), displayTime=time, subject_id=subject_id)

                    extracts_mask = row_box_labels == i
                    extracts_x = row_box_x[extracts_mask]
                    extracts_y = row_box_y[extracts_mask]
                    extracts_w = row_box_w[extracts_mask]
                    extracts_h = row_box_h[extracts_mask]
                    extracts_a = np.radians(row_box_a[extracts_mask])
                    extracts_time = row_box_time[extracts_mask]

                    extract_boxes = []
                    for j, (x, y, w, h, a, time) in enumerate(zip(extracts_x, extracts_y, extracts_w, extracts_h, extracts_a, extracts_time)):
                        box_ext = Box(xcenter=x, ycenter=y, width=w, height=h, angle=a, displayTime=time, subject_id=subject_id)
                        extract_boxes.append(box_ext)

                    box.extracts = extract_boxes

                    self.box_data.append(box)

    def get_subjects(self):
        '''
            Return a list of known subjects in the reduction data

            Outputs
            -------
            subjects : numpy.ndarray
                Array of subject IDs on Zooniverse
        '''
        subjects = np.unique(np.concatenate([[e.subject_id for e in data] for data in [self.box_data, self.start_points_data, self.end_points_data]]))

        return np.unique(subjects)

    def get_points_data(self, subject):
        '''
            Get the points data and cluster, and associated probabilities and labels
            for a givens subject and task. s corresponds to the start and e corresponds to end

            Inputs
            ------
            subject : int
                Subject ID in zooniverse
            task : string
                Either 'T1' or 'T5' for the first jet or second jet

            Outputs
            -------
            data : dict
                Raw classification data for the x, y (xs, ys for the start and xe, ye for the end)
            clusters : dict
                Cluster shape (x, y) for start and end and probabilities and labels of the
                data points
        '''
        return (np.asarray([point for point in self.start_points_data if point.subject_id == subject]),
                np.asarray([point for point in self.end_points_data if point.subject_id == subject]))

    def get_box_data(self, subject):
        '''
            Get the box data and cluster shapes, and associated probabilities and labels
            for a givens subject and task

            Inputs
            ------
            subject : int
                Subject ID in zooniverse
            task : string
                Either 'T1' or 'T5' for the first jet or second jet

            Outputs
            -------
            data : dict
                Raw classification data for the x, y, width, height and angle (degrees)
            clusters : dict
                Cluster shape (x, y, width, height and angle) and probabilities and labels of the
                data points
        '''

        return np.asarray([box for box in self.box_data if box.subject_id == subject])

    def get_frame_time_base(self, subject, task='T1'):
        '''
            get the distribution of classifications by frame number for the base of the jet (both start and end)
            and determine the best frame for each, based on a cluster probability based metric

            Inputs
            ------
            subject : int
                Zooniverse subject ID
            task : string
                task for the Zooniverse workflow (T1 for first jet and T2 for second jet)

            Outputs
            -------
            frame_info : dict
                Dictionary with six keys:
                    - start_frames : list
                        List of coordinates for the base point for each frame in the subject
                    - start_score : list
                        Score for each frame (given by the extract probability for each cluster)
                    - start_best : int
                        Frame corresponding to the start point with the best score
                    - end_frames : list
                        List of coordinates for the base point for each frame in the subject
                        (for the base point for the last frame that the jet is visible in)
                    - end_score : list
                        Score for each frame (given by the extract probability for each cluster)
                    - end_best : int
                        Frame corresponding to the end point with the best score
        '''
        # get the points data and associated cluster
        points_data, points_clusters = self.get_points_data(subject, task)

        x0_i = points_data['x_start']
        y0_i = points_data['y_start']
        x1_i = points_data['x_end']
        y1_i = points_data['y_end']

        p0_i = points_clusters['prob_start']
        p1_i = points_clusters['prob_end']

        # get the extract data so that we can obtain frame_time info
        # you need to run load_extractor data first
        assert hasattr(
            self, 'point_extracts'), "Please load the extractor data using the load_extractor_data method"
        point_extractsi = self.point_extracts[:][(
            self.point_extracts['subject_id'] == subject) & (self.point_extracts['task'] == task)]

        # empty lists to hold the frame info
        start_frames = []
        end_frames = []

        # empty lists to hold the frame score
        start_weights = []
        end_weights = []

        for frame in range(15):
            # start of the jet is tool0
            frame_start_ext = point_extractsi[f'data.frame{frame}.{task}_tool0_x',
                                              f'data.frame{frame}.{task}_tool0_y'][:]
            for col in frame_start_ext.colnames:
                frame_start_ext[col].fill_value = 'None'

            # rename the columns for easier use
            frame_start_ext.rename_column(
                f'data.frame{frame}.{task}_tool0_x', f'frame{frame}_x')
            frame_start_ext.rename_column(
                f'data.frame{frame}.{task}_tool0_y', f'frame{frame}_y')

            # find the rows with data for this frame
            start_sub = frame_start_ext[:][frame_start_ext.filled()[
                f'frame{frame}_x'][:] != "None"]

            # parse the data from the string value and add it to the list
            points = []
            probs = []
            for i in range(len(start_sub)):
                xx = ast.literal_eval(start_sub[f'frame{frame}_x'][i])[0]
                yy = ast.literal_eval(start_sub[f'frame{frame}_y'][i])[0]
                points.append([xx, yy])

                # find the associated cluster probability based on the index of the points
                # in the cluster info
                ind = np.where((xx == x0_i) & (yy == y0_i))[0]
                probs.append(p0_i[ind])

            # convert to numpy array and append to the list
            points = np.asarray(points)
            start_frames.append(points)
            start_weights.append(probs)

            # end of the jet is tool1
            # do the same process for tool1
            frame_end_ext = point_extractsi[f'data.frame{frame}.{task}_tool1_x',
                                            f'data.frame{frame}.{task}_tool1_y'][:]
            for col in frame_end_ext.colnames:
                frame_end_ext[col].fill_value = 'None'

            frame_end_ext.rename_column(
                f'data.frame{frame}.{task}_tool1_x', f'frame{frame}_x')
            frame_end_ext.rename_column(
                f'data.frame{frame}.{task}_tool1_y', f'frame{frame}_y')

            end_sub = frame_end_ext[:][frame_end_ext.filled()[
                f'frame{frame}_x'][:] != "None"]

            points = []
            probs = []
            for i in range(len(end_sub)):
                xx = ast.literal_eval(end_sub[f'frame{frame}_x'][i])[0]
                yy = ast.literal_eval(end_sub[f'frame{frame}_y'][i])[0]

                points.append([xx, yy])

                # find the associated cluster probability based on the index of the points
                # in the cluster info
                ind = np.where((xx == x1_i) & (yy == y1_i))[0]
                probs.append(p1_i[ind])

            points = np.asarray(points)
            end_frames.append(points)
            end_weights.append(probs)

        # the score is the sum of the probabilities at each frame
        start_score = [np.sum(starti) for starti in start_weights]
        end_score = [np.sum(endi) for endi in end_weights]

        return {'start': start_frames, 'start_score': start_score, 'start_best': np.argmax(start_score),
                'end': end_frames, 'end_score': end_score, 'end_best': np.argmax(end_score)}

    def get_frame_time_box(self, subject, task='T1'):
        '''
            get the distribution of classifications by frame number for box. similar to _base above

            Inputs
            ------
            subject : int
                Zooniverse subject ID
            task : string
                task for the Zooniverse workflow (T1 for first jet and T2 for second jet)

            Outputs
            -------
            frame_info : dict
                Dictionary with two keys:
                  - box_frames : list
                        List of box parameters for each frame in the subject
                  - box_score : list
                        Score for each frame (given by the box sigma)
        '''
        # get the cluster and points information from the reduced data
        box_data, box_clusters = self.get_box_data(subject, task)
        x_i = box_data['x']
        y_i = box_data['y']
        w_i = box_data['w']
        h_i = box_data['h']
        a_i = box_data['a']

        p0_i = box_clusters['prob']

        # get the extract data so that we can obtain frame_time info
        # you need to run load_extractor data first
        assert hasattr(
            self, 'box_extracts'), "Please load the extractor data using the load_extractor_data method"
        box_extractsi = self.box_extracts[:][(
            self.box_extracts['subject_id'] == subject) & (self.box_extracts['task'] == task)]

        # empty lists to hold the frame info
        frames = []
        weights = []

        for frame in range(15):
            # rename the columns to make it easier to work with
            box_extractsi.rename_column(
                f'data.frame{frame}.{task}_tool2_x', f'frame{frame}_x')
            box_extractsi.rename_column(
                f'data.frame{frame}.{task}_tool2_y', f'frame{frame}_y')
            box_extractsi.rename_column(
                f'data.frame{frame}.{task}_tool2_width', f'frame{frame}_w')
            box_extractsi.rename_column(
                f'data.frame{frame}.{task}_tool2_height', f'frame{frame}_h')
            box_extractsi.rename_column(
                f'data.frame{frame}.{task}_tool2_angle', f'frame{frame}_a')

            # start of the jet is tool0
            frame_box = box_extractsi[f'frame{frame}_x', f'frame{frame}_y',
                                      f'frame{frame}_w', f'frame{frame}_h', f'frame{frame}_a'][:]
            for col in frame_box.colnames:
                frame_box[col].fill_value = 'None'

            # find the rows with data for this frame
            box_sub = frame_box[:][frame_box.filled(
            )[f'frame{frame}_x'][:] != "None"]

            # parse the data from the string value and add it to the list
            box = []
            probs = []
            for i in range(len(box_sub)):
                xx = ast.literal_eval(box_sub[f'frame{frame}_x'][i])[0]
                yy = ast.literal_eval(box_sub[f'frame{frame}_y'][i])[0]
                ww = ast.literal_eval(box_sub[f'frame{frame}_w'][i])[0]
                hh = ast.literal_eval(box_sub[f'frame{frame}_h'][i])[0]
                aa = ast.literal_eval(box_sub[f'frame{frame}_a'][i])[0]
                box.append([xx, yy, ww, hh, aa])

                # find the associated cluster probability based on the index of the box
                # in the cluster info
                ind = np.where((xx == x_i) & (yy == y_i) & (
                    ww == w_i) & (hh == h_i) & (aa == a_i))[0]
                probs.append(p0_i[ind])

            # convert to numpy array and append to the list
            box = np.asarray(box)
            frames.append(box)
            weights.append(probs)

        # the score is the sum of the probabilities at each frame
        score = [np.sum(weight) for weight in weights]

        return {'box_frames': frames, 'box_score': score}

    def find_unique_jets(self, subject, plot=False):
        '''
            Filters the box clusters for a subject from both T1 and T5
            and finds a list of unique jets that have minimal overlap

            Inputs
            ------
            subject : int
                The subject ID in Zooniverse

            plot : bool
                Flag for whether to plot the boxes or not

            Outputs
            --------
            clust_boxes : list
                List of `shapely.Polygon` objects which correspond to
                the cluster box
        '''
        # get the box data
        boxes = self.get_box_data(subject)

        # add all the boxes to a bucket as long as they are
        # valid clusters (iou > 0)
        temp_boxes = []

        for i, box in enumerate(boxes):
            if box.extract_IoU > 1.e-6:
                temp_boxes.append(box)

        temp_boxes = np.asarray(temp_boxes)

        # now loop over this bucket of polygons
        # and see how well they match with each other
        # we will move the "good" boxes to a new list
        # so we can keep track of progress based on how
        # many items are still in the queue
        clust_boxes = []

        # we are going to sort by the IoU of each box
        # so that the best boxes are processed first
        sort_mask = np.argsort([box.extract_IoU * len(box.extracts) for box in temp_boxes])
        temp_boxes = temp_boxes[sort_mask]

        while len(temp_boxes) > 0:
            nboxes = len(temp_boxes)

            # compare against the first box in the bucket
            # this will get removed at the end of this loop
            box0 = temp_boxes[0]

            # to compare iou of box0 with other boxes
            ious = np.ones(nboxes)

            # to see if box0 needs to be merged with another
            # box
            merge_mask = [False] * nboxes
            merge_mask[0] = True

            for j in range(1, nboxes):
                # find IoU for box0 vs boxj
                bj = temp_boxes[j]
                ious[j] = get_box_iou(box0, bj)

                # if the IoU is better than the worst IoU of the classifications
                # for either box, then we should merge these two
                # this metric could be changed to be more robust in the future
                if ious[j] > np.min([box0.extract_IoU, bj.extract_IoU, 0.1]):
                    merge_mask[j] = True

            # add the box with the best iou to the cluster list
            best = np.argmax(np.asarray([box.extract_IoU for box in temp_boxes])[merge_mask])
            clust_boxes.append(temp_boxes[merge_mask][best])

            if plot:
                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.imshow(get_subject_image(subject))
                # ax.plot(*box0.exterior.xy, '-', color='k')
                for j in range(1, nboxes):
                    bj = temp_boxes[j]
                    if merge_mask[j]:
                        ax.plot(*bj.get_box_edges(), 'k--', linewidth=0.5)
                    else:
                        ax.plot(*bj.get_box_edges(), 'k-', linewidth=0.5)
                for j in range(nboxes):
                    # calculate the bounding box for the cluster confidence
                    # and get the boxes edges
                    plus_sigma_box, minus_sigma_box = bj.get_plus_minus_sigma()

                    # create a fill between the - and + sigma boxes
                    x_p = plus_sigma_box[:, 0]
                    y_p = plus_sigma_box[:, 1]
                    x_m = minus_sigma_box[:, 0]
                    y_m = minus_sigma_box[:, 1]
                    ax.fill(np.append(
                        x_p, x_m[::-1]), np.append(y_p, y_m[::-1]), color='white', alpha=0.25)

                ax.axis('off')
                plt.tight_layout()
                plt.show()

            # and remove all the overlapping boxes from the list
            temp_boxes = np.delete(temp_boxes, merge_mask)

        return clust_boxes

    def find_unique_jet_points(self, subject, plot=False):
        '''
            Similar to `find_unique_jets` but for base points.
            Identifies base point clusters which fall within each others
            radius of confidence and merges them. Confidence radius is determined
            by the average distance between the extracts that belong in that cluster
            and the cluster center.

            Inputs
            ------
            subject : int
                Zooniverse subject ID for the image
            plot : bool [default=False]
                flag for whether to plot the intermediate steps

            Outputs
            -------
            start_points : `numpy.ndarray`
                Final set of base start points for the jet (post merger)
            end_points : `numpy.ndarray`
                Final set of base end points for the jet (post merger)

        '''
        start_points, end_points = self.get_points_data(subject)

        # create a copy of the start and end points
        temp_starts = start_points.copy()
        temp_ends = end_points.copy()
        start_extract_dists = np.asarray([point.extract_dists for point in temp_starts])
        end_extract_dists = np.asarray([point.extract_dists for point in temp_ends])

        # now loop over this bucket of start points
        # and see how well they match with each other
        # we will move the "good" points to a new list
        # so we can keep track of progress based on how
        # many items are still in the queue
        clust_starts = []
        while len(temp_starts) > 0:
            npoints = len(temp_starts)

            # compare against the first point in the bucket
            # this will get removed at the end of this loop
            point0 = temp_starts[0]

            # to compare distance of this 0th point with other points
            dists = np.zeros(npoints)

            # to see if start0 needs to be merged with another point
            merge_mask = [False] * npoints

            # we will always remove this first point from the queue
            merge_mask[0] = True

            for j in range(1, npoints):
                point = temp_starts[j]

                # find distance for the first and jth point
                dists[j] = get_point_distance(point0, point)

                # if the distance is better than the 1.5x the mean distance of
                # point that make up this cluster, then we should merge these two
                # this metric could be changed to be more robust in the future
                if dists[j] < 1.5 * np.max([start_extract_dists[0], start_extract_dists[j]]):
                    merge_mask[j] = True

            # add the point with the most compact intra-cluster distance to the cluster list
            clust_starts.append(temp_starts[merge_mask][np.argmin(start_extract_dists[merge_mask])])

            if plot:
                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.imshow(get_subject_image(subject))

                cir = Point((point0.x, point0.y)).buffer(1.5 * start_extract_dists[0])
                ax.plot(point0.x, point0.y, 'bx')
                ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                for j in range(1, npoints):
                    pointj = temp_starts[j]
                    ax.plot(pointj.x, pointj.y, 'kx')
                    cir = Point((pointj.x, pointj.y)).buffer(1.5 * start_extract_dists[j])
                    ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                ax.axis('off')
                plt.show()

            # and remove all the merged points from the list
            temp_starts = np.delete(temp_starts, merge_mask, axis=0)
            start_extract_dists = np.delete(start_extract_dists, merge_mask, axis=0)

        clust_ends = []
        while len(temp_ends) > 0:
            npoints = len(temp_ends)

            # compare against the first point in the bucket
            # this will get removed at the end of this loop
            point0 = temp_ends[0]

            # to compare distance of this 0th point with other points
            dists = np.zeros(npoints)

            # to see if end0 needs to be merged with another point
            merge_mask = [False] * npoints

            # we will always remove this first point from the queue
            merge_mask[0] = True

            for j in range(1, npoints):
                point = temp_ends[j]

                # find distance for the first and jth point
                dists[j] = get_point_distance(point0, point)

                # if the distance is better than the 1.5x the mean distance of
                # point that make up this cluster, then we should merge these two
                # this metric could be changed to be more robust in the future
                if dists[j] < 1.5 * np.max([end_extract_dists[0], end_extract_dists[j]]):
                    merge_mask[j] = True

            # add the point with the most compact intra-cluster distance to the cluster list
            clust_ends.append(temp_ends[merge_mask][np.argmin(end_extract_dists[merge_mask])])

            if plot:
                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.imshow(get_subject_image(subject))

                cir = Point((point0.x, point0.y)).buffer(1.5 * end_extract_dists[0])
                ax.plot(point0.x, point0.y, 'bx')
                ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                for j in range(1, npoints):
                    pointj = temp_ends[j]
                    ax.plot(pointj.x, pointj.y, 'kx')
                    cir = Point((pointj.x, pointj.y)).buffer(1.5 * end_extract_dists[j])
                    ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                ax.axis('off')
                plt.show()

            # and remove all the merged points from the list
            temp_ends = np.delete(temp_ends, merge_mask, axis=0)
            end_extract_dists = np.delete(end_extract_dists, merge_mask, axis=0)

        return np.asarray(clust_starts), np.asarray(clust_ends)

    def filter_classifications(self, subject, plot=False):
        '''
            Find a list of unique jets in the subject
            and segregate the classifications into each cluster
            based on IoU calculations

            Inputs
            ------
            subject : int
                The subject ID in Zooniverse

            Outputs
            --------
            jets : list
                List of `Jet` objects that are unique per subject (i.e.,
                they do not share overlap with other jets in the subject)
        '''
        # get the box data and clusters for the two tasks
        cluster_boxes = self.get_box_data(subject)
        cluster_start_points, cluster_end_points = self.get_points_data(subject)

        boxes = []
        for box in cluster_boxes:
            boxes.extend(box.extracts)

        start_points = []
        for point in cluster_start_points:
            start_points.extend(point.extracts)
        end_points = []
        for point in cluster_end_points:
            end_points.extend(point.extracts)

        # and the unique clusters
        unique_jets = self.find_unique_jets(subject)

        if len(unique_jets) < 1:
            return []

        unique_starts, unique_ends = self.find_unique_jet_points(subject)

        jets = []
        # for each jet box, find the best start/end points
        for i, boxi in enumerate(unique_jets):
            box_points = boxi.get_box_edges()[:4]

            dists = []
            # calculate the distance between each start point
            # and the box edges
            for j, point in enumerate(unique_starts):
                disti = np.median([np.linalg.norm(point.coordinate - pointi)
                                  for pointi in box_points])
                dists.append(disti)

            # the best start point is the one with the minimum
            # distance with all 4
            best_start = unique_starts[np.argmin(dists)]

            dists = []
            # do the same for the end points
            for j, point in enumerate(unique_ends):
                disti = np.median([np.linalg.norm(point.coordinate - pointi) * np.linalg.norm(point.coordinate - best_start.coordinate) for pointi in box_points])
                dists.append(disti)

            best_end = unique_ends[np.argmin(dists)]

            jet_obj_i = Jet(subject, best_start, best_end, boxi)

            # remove the extracts (we will add them back in in just a bit)
            jet_obj_i.box.extracts = []
            jet_obj_i.start.extracts = []
            jet_obj_i.end.extracts = []

            jets.append(jet_obj_i)

        # add the raw classifications back to the jet object
        # loop through the classifications
        for i, box in enumerate(boxes):
            # and the find the iou of this box wrt to the
            # unique jet clusters
            ious = np.zeros(len(unique_jets))
            for j, jet in enumerate(unique_jets):
                ious[j] = get_box_iou(jet, box)

            # we're going to find the "best" cluster i.e., the one with the
            # highest IoU
            index = np.argmax(ious)

            # and add the raw data to that cluster
            jets[index].box.extracts.append(box)

        # now do the same for the base/end points
        for i, point in enumerate(start_points):
            # and the find the distance between this point and
            # the cluster points
            dists = np.zeros(len(jets))
            for j, jet in enumerate(jets):
                dists[j] = get_point_distance(point, jet.start)

            # we're going to find the "best" cluster i.e., the one with the
            # lowest distance
            index = np.argmin(dists)

            # and add the raw data to that cluster
            jets[index].start.extracts.append(point)

        # now do the same for the base/end points
        for i, point in enumerate(end_points):
            # and the find the distance between this point and
            # the cluster points
            dists = np.zeros(len(jets))
            for j, jet in enumerate(jets):
                dists[j] = get_point_distance(point, jet.end)

            # we're going to find the "best" cluster i.e., the one with the
            # lowest distance
            index = np.argmin(dists)

            # and add the raw data to that cluster
            jets[index].end.extracts.append(point)

        if plot:
            fig, ax = plt.subplots(1, 1, dpi=150)

            img = get_subject_image(subject)
            ax.imshow(img)

            for point in unique_starts:
                ax.plot(*point.coordinates, 'b.', zorder=9)
            for point in unique_ends:
                ax.plot(*point.coordinates, 'y.', zorder=9)
            for jet in jets:
                jet.plot(ax)

            ax.axis('off')
            plt.tight_layout()
            plt.show()

        return jets
