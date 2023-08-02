import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import tqdm
import json
from shapely.geometry import Polygon, Point
from .jet import Jet
from .shape_utils import get_box_edges, get_point_distance, sigma_shape, BasePoint, Box
from .zoo_utils import get_subject_image


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

        for i, subject_id in enumerate(tqdm.tqdm(self.subjects, desc='Getting data')):
            point_rows = point_reductions[(point_reductions['subject_id'] == subject_id) & (point_reductions['task'] == 'T0')]

            for row in point_rows:
                try:
                    cluster_start_x = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_clusters_x']))
                    cluster_start_y = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_clusters_y']))
                    cluster_start_time = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_clusters_displayTime']))
                except json.JSONDecodeError:
                    continue

                row_start_x = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_points_x']))
                row_start_y = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_points_y']))
                row_start_time = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_points_displayTime']))
                row_start_labels = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_cluster_labels']))
                row_start_probs = np.asarray(json.loads(row['data.frame0.T0_toolIndex0_cluster_probabilities']))

                for i, (x, y, time) in enumerate(zip(cluster_start_x, cluster_start_y, cluster_start_time)):
                    start_point = BasePoint(x=x, y=y, displayTime=time, subject_id=subject_id)

                    extracts_mask = row_start_labels == i
                    extracts_x = row_start_x[extracts_mask]
                    extracts_y = row_start_y[extracts_mask]
                    extracts_time = row_start_time[extracts_mask]
                    extracts_prob = row_start_probs[extracts_mask]

                    extract_points = []
                    for j, (x, y, time, prob) in enumerate(zip(extracts_x, extracts_y, extracts_time, extracts_prob)):
                        extract_points.append(BasePoint(x=x, y=y, displayTime=time, probability=prob, subject_id=subject_id))

                    start_point.extracts = extract_points

                    self.start_points_data.append(start_point)

                try:
                    cluster_end_x = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_clusters_x']))
                    cluster_end_y = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_clusters_y']))
                    cluster_end_time = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_clusters_displayTime']))
                except json.JSONDecodeError:
                    continue

                row_end_x = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_points_x']))
                row_end_y = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_points_y']))
                row_end_time = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_points_displayTime']))
                row_end_labels = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_cluster_labels']))
                row_end_probs = np.asarray(json.loads(row['data.frame0.T0_toolIndex1_cluster_probabilities']))

                for i, (x, y, time) in enumerate(zip(cluster_end_x, cluster_end_y, cluster_end_time)):
                    end_point = BasePoint(x=x, y=y, displayTime=time, subject_id=subject_id)

                    extracts_mask = row_end_labels == i
                    extracts_x = row_end_x[extracts_mask]
                    extracts_y = row_end_y[extracts_mask]
                    extracts_time = row_end_time[extracts_mask]
                    extracts_prob = row_end_probs[extracts_mask]

                    extract_points = []
                    for j, (x, y, time, prob) in enumerate(zip(extracts_x, extracts_y, extracts_time, extracts_prob)):
                        extract_points.append(BasePoint(x=x, y=y, displayTime=time, probability=prob, subject_id=subject_id))

                    end_point.extracts = extract_points

                    self.end_points_data.append(end_point)

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
        subjects = [e.subject_id for e in self.reductions_data]
        # if min([len(e[key]['clusters']) for key in ['box','start','end']]) > 0]

        return np.unique(subjects)

    def get_points_data(self, subject, task='T1'):
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
        points_data = list(filter(lambda e: e['subject_id'] == subject, self.reductions_data))

        assert len(points_data) == 1, f"Found {len(points_data)} matching reductions for points data!"

        # extrac the dictionary for this subject
        points_data = points_data[0]

        data = {}

        data['x_start'] = np.asarray(points_data['start']['extracts']['x'])
        data['y_start'] = np.asarray(points_data['start']['extracts']['y'])
        data['x_end'] = np.asarray(points_data['end']['extracts']['x'])
        data['y_end'] = np.asarray(points_data['end']['extracts']['y'])

        clusters = {}
        clusters['x_start'] = np.asarray(points_data['start']['clusters']['x'])
        clusters['y_start'] = np.asarray(points_data['start']['clusters']['y'])
        clusters['x_end'] = np.asarray(points_data['end']['clusters']['x'])
        clusters['y_end'] = np.asarray(points_data['end']['clusters']['y'])

        clusters['prob_start'] = np.asarray(points_data['start']['extracts']['cluster_probabilities'])
        clusters['prob_end'] = np.asarray(points_data['end']['extracts']['cluster_probabilities'])
        clusters['labels_start'] = np.asarray(points_data['start']['extracts']['cluster_labels'])
        clusters['labels_end'] = np.asarray(points_data['end']['extracts']['cluster_labels'])

        return data, clusters

    def get_box_data(self, subject, task):
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

        box_data = list(filter(lambda e: e['subject_id'] == subject, self.reductions_data))

        assert len(box_data) == 1, f"Found {len(box_data)} matching reductions for box data!"

        # extrac the dictionary for this subject
        box_data = box_data[0]

        data = {}

        data['x'] = np.asarray(box_data['box']['extracts']['x'])
        data['y'] = np.asarray(box_data['box']['extracts']['y'])
        data['w'] = np.asarray(box_data['box']['extracts']['width'])
        data['h'] = np.asarray(box_data['box']['extracts']['height'])
        data['a'] = np.asarray(box_data['box']['extracts']['angle'])

        clusters = {}

        try:
            clusters['x'] = np.asarray(box_data['box']['clusters']['x'])
            clusters['y'] = np.asarray(box_data['box']['clusters']['y'])
            clusters['w'] = np.asarray(box_data['box']['clusters']['width'])
            clusters['h'] = np.asarray(box_data['box']['clusters']['height'])
            clusters['a'] = np.asarray(box_data['box']['clusters']['angle'])
            
            clusters['sigma'] = np.asarray(box_data['box']['clusters']['sigma'])
        except Exception:
            for key in ['x', 'y', 'w', 'h', 'a', 'sigma']:
                clusters[key] = np.asarray([])
        clusters['labels'] = np.asarray(box_data['box']['extracts']['cluster_labels'])
        try:
            clusters['prob'] = box_data['box']['extracts']['cluster_probabilities']
        except KeyError:
            # OPTICS cluster doesn't have probabilities
            probs = np.zeros(len(data['x']))
            for i in range(len(data['x'])):
                labeli = clusters['labels'][i]
                if labeli == -1:
                    probs[i] = 0
                else:
                    boxi_data = Polygon(get_box_edges(data['x'][i], data['y'][i],
                                                      data['w'][i], data['h'][i],
                                                      np.radians(data['a'][i]))[:4])
                    boxi_clust = Polygon(get_box_edges(clusters['x'][labeli], clusters['y'][labeli],
                                                       clusters['w'][labeli], clusters['h'][labeli],
                                                       np.radians(clusters['a'][labeli]))[:4])

                    probs[i] = boxi_data.intersection(
                        boxi_clust).area / boxi_data.union(boxi_clust).area
            clusters['prob'] = probs

        return data, clusters

    def plot_subject_both(self, subject):
        '''
            Plots both tasks for a given subject

            Inputs
            ------
            subject : int
                Zooniverse subject ID
        '''
        fig, ax = plt.subplots(1, 1, dpi=150)

        self.plot_subject(subject, 'T1', ax)
        self.plot_subject(subject, 'T5', ax)

        fig.tight_layout()
        plt.show()

    def plot_subject(self, subject, task, ax=None):
        '''
            Plot the data for a given subject/task

            Inputs
            ------
            subject : int
                Zooniverse subject ID
            task : string
                task for the Zooniverse workflow (T1 for first jet and T2 for second jet)
            ax : matplotlib.Axes
                pass an axis variable to append the subject to a given axis (e.g., when
                making multi-subject plots where each axis corresponds to a subject).
                Default is None, and will create a new figure/axis combo.
        '''

        # get the points data and associated cluster
        points_data, points_clusters = self.get_points_data(subject, task)

        x0_i = points_data['x_start']
        y0_i = points_data['y_start']
        x1_i = points_data['x_end']
        y1_i = points_data['y_end']

        cx0_i = points_clusters['x_start']
        cy0_i = points_clusters['y_start']
        p0_i = points_clusters['prob_start']
        cx1_i = points_clusters['x_end']
        cy1_i = points_clusters['y_end']
        p1_i = points_clusters['prob_end']

        # do the same for the box
        box_data, box_clusters = self.get_box_data(subject, task)
        x_i = box_data['x']
        y_i = box_data['y']
        w_i = box_data['w']
        h_i = box_data['h']
        a_i = box_data['a']

        cx_i = box_clusters['x']
        cy_i = box_clusters['y']
        cw_i = box_clusters['w']
        ch_i = box_clusters['h']
        ca_i = box_clusters['a']
        sg_i = box_clusters['sigma']
        pb_i = box_clusters['prob']

        img = get_subject_image(subject)

        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=150)
            plot = True
        else:
            plot = False

        # plot the subject
        ax.imshow(img)

        # plot the raw classifications using a .
        alphai = np.asarray(p0_i) * 0.5 + 0.5
        ax.scatter(x0_i, y0_i, 5.0, marker='.', color='blue', alpha=alphai)

        # and the end points with a yellow .
        alphai = np.asarray(p1_i) * 0.5 + 0.5
        ax.scatter(x1_i, y1_i, 5.0, marker='.', color='yellow', alpha=alphai)

        # plot the clustered start/end with an x
        ax.scatter(cx0_i, cy0_i, 10.0, marker='x', color='blue')
        ax.scatter(cx1_i, cy1_i, 10.0, marker='x', color='yellow')

        # plot the raw boxes with a gray line
        for j in range(len(x_i)):
            points = get_box_edges(
                x_i[j], y_i[j], w_i[j], h_i[j], np.radians(a_i[j]))
            linewidthi = 0.2 * pb_i[j] + 0.1
            ax.plot(points[:, 0], points[:, 1], '-',
                    color='limegreen', linewidth=linewidthi)

        # plot the clustered box in blue
        for j in range(len(cx_i)):
            clust = get_box_edges(
                cx_i[j], cy_i[j], cw_i[j], ch_i[j], np.radians(ca_i[j]))

            # calculate the bounding box for the cluster confidence
            plus_sigma, minus_sigma = sigma_shape(
                [cx_i[j], cy_i[j], cw_i[j], ch_i[j], np.radians(ca_i[j])], sg_i[j])

            # get the boxes edges
            plus_sigma_box = get_box_edges(*plus_sigma)
            minus_sigma_box = get_box_edges(*minus_sigma)

            # create a fill between the - and + sigma boxes
            x_p = plus_sigma_box[:, 0]
            y_p = plus_sigma_box[:, 1]
            x_m = minus_sigma_box[:, 0]
            y_m = minus_sigma_box[:, 1]
            ax.fill(
                np.append(x_p, x_m[::-1]), np.append(y_p, y_m[::-1]), color='white', alpha=0.3)

            ax.plot(clust[:, 0], clust[:, 1], '-', linewidth=0.85, color='white')

        ax.axis('off')

        ax.set_xlim((0, img.shape[1]))
        ax.set_ylim((img.shape[0], 0))

        if plot:
            fig.tight_layout()
            plt.show()

    def load_classification_data(self, classification_file='box-the-jets-classifications.csv'):
        '''
            Load the classification data into the aggregator from the CSV file

            Inputs
            ------
            classification_file : str
                path to the classification file (in Zooniverse format)
        '''
        self.classification_file = classification_file
        self.classification_data = ascii.read(
            classification_file, delimiter=',')

    def get_retired_subjects(self):
        '''
            Loop through each subject and find out whether it has been retired.
            This is currently very slow since it queries each subject against the
            Zooniverse backend. Not required if the workflow has been completed.
            Retired subjects are added to the `retired_subjects` attribute.
        '''
        subjects = np.unique(
            np.vstack((self.points_data['subject_id'][:], self.box_data['subject_id'][:])))
        print(len(subjects))

        self.retired_subjects = []

        assert hasattr(self, 'classification_data'), \
            "Please load the classification data using load_classification_data"

        for i, subject in enumerate(subjects):
            print("\r [%-20s] %d/%d" %
                  (int(i / len(subjects) * 20) * '=', i + 1, len(subjects)), end='')

            # find the list of classifications for this subject
            subject_classifications = self.classification_data[:][
                self.classification_data['subject_ids'][:] == subject]

            subject_metadata = subject_classifications['subject_data']

            for k, metak in enumerate(subject_metadata):
                metak = json.loads(metak)
                retired_at = metak[f"{subject}"]['retired']
                if retired_at is not None:
                    # if this subject was retired, add it to the list
                    # and continue to the next subject
                    self.retired_subjects.append(subject)
                    break

    def load_extractor_data(self, point_extractor_file='point_extractor_by_frame_box_the_jets.csv',
                            box_extractor_file='shape_extractor_rotateRectangle_box_the_jets.csv'):
        '''
            Loads the file containing the raw extract data (before squashing the frame data
            together) and adds the table to the class
        '''
        self.point_extract_file = point_extractor_file
        self.point_extracts = ascii.read(point_extractor_file, delimiter=',')

        self.box_extract_file = box_extractor_file
        self.box_extracts = ascii.read(box_extractor_file, delimiter=',')

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

    def plot_frame_info(self, subject, task='T1'):
        '''
            plot the distribution of classifications by frame time

            Inputs
            ------
            subject : int
                Zooniverse subject ID
            task : string
                task for the Zooniverse workflow (T1 for first jet and T2 for second jet)
        '''
        base_points = self.get_frame_time_base(subject, task)
        box = self.get_frame_time_box(subject, task)

        if (base_points is None) or (box is None):
            print(f"{subject} has no cluster.")
            return

        start_score = base_points['start_score']
        end_score = base_points['end_score']
        box_score = box['box_score']

        # get the number of classifications for each point in time
        npoints_start = [len(starti) for starti in base_points['start']]
        npoints_box = [len(boxi) for boxi in box['box_frames']]
        npoints_end = [len(endi) for endi in base_points['end']]

        # plot this "histogram"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8),
                                       dpi=150, gridspec_kw={'height_ratios': [0.4, 1]})

        ax1.plot(npoints_start, 'r-', label='Start')
        ax1.plot(npoints_box, 'g-', label='Box')
        ax1.plot(npoints_end, 'b-', label='End')

        ax1.plot(start_score, 'r--', label='Score')
        ax1.plot(end_score, 'b--', label='Score')
        ax1.plot(box_score, 'g--', label='Score')

        ax1.plot(np.argmax(start_score), np.max(
            start_score), 'rx', label='Best')
        ax1.plot(np.argmax(end_score), np.max(end_score), 'bx', label='Best')
        ax1.plot(np.argmax(box_score), np.max(box_score), 'gx', label='Best')

        ax1.set_ylabel("# of classifications")
        ax1.set_xlabel("Frame # ")
        ax1.set_xlim((-1, 15))
        ax1.set_title(fr'{subject}')

        ax1.legend(loc='upper center', ncol=3)

        self.plot_subject(subject, task=task, ax=ax2)

        plt.tight_layout()
        plt.show()

    def get_cluster_confidence(self, subject, task='T1'):
        '''
            Calculate a confidence metric for a given subject for each of the start/end/box clusters.
            Start and end points are given by the mean euclidian distance between each point and the corresponding cluster
            while the box confidence is calculated from mean intersection over union.

            This method assumes that clusters exist, and will raise a `ValueError` if clusters do not exist for
            a given task.

            Inputs
            ------
            subject : int
                Zooniverse subject ID
            task : string
                task for the Zooniverse workflow (T1 for first jet and T2 for second jet)

            Outputs
            -------
            start_dist : numpy.ndarray
                Average distance between each extract with the cluster center
                for each cluster (for base start point)
            end_dist : numpy.ndarray
                Average distance between each extract with the cluster center
                for each cluster (for base end point)
            box_ious : numpy.ndarray
                Average IoU between extract boxes and the average cluster box
                for each cluster
            box_gamma : numpy.ndarray
                Average gamma scale (confidence interval) for each cluster
        '''
        # convert the data from the csv into an array
        points_data, point_clust = self.get_points_data(subject, task)
        box_data, box_clust = self.get_box_data(subject, task)

        x0 = points_data['x_start']
        y0 = points_data['y_start']
        x1 = points_data['x_end']
        y1 = points_data['y_end']

        x = box_data['x']
        y = box_data['y']
        w = box_data['w']
        h = box_data['h']
        a = box_data['a']

        cx0 = np.asarray(point_clust['x_start'])
        cy0 = np.asarray(point_clust['y_start'])
        cl0 = np.asarray(point_clust['labels_start'])

        cx1 = np.asarray(point_clust['x_end'])
        cy1 = np.asarray(point_clust['y_end'])
        cl1 = np.asarray(point_clust['labels_end'])

        cx = np.asarray(box_clust['x'])
        cy = np.asarray(box_clust['y'])
        cw = np.asarray(box_clust['w'])
        ch = np.asarray(box_clust['h'])
        ca = np.asarray(box_clust['a'])

        csg = np.asarray(box_clust['sigma'])
        clb = np.asarray(box_clust['labels'])

        # get distances for the start points
        start_dist = np.zeros(len(cx0))
        nc0 = len(cx0)
        for i in range(nc0):
            # subset the clusters based on the cluster label
            mask = cl0 == i
            x0i = x0[mask]
            y0i = y0[mask]
            dists = np.zeros(len(x0i))

            # for each point, calculate the Euclidian distance
            for j in range(len(x0i)):
                dists[j] = get_point_distance(cx0[i], cy0[i], x0i[j], y0i[j])

            # the distance for this cluster is the average distance
            # of each clasification
            start_dist[i] = np.mean(dists)  # /np.max(dists)

        # get distances for the end points
        end_dist = np.zeros(len(cx1))
        nc1 = len(cx1)
        for i in range(nc1):
            # subset the clusters
            mask = cl1 == i
            x1i = x1[mask]
            y1i = y1[mask]
            dists = np.zeros(len(x1i))
            for j in range(len(x1i)):
                dists[j] = get_point_distance(cx1[i], cy1[i], x1i[j], y1i[j])

            end_dist[i] = np.mean(dists)  # /np.max(dists)

        # for the boxes
        box_iou = np.zeros(len(cx))
        ncb = len(cx)
        for i in range(ncb):
            # subset the clusters
            mask = clb == i
            xi = x[mask]
            yi = y[mask]
            wi = w[mask]
            hi = h[mask]
            ai = a[mask]

            # calculate the IoU using Shapely's in built methods
            cb = Polygon(get_box_edges(
                cx[i], cy[i], cw[i], ch[i], np.radians(ca[i]))[:4])
            ious = np.zeros(len(xi))
            for j in range(len(xi)):
                bj = Polygon(get_box_edges(
                    xi[j], yi[j], wi[j], hi[j], np.radians(ai[j]))[:4])
                ious[j] = cb.intersection(bj).area / cb.union(bj).area

            # average all the IoUs for a cluster
            box_iou[i] = np.mean(ious)

        # use the sigma in the box size to
        # get the cluster uncertainty
        box_gamma = np.zeros(len(cx))
        for i in range(len(cx)):
            box_gamma[i] = np.sqrt(1. - csg[i])

        return start_dist, end_dist, box_iou, box_gamma

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
        # get the box data and clusters for the two tasks
        data_T1, clusters_T1 = self.get_box_data(subject, 'T1')
        _, _, box_iou_T1, _ = self.get_cluster_confidence(subject, 'T1')
        data_T5, clusters_T5 = self.get_box_data(subject, 'T5')
        if len(clusters_T5['x']) > 0:
            _, _, box_iou_T5, _ = self.get_cluster_confidence(subject, 'T5')
        else:
            box_iou_T5 = []

        # combine the box data from the two tasks
        combined_boxes = {}
        for key in clusters_T1.keys():
            if key == 'labels':
                T5_labels = clusters_T5[key]
                # move the T5 boxes to a different label so they don't
                # overlap with the T1 labels
                T5_labels = [
                    labeli + 1 + np.max(clusters_T1[key]) for labeli in T5_labels if labeli > -1]
                combined_boxes[key] = [*clusters_T1[key], *T5_labels]
            else:
                combined_boxes[key] = [*clusters_T1[key], *clusters_T5[key]]

        combined_boxes['iou'] = [*box_iou_T1, *box_iou_T5]

        # add all the boxes to a bucket as long as they are
        # valid clusters (iou > 0)
        temp_boxes = {}

        for key in combined_boxes.keys():
            temp_boxes[key] = []
        temp_boxes['box'] = []
        temp_boxes['count'] = []

        for i in range(len(combined_boxes['x'])):
            x = combined_boxes['x'][i]
            y = combined_boxes['y'][i]
            w = combined_boxes['w'][i]
            h = combined_boxes['h'][i]
            a = np.radians(combined_boxes['a'][i])
            if combined_boxes['iou'][i] > 1.e-6:
                temp_boxes['box'].append(
                    Polygon(get_box_edges(x, y, w, h, a)[:4]))
                temp_boxes['count'].append(
                    np.sum(np.asarray(combined_boxes['labels']) == i))
                for key in combined_boxes.keys():
                    if key == 'iou':
                        temp_boxes[key].append(combined_boxes[key][i] * temp_boxes['count'][-1])
                    else: 
                        temp_boxes[key].append(combined_boxes[key][i])

        for key in temp_boxes.keys():
            temp_boxes[key] = np.asarray(temp_boxes[key])

        # now loop over this bucket of polygons
        # and see how well they match with each other
        # we will move the "good" boxes to a new list
        # so we can keep track of progress based on how
        # many items are still in the queue
        clust_boxes = {}
        for key in temp_boxes.keys():
            clust_boxes[key] = []

        # we are going to sort by the IoU of each box
        # so that the best boxes are processed first
        sort_mask = np.argsort(temp_boxes['iou'] * temp_boxes['count'])[::-1]
        for key in temp_boxes.keys():
            temp_boxes[key] = temp_boxes[key][sort_mask]

        while len(temp_boxes['box']) > 0:
            nboxes = len(temp_boxes['box'])

            # compare against the first box in the bucket
            # this will get removed at the end of this loop
            box0 = temp_boxes['box'][0]

            # to compare iou of box0 with other boxes
            ious = np.ones(nboxes)

            # to see if box0 needs to be merged with another
            # box
            merge_mask = [False] * nboxes
            merge_mask[0] = True

            for j in range(1, nboxes):
                # find IoU for box0 vs boxj
                bj = temp_boxes['box'][j]
                ious[j] = box0.intersection(bj).area / box0.union(bj).area

                # if the IoU is better than the worst IoU of the classifications
                # for either box, then we should merge these two
                # this metric could be changed to be more robust in the future
                if ious[j] > np.min([temp_boxes['iou'][0], temp_boxes['iou'][j], 0.1]):
                    merge_mask[j] = True

            # add the box with the best iou to the cluster list
            for key in temp_boxes.keys():
                # np.argmax(sum_ious)
                best = np.argmax(temp_boxes['iou'][merge_mask])
                clust_boxes[key].append(temp_boxes[key][merge_mask][best])

            if plot:
                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.imshow(get_subject_image(subject))
                # ax.plot(*box0.exterior.xy, '-', color='k')
                for j in range(1, nboxes):
                    bj = temp_boxes['box'][j]
                    if merge_mask[j]:
                        ax.plot(*bj.exterior.xy, 'k--', linewidth=0.5)
                    else:
                        ax.plot(*bj.exterior.xy, 'k-', linewidth=0.5)
                for j in range(nboxes):
                    #ax.text((bj.exterior.xy[0][0] + bj.exterior.xy[0][1])/2.,
                    #        (bj.exterior.xy[1][0] + bj.exterior.xy[1][1])/2.,
                    #        f'IoU = {temp_boxes["iou"][j]:.1f}', fontsize=10)

                    # calculate the bounding box for the cluster confidence
                    plus_sigma, minus_sigma = sigma_shape(
                        [temp_boxes['x'][j],
                         temp_boxes['y'][j],
                         temp_boxes['w'][j],
                         temp_boxes['h'][j],
                         np.radians(temp_boxes['a'][j])],
                        temp_boxes['sigma'][j])

                    # get the boxes edges
                    plus_sigma_box = get_box_edges(*plus_sigma)
                    minus_sigma_box = get_box_edges(*minus_sigma)

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
            for key in temp_boxes.keys():
                temp_boxes[key] = np.delete(temp_boxes[key], merge_mask)

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
        # get the box data and clusters for the two tasks
        data_T1, clusters_T1 = self.get_points_data(subject, 'T1')
        start_dist_T1, end_dist_T1, _, _ = self.get_cluster_confidence(
            subject, 'T1')
        data_T5, clusters_T5 = self.get_points_data(subject, 'T5')
        start_dist_T5, end_dist_T5, _, _ = self.get_cluster_confidence(
            subject, 'T5')

        # combine the box data from the two tasks
        combined_starts = {}
        combined_ends = {}
        for key in clusters_T1.keys():
            key_new = key.replace('_start', '').replace('_end', '')
            if 'start' in key:
                combined_starts[key_new] = [
                    *clusters_T1[key], *clusters_T5[key]]
            elif 'end' in key:
                combined_ends[key_new] = [*clusters_T1[key], *clusters_T5[key]]

        combined_starts['dist'] = [*start_dist_T1, *start_dist_T5]
        combined_ends['dist'] = [*end_dist_T1, *end_dist_T5]

        # add all the start points to a bucket
        temp_clust_starts = []
        temp_start_dists = []
        for i in range(len(combined_starts['x'])):
            x = combined_starts['x'][i]
            y = combined_starts['y'][i]
            temp_clust_starts.append([x, y])
            temp_start_dists.append(combined_starts['dist'][i])

        temp_clust_starts = np.asarray(temp_clust_starts)
        temp_start_dists = np.asarray(temp_start_dists)

        temp_clust_ends = []
        temp_end_dists = []
        for i in range(len(combined_ends['x'])):
            x = combined_ends['x'][i]
            y = combined_ends['y'][i]
            temp_clust_ends.append([x, y])
            temp_end_dists.append(combined_ends['dist'][i])

        temp_clust_ends = np.asarray(temp_clust_ends)
        temp_end_dists = np.asarray(temp_end_dists)

        # now loop over this bucket of start points
        # and see how well they match with each other
        # we will move the "good" points to a new list
        # so we can keep track of progress based on how
        # many items are still in the queue
        clust_starts = []
        while len(temp_clust_starts) > 0:
            npoints = len(temp_clust_starts)

            # compare against the first point in the bucket
            # this will get removed at the end of this loop
            start0 = temp_clust_starts[0]

            # to compare distance of this 0th point with other points
            dists = np.zeros(npoints)

            # to see if start0 needs to be merged with another point
            merge_mask = [False] * npoints

            # we will always remove this first point from the queue
            merge_mask[0] = True

            for j in range(1, npoints):
                # find distance for the first and jth point
                pointj = temp_clust_starts[j]
                dists[j] = get_point_distance(*start0, *pointj)

                # if the distance is better than the 1.5x the mean distance of
                # point that make up this cluster, then we should merge these two
                # this metric could be changed to be more robust in the future
                if dists[j] < 1.5 * np.max([temp_start_dists[0], temp_start_dists[j]]):
                    merge_mask[j] = True

            # add the point with the most compact intra-cluster distance to the cluster list
            clust_starts.append(
                temp_clust_starts[merge_mask][np.argmin(temp_start_dists[merge_mask])])

            if plot:
                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.imshow(get_subject_image(subject))

                cir = Point(*start0).buffer(1.5 * temp_start_dists[0])
                ax.plot(*start0, 'bx')
                ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                for j in range(1, npoints):
                    pointj = temp_clust_starts[j]
                    ax.plot(*pointj, 'kx')
                    cir = Point(*pointj).buffer(1.5 * temp_start_dists[j])
                    ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                ax.axis('off')
                plt.show()

            # and remove all the merged points from the list
            temp_clust_starts = np.delete(
                temp_clust_starts, merge_mask, axis=0)
            temp_start_dists = np.delete(temp_start_dists, merge_mask)

        # repeat for the end points
        clust_ends = []
        while len(temp_clust_ends) > 0:
            npoints = len(temp_clust_ends)

            # compare against the first point in the bucket
            # this will get removed at the end of this loop
            end0 = temp_clust_ends[0]

            # to compare distance of this 0th point with other points
            dists = np.zeros(npoints)

            # to see if end0 needs to be merged with another point
            merge_mask = [False] * npoints

            # we will always remove this first point from the queue
            merge_mask[0] = True

            for j in range(1, npoints):
                # find distance for the first and jth point
                pointj = temp_clust_ends[j]
                dists[j] = get_point_distance(*end0, *pointj)

                # if the distance is better than the 1.5x the mean distance of
                # point that make up this cluster, then we should merge these two
                # this metric could be changed to be more robust in the future
                if dists[j] < 1.5 * np.max([temp_end_dists[0], temp_end_dists[j]]):
                    merge_mask[j] = True

            # add the point with the most compact intra-cluster distance to the cluster list
            clust_ends.append(
                temp_clust_ends[merge_mask][np.argmin(temp_end_dists[merge_mask])])

            if plot:
                fig, ax = plt.subplots(1, 1, dpi=150)
                ax.imshow(get_subject_image(subject))

                cir = Point(*end0).buffer(1.5 * temp_end_dists[0])
                ax.plot(*end0, 'yx')
                ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                for j in range(1, npoints):
                    pointj = temp_clust_ends[j]
                    ax.plot(*pointj, 'kx')
                    cir = Point(*pointj).buffer(1.5 * temp_end_dists[j])
                    ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                ax.axis('off')
                plt.show()

            # and remove all the merged points from the list
            temp_clust_ends = np.delete(temp_clust_ends, merge_mask, axis=0)
            temp_end_dists = np.delete(temp_end_dists, merge_mask)

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
        data_T1, _ = self.get_box_data(subject, 'T1')
        data_T5, _ = self.get_box_data(subject, 'T5')

        point_data_T1, _ = self.get_points_data(subject, 'T1')
        point_data_T5, _ = self.get_points_data(subject, 'T5')

        # and the unique clusters
        unique_jets = self.find_unique_jets(subject)

        if len(unique_jets['box']) < 1:
            return []

        unique_starts, unique_ends = self.find_unique_jet_points(subject)

        # combine the T1 and T5 raw data
        combined_boxes = {}
        for key in data_T1.keys():
            combined_boxes[key] = [*data_T1[key], *data_T5[key]]

        # combined start/end points
        combined_starts = {}
        for key in point_data_T1.keys():
            if 'start' in key:
                combined_starts[key] = [
                    *point_data_T1[key], *point_data_T5[key]]

        combined_ends = {}
        for key in point_data_T1.keys():
            if 'end' in key:
                combined_ends[key] = [*point_data_T1[key], *point_data_T5[key]]

        jets = []

        # for each jet box, find the best start/end points
        for i, jeti in enumerate(unique_jets['box']):
            box_points = np.transpose(jeti.exterior.xy)[:4]

            dists = []
            # calculate the distance between each start point
            # and the box edges
            for j, point in enumerate(unique_starts):
                disti = np.median([np.linalg.norm(point - pointi)
                                  for pointi in box_points])
                dists.append(disti)

            # the best start point is the one with the minimum
            # distance with all 4
            best_start = unique_starts[np.argmin(dists)]

            dists = []
            # do the same for the end points
            for j, point in enumerate(unique_ends):
                disti = np.median([np.linalg.norm(
                    point - pointi) * np.linalg.norm(point - best_start) for pointi in box_points])
                dists.append(disti)

            best_end = unique_ends[np.argmin(dists)]

            # create the jet parameters (edge, width, height, angle)
            jet_params = [unique_jets['x'][i],
                          unique_jets['y'][i],
                          unique_jets['w'][i],
                          unique_jets['h'][i],
                          np.radians(unique_jets['a'][i])]

            jet_obj_i = Jet(subject, best_start, best_end, jeti, jet_params)

            jet_obj_i.sigma = unique_jets['sigma'][i]

            jets.append(jet_obj_i)

        # add the raw classifications back to the jet object
        # loop through the classifications
        for i in range(len(combined_boxes['x'])):
            x = combined_boxes['x'][i]
            y = combined_boxes['y'][i]
            w = combined_boxes['w'][i]
            h = combined_boxes['h'][i]
            a = np.radians(combined_boxes['a'][i])

            # get the box
            boxi = Polygon(get_box_edges(x, y, w, h, a)[:4])

            # and the find the iou of this box wrt to the
            # unique jet clusters
            ious = np.zeros(len(unique_jets['box']))
            for j, jet in enumerate(unique_jets['box']):
                ious[j] = boxi.intersection(jet).area / boxi.union(jet).area

            # we're going to find the "best" cluster i.e., the one with the
            # highest IoU
            index = np.argmax(ious)

            # and add the raw data to that cluster
            for key in data_T1.keys():
                jets[index].box_extracts[key].append(combined_boxes[key][i])

        # now do the same for the base/end points
        for i in range(len(combined_starts['x_start'])):
            x = combined_starts['x_start'][i]
            y = combined_starts['y_start'][i]

            # and the find the distance between this point and
            # the cluster points
            dists = np.zeros(len(jets))
            for j, jet in enumerate(jets):
                dists[j] = get_point_distance(x, y, *jet.start)

            # we're going to find the "best" cluster i.e., the one with the
            # lowest distance
            index = np.argmin(dists)

            # and add the raw data to that cluster
            for key in combined_starts.keys():
                jets[index].start_extracts[key.replace(
                    '_start', '')].append(combined_starts[key][i])

        # same for the ends
        for i in range(len(combined_ends['x_end'])):
            x = combined_ends['x_end'][i]
            y = combined_ends['y_end'][i]

            dists = np.zeros(len(jets))
            for j, jet in enumerate(jets):
                dists[j] = get_point_distance(x, y, *jet.end)

            # we're going to find the "best" cluster i.e., the one with the
            # lowest distance
            index = np.argmin(dists)

            # and add the raw data to that cluster
            for key in combined_ends.keys():
                jets[index].end_extracts[key.replace(
                    '_end', '')].append(combined_ends[key][i])

        if plot:
            fig, ax = plt.subplots(1, 1, dpi=150)

            img = get_subject_image(subject)
            ax.imshow(img)

            x_s = [*point_data_T1['x_start'], *point_data_T5['x_start']]
            y_s = [*point_data_T1['y_start'], *point_data_T5['y_start']]

            for point in zip(x_s, y_s):
                ax.plot(*point, 'k.', markersize=1.5, zorder=9)
            for point in unique_starts:
                ax.plot(*point, 'b.', zorder=9)
            for point in unique_ends:
                ax.plot(*point, 'y.', zorder=9)
            for jet in jets:
                jet.plot(ax)

            ax.axis('off')
            plt.tight_layout()
            plt.show()

        return jets
