from multiprocessing.sharedctypes import Value
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ast
import json
from panoptes_client import Panoptes, Subject, Workflow
from skimage import io, transform
import getpass
from shapely.geometry import Polygon, Point

def connect_panoptes():
    '''
        Login interface for the Panoptes client
    '''
    username = getpass.getpass('Username:')
    password = getpass.getpass('Password:')
    Panoptes.connect(username=username, password=password)

def get_box_edges(x, y, w, h, a):
    '''
        Return the corners of the box given one corner, width, height
        and angle
    '''
    cx = (2*x+w)/2
    cy = (2*y+h)/2
    centre = np.array([cx, cy])
    original_points = np.array(
      [
          [cx - 0.5 * w, cy - 0.5 * h],  # This would be the box if theta = 0
          [cx + 0.5 * w, cy - 0.5 * h],
          [cx + 0.5 * w, cy + 0.5 * h],
          [cx - 0.5 * w, cy + 0.5 * h],
          [cx - 0.5 * w, cy - 0.5 * h] # repeat the first point to close the loop
      ]
    )
    rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    corners = np.matmul(original_points - centre, rotation) + centre
    return corners

def get_subject_image(subject, frame=7): 
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    try:
        frame0_url = subjecti.raw['locations'][frame]['image/png']
    except KeyError:
        frame0_url = subjecti.raw['locations'][frame]['image/jpeg']
    
    img = io.imread(frame0_url)

    # for subjects that have an odd size, resize them
    if img.shape[0] != 1920:
        img = transform.resize(img, (1440, 1920))

    return img

def get_point_distance(x0, y0, x1, y1):
    return np.sqrt((x0-x1)**2. + (y0-y1)**2.)

def get_box_distance(box1, box2):
    b1_edges = get_box_edges(*box1)[:4]
    b2_edges = get_box_edges(*box2)[:4]

    # build a distance matrix between the 4 edges
    # since the order of edges may not be the same 
    # for the two boxes
    dists = np.zeros((4,4))
    for c1 in range(4):
        for c2 in range(4):
            dists[c1,c2] = get_point_distance(*b1_edges[c1], *b2_edges[c2])

    # then collapse the matrix into the minimum distance for each point
    # does not matter which axis, since we get the least distance anyway
    mindist = dists.min(axis=0)

    return np.average(mindist)


def create_gif(jets):
    subject = jets[0].subject

    fig, ax = plt.subplots(1,1, dpi=150)
    ax.imshow(get_subject_image(subject, 0))
    ax.axis('off')
    fig.tight_layout()

    ims = []
    for i in range(15):
        img = get_subject_image(subject, i)
        im1 = ax.imshow(img)
        jetims = []
        for jet in jets:
            jetims.extend(jet.plot(ax))
        
        ims.append([im1, *jetims])

    ani = animation.ArtistAnimation(fig, ims)
    ani.save(f'{subject}.gif', writer='imagemagick')

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
        self.points_file = points_file
        self.points_data = ascii.read(points_file, delimiter=',')

        self.box_file = box_file
        self.box_data = ascii.read(box_file, delimiter=',')

        for col in self.box_data.colnames:
            self.box_data[col].fill_value    = '[]'

        for col in self.points_data.colnames:
            self.points_data[col].fill_value = '[]'

    def get_points_data(self, subject, task):
        '''
            Get the points data and cluster, and associated probabilities and labels
            for a givens subject and task. s corresponds to the start and e corresponds to end

            Inputs
            ------
            subject : int
                Subject ID in zooniverse
            task : string
                Either 'T1' or 'T5' for the first jet or second jet

            Returns
            -------
            data : dict
                Raw classification data for the x, y (xs, ys for the start and xe, ye for the end)
            clusters : dict
                Cluster shape (x, y) for start and end and probabilities and labels of the 
                data points
        '''
        points_data = self.points_data[:][(self.points_data['subject_id']==subject)&(self.points_data['task']==task)]

        points_data = points_data.filled()

        data = {}

        data['x_start'] = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool0_points_x'][0]))
        data['y_start'] = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool0_points_y'][0]))
        data['x_end']   = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool1_points_x'][0]))
        data['y_end']   = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool1_points_y'][0]))

        clusters = {}
        clusters['x_start'] = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool0_clusters_x'][0]))
        clusters['y_start'] = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool0_clusters_y'][0]))
        clusters['x_end']   = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool1_clusters_x'][0]))
        clusters['y_end']   = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool1_clusters_y'][0]))

        clusters['prob_start']   = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool0_cluster_probabilities'][0]))
        clusters['labels_start'] = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool0_cluster_labels'][0]))
        clusters['prob_end']     = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool1_cluster_probabilities'][0]))
        clusters['labels_end']   = np.asarray(ast.literal_eval(points_data[f'data.frame0.{task}_tool1_cluster_labels'][0]))

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

            Returns
            -------
            data : dict
                Raw classification data for the x, y, width, height and angle (degrees)
            clusters : dict
                Cluster shape (x, y, width, height and angle) and probabilities and labels of the 
                data points
        '''
        box_data = self.box_data[:][(self.box_data['subject_id']==subject)&(self.box_data['task']==task)]

        box_data = box_data.filled()

        data = {}

        data['x'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_rotateRectangle_x'][0]))
        data['y'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_rotateRectangle_y'][0]))
        data['w'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_rotateRectangle_width'][0]))
        data['h'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_rotateRectangle_height'][0]))
        data['a'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_rotateRectangle_angle'][0]))

        clusters = {}

        clusters['x'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_clusters_x'][0]))
        clusters['y'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_clusters_y'][0]))
        clusters['w'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_clusters_width'][0]))
        clusters['h'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_clusters_height'][0]))
        clusters['a'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_clusters_angle'][0]))

        clusters['labels'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_cluster_labels'][0]))

        try:
            clusters['prob'] = np.asarray(ast.literal_eval(box_data[f'data.frame0.{task}_tool2_cluster_probabilities'][0]))
        except KeyError:
            probs = np.zeros(len(data['x']))
            for i in range(len(data['x'])):
                labeli = clusters['labels'][i]
                if labeli == -1:
                    probs[i] = 0
                else:
                    boxi_data   = Polygon(get_box_edges(data['x'][i], data['y'][i], 
                                                      data['w'][i], data['h'][i], 
                                                      np.radians(data['a'][i]))[:4])
                    boxi_clust = Polygon(get_box_edges(clusters['x'][labeli], clusters['y'][labeli], 
                                                      clusters['w'][labeli], clusters['h'][labeli], 
                                                      np.radians(clusters['a'][labeli]))[:4])

                    probs[i] = boxi_data.intersection(boxi_clust).area/boxi_data.union(boxi_clust).area
            clusters['prob'] = probs
                
        return data, clusters

    def filter_by_task(self, task):
        '''
            Return a subset of the data where the task 
            corresponds to the input value

            task : str
                the task key (from Zooniverse lab)
        '''

        task_mask = self.points_data['task']==task
        data_points = self.points_data[:][task_mask]

        task_mask = self.box_data['task']==task
        data_box = self.box_data[:][task_mask]
        
        return data_points, data_box

    def plot_clusters(self, task='both'):
        '''
            Plot the aggregated start/end/box for all subjects that have
            a cluster 

            task : str
                the task key (from Zooniverse lab)
        '''

        if task=='both':
            subs = []
            for task in ['T1', 'T5']:
                # filter the data by the task key
                points_data, box_data = self.filter_by_task(task)

                # get the subset of clustered data
                clustered_points_mask = (points_data[f'data.frame0.{task}_tool0_clusters_x'].filled()!='None')&\
                    (points_data[f'data.frame0.{task}_tool1_clusters_x'].filled()!='None')
                clustered_box_mask = box_data[f'data.frame0.{task}_tool2_clusters_x'].filled()!='None'

                # get a list of subjects from the subset        
                clust_subs = np.hstack((\
                        points_data['subject_id'][clustered_points_mask], box_data['subject_id'][clustered_box_mask]\
                    ))
                subs.append(np.unique(clust_subs))

            subs_both = np.intersect1d(subs[0], subs[1])

            print(f"Found {len(subs_both)} subjects with clusters")

            for subject in subs_both:
                fig, ax = plt.subplots(1,1, dpi=150)

                self.plot_subject(subject, 'T1', ax)
                self.plot_subject(subject, 'T5', ax)

                fig.tight_layout()
                plt.show()

        else:
            # filter the data by the task key
            points_data, box_data = self.filter_by_task(task)

            # get the subset of clustered data
            clustered_points_mask = (points_data[f'data.frame0.{task}_tool0_clusters_x'].filled()!='None')&\
                (points_data[f'data.frame0.{task}_tool1_clusters_x'].filled()!='None')
            clustered_box_mask = box_data[f'data.frame0.{task}_tool2_clusters_x'].filled()!='None'

            # get a list of subjects from the subset        
            clust_subs = np.hstack((\
                    points_data['subject_id'][clustered_points_mask], box_data['subject_id'][clustered_box_mask]\
                ))
            clustered_subs = np.unique(clust_subs)

            print(f"Found {len(clustered_subs)} subjects with clusters")

            plt.style.use('default')
            for i, subject in enumerate(clustered_subs):
                self.plot_subject(subject, task)

    def plot_subject_both(self, subject):
        '''
            Plots both tasks for a given subject
        '''
        fig, ax = plt.subplots(1,1, dpi=150)

        self.plot_subject(subject, 'T1', ax)
        self.plot_subject(subject, 'T5', ax)

        fig.tight_layout()
        plt.show()

    def plot_subject(self, subject, task, ax=None):
        '''
            Plot the data for a given subject/task 
        '''

        # get the points data and associated cluster
        points_data, points_clusters = self.get_points_data(subject, task)

        x0_i = points_data['x_start']
        y0_i = points_data['y_start']
        x1_i = points_data['x_end']
        y1_i = points_data['y_end']

        cx0_i = points_clusters['x_start']
        cy0_i = points_clusters['y_start']
        p0_i  = points_clusters['prob_start']
        cx1_i = points_clusters['x_end']
        cy1_i = points_clusters['y_end']
        p1_i  = points_clusters['prob_end']

        # do the same for the box
        box_data, box_clusters = self.get_box_data(subject, task)
        x_i  = box_data['x']
        y_i  = box_data['y']
        w_i  = box_data['w']
        h_i  = box_data['h']
        a_i  = box_data['a']
        
        cx_i  = box_clusters['x']
        cy_i  = box_clusters['y']
        cw_i  = box_clusters['w']
        ch_i  = box_clusters['h']
        ca_i  = box_clusters['a']
        pb_i  = box_clusters['prob']

        img = get_subject_image(subject)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=150)
            plot = True
        else:
            plot = False

        # plot the subject
        ax.imshow(img)
        
        # plot the raw classifications using a .
        alphai = np.asarray(p0_i)*0.5 + 0.5
        ax.scatter(x0_i, y0_i, 5.0, marker='.', color='blue', alpha=alphai)
        
        # and the end points with a yellow .
        alphai = np.asarray(p1_i)*0.5 + 0.5
        ax.scatter(x1_i, y1_i, 5.0, marker='.', color='yellow', alpha=alphai)
        
        # plot the clustered start/end with an x
        ax.scatter(cx0_i, cy0_i, 10.0, marker='x', color='blue')
        ax.scatter(cx1_i, cy1_i, 10.0, marker='x', color='yellow')
        
        # plot the raw boxes with a gray line
        for j in range(len(x_i)):
            points = get_box_edges(x_i[j], y_i[j], w_i[j], h_i[j], np.radians(a_i[j]))
            linewidthi = 0.2*pb_i[j]+0.1
            ax.plot(points[:,0], points[:,1], '-', color='#aaa', linewidth=linewidthi)
            
        # plot the clustered box in blue
        for j in range(len(cx_i)):
            clust = get_box_edges(cx_i[j], cy_i[j], cw_i[j], ch_i[j], np.radians(ca_i[j]))
            ax.plot(clust[:,0], clust[:,1], '-', linewidth=0.5, color='blue')
        
        ax.axis('off')

        ax.set_xlim((0, img.shape[1]))
        ax.set_ylim((img.shape[0], 0))

        if plot:
            fig.tight_layout()
            plt.show()

    def load_classification_data(self, classification_file='box-the-jets-classifications.csv'):
        self.classification_file = classification_file
        self.classification_data = ascii.read(classification_file, delimiter=',')

    def get_retired_subjects(self):
        subjects = np.unique(np.vstack((self.points_data['subject_id'][:], self.box_data['subject_id'][:])))
        print(len(subjects))

        self.retired_subjects = []

        assert hasattr(self, 'classification_data'), \
            "Please load the classification data using load_classification_data"

        for i, subject in enumerate(subjects):
            print("\r [%-20s] %d/%d"%(int(i/len(subjects)*20)*'=', i+1, len(subjects)), end='')
            
            # find the list of classifications for this subject
            subject_classifications = self.classification_data[:][\
                self.classification_data['subject_ids'][:]==subject]

            subject_metadata = subject_classifications['subject_data']

            for k, metak in enumerate(subject_metadata):
                metak = json.loads(metak)
                retired_at = metak[f"{subject}"]['retired']
                if retired_at is not None:
                    # if this subject was retired, add it to the list 
                    # and continue to the next subject
                    self.retired_subjects.append(subject)
                    break

    def load_extractor_data(self, point_extractor_file='point_extractor_by_frame_box_the_jets.csv', \
        box_extractor_file='shape_extractor_rotateRectangle_box_the_jets.csv'):
        self.point_extract_file = point_extractor_file
        self.point_extracts     = ascii.read(point_extractor_file, delimiter=',')

        self.box_extract_file = box_extractor_file
        self.box_extracts     = ascii.read(box_extractor_file, delimiter=',')

    def get_frame_time_base(self, subject, task='T1'):
        '''
            get the distribution of classifications by frame number for the base of the jet (both start and end)
            and determine the best frame for each, based on a cluster probability based metric
        '''
        # get the points data and associated cluster
        points_data, points_clusters = self.get_points_data(subject, task)

        x0_i = points_data['x_start']
        y0_i = points_data['y_start']
        x1_i = points_data['x_end']
        y1_i = points_data['y_end']

        cx0_i = points_clusters['x_start']
        cy0_i = points_clusters['y_start']
        p0_i  = points_clusters['prob_start']
        cx1_i = points_clusters['x_end']
        cy1_i = points_clusters['y_end']
        p1_i  = points_clusters['prob_end']

        # get the extract data so that we can obtain frame_time info
        # you need to run load_extractor data first
        assert hasattr(self, 'point_extracts'), "Please load the extractor data using the load_extractor_data method"
        point_extractsi = self.point_extracts[:][(self.point_extracts['subject_id']==subject)&(self.point_extracts['task']==task)]

        # empty lists to hold the frame info
        start_frames = []
        end_frames   = []

        # empty lists to hold the frame score
        start_weights = []
        end_weights   = []

        for frame in range(15):
            ## start of the jet is tool0
            frame_start_ext = point_extractsi[f'data.frame{frame}.{task}_tool0_x',f'data.frame{frame}.{task}_tool0_y'][:]
            for col in frame_start_ext.colnames:
                frame_start_ext[col].fill_value = 'None'
            
            # rename the columns for easier use
            frame_start_ext.rename_column(f'data.frame{frame}.{task}_tool0_x', f'frame{frame}_x')
            frame_start_ext.rename_column(f'data.frame{frame}.{task}_tool0_y', f'frame{frame}_y')
                
            # find the rows with data for this frame
            start_sub = frame_start_ext[:][frame_start_ext.filled()[f'frame{frame}_x'][:]!="None"]

            # parse the data from the string value and add it to the list
            points = []
            probs  = []
            for i in range(len(start_sub)):
                xx = ast.literal_eval(start_sub[f'frame{frame}_x'][i])[0]
                yy = ast.literal_eval(start_sub[f'frame{frame}_y'][i])[0]
                points.append([xx, yy])
                
                # find the associated cluster probability based on the index of the points
                # in the cluster info
                ind = np.where((xx==x0_i)&(yy==y0_i))[0]
                probs.append(p0_i[ind])

            # convert to numpy array and append to the list
            points = np.asarray(points)
            start_frames.append(points)
            start_weights.append(probs)
            
            # end of the jet is tool1
            # do the same process for tool1
            frame_end_ext = point_extractsi[f'data.frame{frame}.{task}_tool1_x',f'data.frame{frame}.{task}_tool1_y'][:]
            for col in frame_end_ext.colnames:
                frame_end_ext[col].fill_value = 'None'

            frame_end_ext.rename_column(f'data.frame{frame}.{task}_tool1_x', f'frame{frame}_x')
            frame_end_ext.rename_column(f'data.frame{frame}.{task}_tool1_y', f'frame{frame}_y')
                
            end_sub = frame_end_ext[:][frame_end_ext.filled()[f'frame{frame}_x'][:]!="None"]

            points = []
            probs = []
            for i in range(len(end_sub)):
                xx = ast.literal_eval(end_sub[f'frame{frame}_x'][i])[0]
                yy = ast.literal_eval(end_sub[f'frame{frame}_y'][i])[0]
                
                points.append([xx, yy])
                
                # find the associated cluster probability based on the index of the points
                # in the cluster info
                ind = np.where((xx==x1_i)&(yy==y1_i))[0]
                probs.append(p1_i[ind])

            points = np.asarray(points)
            end_frames.append(points)
            end_weights.append(probs)
        
        # the score is the sum of the probabilities at each frame
        start_score = [np.sum(starti) for starti in start_weights]
        end_score   = [np.sum(endi) for endi in end_weights]

        return {'start': start_frames, 'start_score': start_score, 'start_best': np.argmax(start_score), 
                'end': end_frames, 'end_score': end_score, 'end_best': np.argmax(end_score)}

    def get_frame_time_box(self, subject, task='T1'):
        '''
            get the distribution of classifications by frame number for box. similar to _base above
        '''
        # get the cluster and points information from the reduced data
        box_data, box_clusters = self.get_box_data(subject, task)
        x_i  = box_data['x']
        y_i  = box_data['y']
        w_i  = box_data['w']
        h_i  = box_data['h']
        a_i  = box_data['a']
        
        cx_i  = box_clusters['x']
        cy_i  = box_clusters['y']
        cw_i  = box_clusters['w']
        ch_i  = box_clusters['h']
        ca_i  = box_clusters['a']
        p0_i  = box_clusters['prob']

        # get the extract data so that we can obtain frame_time info
        # you need to run load_extractor data first
        assert hasattr(self, 'box_extracts'), "Please load the extractor data using the load_extractor_data method"
        box_extractsi = self.box_extracts[:][(self.box_extracts['subject_id']==subject)&(self.box_extracts['task']==task)]

        # empty lists to hold the frame info
        frames  = []
        weights = []

        for frame in range(15):
            # rename the columns to make it easier to work with
            box_extractsi.rename_column(f'data.frame{frame}.{task}_tool2_x', f'frame{frame}_x')
            box_extractsi.rename_column(f'data.frame{frame}.{task}_tool2_y', f'frame{frame}_y')
            box_extractsi.rename_column(f'data.frame{frame}.{task}_tool2_width', f'frame{frame}_w')
            box_extractsi.rename_column(f'data.frame{frame}.{task}_tool2_height', f'frame{frame}_h')
            box_extractsi.rename_column(f'data.frame{frame}.{task}_tool2_angle', f'frame{frame}_a')

            ## start of the jet is tool0
            frame_box = box_extractsi[f'frame{frame}_x',f'frame{frame}_y',f'frame{frame}_w',f'frame{frame}_h',f'frame{frame}_a'][:]
            for col in frame_box.colnames:
                frame_box[col].fill_value = 'None'
            
              
            # find the rows with data for this frame
            box_sub = frame_box[:][frame_box.filled()[f'frame{frame}_x'][:]!="None"]

            # parse the data from the string value and add it to the list
            box = []
            probs  = []
            for i in range(len(box_sub)):
                xx = ast.literal_eval(box_sub[f'frame{frame}_x'][i])[0]
                yy = ast.literal_eval(box_sub[f'frame{frame}_y'][i])[0]
                ww = ast.literal_eval(box_sub[f'frame{frame}_w'][i])[0]
                hh = ast.literal_eval(box_sub[f'frame{frame}_h'][i])[0]
                aa = ast.literal_eval(box_sub[f'frame{frame}_a'][i])[0]
                box.append([xx, yy, ww, hh, aa])
                
                # find the associated cluster probability based on the index of the box
                # in the cluster info
                ind = np.where((xx==x_i)&(yy==y_i)&(ww==w_i)&(hh==h_i)&(aa==a_i))[0]
                probs.append(p0_i[ind])

            # convert to numpy array and append to the list
            box = np.asarray(box)
            frames.append(box)
            weights.append(probs)
        
        # the score is the sum of the probabilities at each frame
        score = [np.sum(weight) for weight in weights]

        # get the number of classifications for each point in time
        npoints = [len(starti) for starti in frames]
    
        return {'box_frames': frames, 'box_score': score}

    def plot_frame_info(self, subject, task='T1'):
        '''
            plot the distribution of classifications by frame time
        '''
        base_points = self.get_frame_time_base(subject, task)
        box         = self.get_frame_time_box(subject, task)

        if (base_points is None) or (box is None):
            print(f"{subject} has no cluster.")
            return

        start_score = base_points['start_score']
        end_score   = base_points['end_score']
        box_score   = box['box_score']

        # get the number of classifications for each point in time
        npoints_start = [len(starti) for starti in base_points['start']]
        npoints_box   = [len(boxi) for boxi in box['box_frames']]
        npoints_end   = [len(endi) for endi in base_points['end']]
        
        # plot this "histogram"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=150, gridspec_kw={'height_ratios': [0.4, 1]})

        ax1.plot(npoints_start, 'r-', label='Start')
        ax1.plot(npoints_box, 'g-', label='Box')
        ax1.plot(npoints_end, 'b-', label='End')

        ax1.plot(start_score, 'r--', label='Score')
        ax1.plot(end_score, 'b--', label='Score')
        ax1.plot(box_score, 'g--', label='Score')

        ax1.plot(np.argmax(start_score), np.max(start_score), 'rx', label='Best')
        ax1.plot(np.argmax(end_score), np.max(end_score), 'bx', label='Best')
        ax1.plot(np.argmax(box_score), np.max(box_score), 'gx', label='Best')

        ax1.set_ylabel("# of classifications")
        ax1.set_xlabel("Frame #")
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
        clb = np.asarray(box_clust['labels'])

        # get distances for the start points
        start_dist = np.zeros(len(cx0))
        nc0        = len(cx0)
        for i in range(nc0):
            # subset the clusters based on the cluster label
            mask = cl0==i
            x0i  = x0[mask]
            y0i  = y0[mask]
            dists = np.zeros(len(x0i))

            # for each point, calculate the Euclidian distance
            for j in range(len(x0i)):
                dists[j] = get_point_distance(cx0[i], cy0[i], x0i[j], y0i[j])

            # the distance for this cluster is the average distance
            # of each clasification
            start_dist[i] = np.mean(dists)#/np.max(dists)

        # get distances for the end points
        end_dist = np.zeros(len(cx1))
        nc1        = len(cx1)
        for i in range(nc1):
            # subset the clusters
            mask = cl1==i
            x1i  = x1[mask]
            y1i  = y1[mask]
            dists = np.zeros(len(x1i))
            for j in range(len(x1i)):
                dists[j] = get_point_distance(cx1[i], cy1[i], x1i[j], y1i[j])

            end_dist[i] = np.mean(dists)#/np.max(dists)

        # for the boxes
        box_iou = np.zeros(len(cx))
        ncb     = len(cx)
        for i in range(ncb):
            # subset the clusters
            mask = clb==i
            xi  = x[mask]
            yi  = y[mask]
            wi  = w[mask]
            hi  = h[mask]
            ai  = a[mask]

            # calculate the IoU using Shapely's in built methods
            cb = Polygon(get_box_edges(cx[i], cy[i], cw[i], ch[i], np.radians(ca[i]))[:4])
            ious = np.zeros(len(xi))
            for j in range(len(xi)):
                bj = Polygon(get_box_edges(xi[j], yi[j], wi[j], hi[j], np.radians(ai[j]))[:4])
                ious[j] = cb.intersection(bj).area/cb.union(bj).area

            # average all the IoUs for a cluster
            box_iou[i] = np.mean(ious)

        return start_dist, end_dist, box_iou


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
        _, _, box_iou_T1     = self.get_cluster_confidence(subject, 'T1')
        data_T5, clusters_T5 = self.get_box_data(subject, 'T5')
        if len(clusters_T5['x']) > 0:
            _, _, box_iou_T5     = self.get_cluster_confidence(subject, 'T5')
        else:
            box_iou_T5 = []

        # combine the box data from the two tasks
        combined_boxes = {}
        for key in clusters_T1.keys():
            if key=='labels':
                T5_labels = clusters_T5[key]
                T5_labels = [labeli + 1 + np.max(clusters_T1[key]) for labeli in T5_labels if labeli > -1]
                combined_boxes[key] = [*clusters_T1[key], *T5_labels]
            else:
                combined_boxes[key] = [*clusters_T1[key], *clusters_T5[key]]

        combined_boxes['iou'] = [*box_iou_T1, *box_iou_T5]

        # add all the boxes to a bucket as long as they are 
        # valid clusters (iou > 0)
        temp_clust_boxes = []
        temp_box_ious    = []
        temp_box_count   = []
        for i in range(len(combined_boxes['x'])):
            x = combined_boxes['x'][i]
            y = combined_boxes['y'][i]
            w = combined_boxes['w'][i]
            h = combined_boxes['h'][i]
            a = np.radians(combined_boxes['a'][i])
            if combined_boxes['iou'][i] > 1.e-6:
                temp_clust_boxes.append(Polygon(get_box_edges(x, y, w, h, a)[:4]))
                temp_box_ious.append(combined_boxes['iou'][i])
                temp_box_count.append(np.sum(np.asarray(combined_boxes['labels'])==i))

        temp_clust_boxes = np.asarray(temp_clust_boxes)
        temp_box_ious    = np.asarray(temp_box_ious)
        temp_box_count   = np.asarray(temp_box_count)

        # now loop over this bucket of polygons
        # and see how well they match with each other
        # we will move the "good" boxes to a new list
        # so we can keep track of progress based on how 
        # many items are still in the queue
        clust_boxes = []
        while len(temp_clust_boxes) > 0:
            nboxes = len(temp_clust_boxes)

            # compare against the first box in the bucket
            # this will get removed at the end of this loop
            box0 = temp_clust_boxes[0]

            # to compare iou of box0 with other boxes
            ious = np.ones(nboxes)

            # to see if box0 needs to be merged with another 
            # box
            merge_mask = [False]*nboxes
            merge_mask[0] = True

            for j in range(1, nboxes):
                # find IoU for box0 vs boxj
                bj = temp_clust_boxes[j]
                ious[j] = box0.intersection(bj).area/box0.union(bj).area

                # if the IoU is better than the worst IoU of the classifications
                # for either box, then we should merge these two
                # this metric could be changed to be more robust in the future
                if ious[j] > np.min([temp_box_ious[0], temp_box_ious[j], 0.1]):
                    merge_mask[j] = True

            # add the box with the best iou to the cluster list
            sum_ious = temp_box_ious[merge_mask]*temp_box_count[merge_mask]
            clust_boxes.append(\
                temp_clust_boxes[merge_mask][np.argmax(sum_ious)])

            if plot:
                fig, ax = plt.subplots(1,1, dpi=150)
                ax.imshow(get_subject_image(subject))
                ax.plot(*box0.exterior.xy, 'b-')
                for j in range(1, nboxes):
                    bj = temp_clust_boxes[j]
                    if merge_mask[j]:
                        ax.plot(*bj.exterior.xy, 'k--', linewidth=0.5)
                    else:
                        ax.plot(*bj.exterior.xy, 'k-', linewidth=0.5)
                for j in range(nboxes):
                    bj = temp_clust_boxes[j]
                    ax.text(*np.mean(bj.exterior.xy, axis=1), round(ious[j], 2))

                ax.axis('off')
                plt.show()
            
            # and remove all the overlapping boxes from the list
            temp_clust_boxes = np.delete(temp_clust_boxes, merge_mask)
            temp_box_ious    = np.delete(temp_box_ious, merge_mask)
            temp_box_count   = np.delete(temp_box_count, merge_mask)

        return clust_boxes

    def find_unique_jet_points(self, subject, plot=False):
        # get the box data and clusters for the two tasks
        data_T1, clusters_T1 = self.get_points_data(subject, 'T1')
        start_dist_T1, end_dist_T1, _  = self.get_cluster_confidence(subject, 'T1')
        data_T5, clusters_T5 = self.get_points_data(subject, 'T5')
        start_dist_T5, end_dist_T5, _  = self.get_cluster_confidence(subject, 'T5')

        # combine the box data from the two tasks
        combined_starts = {}
        combined_ends   = {}
        for key in clusters_T1.keys():
            key_new = key.replace('_start','').replace('_end','')
            if 'start' in key:
                combined_starts[key_new] = [*clusters_T1[key], *clusters_T5[key]]
            elif 'end' in key:
                combined_ends[key_new]   = [*clusters_T1[key], *clusters_T5[key]]

        combined_starts['dist'] = [*start_dist_T1, *start_dist_T5]
        combined_ends['dist']   = [*end_dist_T1, *end_dist_T5]

        # add all the start points to a bucket 
        temp_clust_starts = []
        temp_start_dists  = []
        for i in range(len(combined_starts['x'])):
            x = combined_starts['x'][i]
            y = combined_starts['y'][i]
            temp_clust_starts.append([x,y])
            temp_start_dists.append(combined_starts['dist'][i])

        temp_clust_starts = np.asarray(temp_clust_starts)
        temp_start_dists  = np.asarray(temp_start_dists)
        
        temp_clust_ends = []
        temp_end_dists  = []
        for i in range(len(combined_ends['x'])):
            x = combined_ends['x'][i]
            y = combined_ends['y'][i]
            temp_clust_ends.append([x,y])
            temp_end_dists.append(combined_ends['dist'][i])

        temp_clust_ends = np.asarray(temp_clust_ends)
        temp_end_dists  = np.asarray(temp_end_dists)

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
            merge_mask = [False]*npoints

            # we will always remove this first point from the queue
            merge_mask[0] = True

            for j in range(1, npoints):
                # find distance for the first and jth point
                pointj = temp_clust_starts[j]
                dists[j] = get_point_distance(*start0, *pointj)

                # if the distance is better than the 1.5x the mean distance of 
                # point that make up this cluster, then we should merge these two
                # this metric could be changed to be more robust in the future
                if dists[j] < 1.5*np.max([temp_start_dists[0], temp_start_dists[j]]):
                    merge_mask[j] = True

            # add the point with the most compact intra-cluster distance to the cluster list
            clust_starts.append(\
                temp_clust_starts[merge_mask][np.argmin(temp_start_dists[merge_mask])])

            if plot:
                fig, ax = plt.subplots(1,1, dpi=150)
                ax.imshow(get_subject_image(subject))

                cir = Point(*start0).buffer(1.5*temp_start_dists[0])
                ax.plot(*start0, 'bx')
                ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                for j in range(1, npoints):
                    pointj = temp_clust_starts[j]
                    ax.plot(*pointj, 'kx')
                    cir = Point(*pointj).buffer(1.5*temp_start_dists[j])
                    ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                ax.axis('off')
                plt.show()
            
            # and remove all the merged points from the list
            temp_clust_starts = np.delete(temp_clust_starts, merge_mask, axis=0)
            temp_start_dists  = np.delete(temp_start_dists, merge_mask)
        
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
            merge_mask = [False]*npoints

            # we will always remove this first point from the queue
            merge_mask[0] = True

            for j in range(1, npoints):
                # find distance for the first and jth point
                pointj = temp_clust_ends[j]
                dists[j] = get_point_distance(*end0, *pointj)

                # if the distance is better than the 1.5x the mean distance of 
                # point that make up this cluster, then we should merge these two
                # this metric could be changed to be more robust in the future
                if dists[j] < 1.5*np.max([temp_end_dists[0], temp_end_dists[j]]):
                    merge_mask[j] = True

            # add the point with the most compact intra-cluster distance to the cluster list
            clust_ends.append(\
                temp_clust_ends[merge_mask][np.argmin(temp_end_dists[merge_mask])])

            if plot:
                fig, ax = plt.subplots(1,1, dpi=150)
                ax.imshow(get_subject_image(subject))

                cir = Point(*end0).buffer(1.5*temp_end_dists[0])
                ax.plot(*end0, 'yx')
                ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                for j in range(1, npoints):
                    pointj = temp_clust_ends[j]
                    ax.plot(*pointj, 'kx')
                    cir = Point(*pointj).buffer(1.5*temp_end_dists[j])
                    ax.plot(*cir.exterior.xy, 'k-', linewidth=0.5)
                ax.axis('off')
                plt.show()
            
            # and remove all the merged points from the list
            temp_clust_ends = np.delete(temp_clust_ends, merge_mask, axis=0)
            temp_end_dists  = np.delete(temp_end_dists, merge_mask)

        return np.asarray(clust_starts), np.asarray(clust_ends)
            
    def filter_classifications(self, subject):
        '''
            Find a list of unique jets in the subject
            and segregate the classifications into each cluster 
            based on IoU calculations

            Inputs
            ------
            subject : int
                The subject ID in Zooniverse

            Returns
            --------
            filtered_box_data : list
                List of dictionaries that have the classification info (x, y, w, h, a) 
                for each box, separated by the cluster number. i.e. each index in the list is 
                a different cluster, and each dictionary entry contains a list of values
                that correspond to that cluster. 
            unique_jets : list
                List of `shapely.geometry.Polygon` objects that define the rectangle for
                the best box for each cluster
        '''
        # get the box data and clusters for the two tasks
        data_T1, _ = self.get_box_data(subject, 'T1')
        data_T5, _ = self.get_box_data(subject, 'T5')
        
        point_data_T1, _ = self.get_points_data(subject, 'T1')
        point_data_T5, _ = self.get_points_data(subject, 'T5')
        
        # and the unique clusters
        unique_jets                = self.find_unique_jets(subject)
        unique_starts, unique_ends = self.find_unique_jet_points(subject)

        # combine the T1 and T5 raw data
        combined_boxes = {}
        for key in data_T1.keys():
            combined_boxes[key] = [*data_T1[key], *data_T5[key]]

        # combined start/end points
        combined_starts = {}
        for key in point_data_T1.keys():
            if 'start' in key:
                combined_starts[key] = [*point_data_T1[key], *point_data_T5[key]]
        
        combined_ends = {}
        for key in point_data_T1.keys():
            if 'end' in key:
                combined_ends[key] = [*point_data_T1[key], *point_data_T5[key]]

        jets = []

        # for each jet box, find the best start/end points
        for i, jeti in enumerate(unique_jets):
            box_points = np.transpose(jeti.exterior.xy)[:4]

            dists = []
            for j, point in enumerate(unique_starts):
                disti = np.median([np.linalg.norm(point-pointi) for pointi in box_points])
                dists.append(disti)

            best_start = unique_starts[np.argmin(dists)]
            
            dists = []
            for j, point in enumerate(unique_ends):
                disti = np.median([np.linalg.norm(point-pointi)*np.linalg.norm(point - best_start) for pointi in box_points])
                dists.append(disti)
            
            best_end = unique_ends[np.argmin(dists)]

            jets.append(Jet(subject, best_start, best_end, jeti))
        

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
            ious = np.zeros(len(unique_jets))
            for j, jet in enumerate(unique_jets):
                ious[j] = boxi.intersection(jet).area/boxi.union(jet).area

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
                jets[index].start_extracts[key.replace('_start', '')].append(combined_starts[key][i])

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
                jets[index].end_extracts[key.replace('_end','')].append(combined_ends[key][i])
        
        fig, ax = plt.subplots(1, 1, dpi=150)


        img = get_subject_image(subject)
        ax.imshow(img)

        x_s = [*point_data_T1['x_start'], *point_data_T5['x_start']]
        y_s = [*point_data_T1['y_start'], *point_data_T5['y_start']]

        for point in zip(x_s, y_s):
            ax.plot(*point, 'k.', markersize=1.5)
        for point in unique_starts:
            ax.plot(*point, 'b.')
        for point in unique_ends:
            ax.plot(*point, 'y.')
        for jet in jets:
            jet.plot(ax)

        ax.axis('off')
        plt.show()

        return jets

class Jet:
    '''
        Oject to hold the data associated with a single jet.
        Contains the start/end positions and associated extracts,
        and the box (as a `shapely.Polygon` object) and corresponding
        extracts
    '''
    def __init__(self, subject, start, end, box):
        self.subject = subject
        self.start   = start
        self.end     = end
        self.box     = box
        
        self.box_extracts   = {'x': [], 'y': [], 'w': [], 'h': [], 'a': []}
        self.start_extracts = {'x': [], 'y': []}
        self.end_extracts   = {'x': [], 'y': []}

        self.autorotate()

    def get_extract_starts(self):
        '''
            Get the extract coordinates associated with the 
            starting base points
        '''
        x_s = self.start_extracts['x']
        y_s = self.start_extracts['y']

        return np.transpose([x_s, y_s])

    def get_extract_ends(self):
        '''
            Get the extract coordinates associated with the 
            final base points
        '''
        x_e = self.end_extracts['x']
        y_e = self.end_extracts['y']

        return np.transpose([x_e, y_e])

    def get_extract_boxes(self):
        '''
            Get the extract shapely Polygons corresponding
            to the boxes
        '''
        boxes = []
        for i in range(len(self.box_extracts['x'])):
            x = self.box_extracts['x'][i]
            y = self.box_extracts['y'][i]
            w = self.box_extracts['w'][i]
            h = self.box_extracts['h'][i]
            a = np.radians(self.box_extracts['a'][i])

            # get the box
            boxes.append(Polygon(get_box_edges(x, y, w, h, a)[:4]))

        return boxes

    def plot(self, ax):
        '''
            Plot the the data for this jet object. Plots the 
            start and end clustered points, and the associated 
            extracts. Also plots the clustered and extracted 
            box. Also plots a vector from the base to the top of the box
        '''
        boxplot,   = ax.plot(*self.box.exterior.xy, 'b-')
        startplot, = ax.plot(*self.start, 'bx')
        endplot,   = ax.plot(*self.end, 'yx')

        start_ext = self.get_extract_starts()
        end_ext   = self.get_extract_ends()

        startextplot, = ax.plot(start_ext[:,0], start_ext[:,1], 'k.', markersize=1.5)
        endextplot,   = ax.plot(end_ext[:,0], end_ext[:,1], 'k.', markersize=1.5)
        boxextplots = []
        for box in self.get_extract_boxes():
            iou = box.intersection(self.box).area/box.union(self.box).area
            boxextplots.append(ax.plot(*box.exterior.xy, 'k-', linewidth=0.5, alpha=0.65*iou+0.05)[0])

        # also plot the center axis
        center   = np.mean(np.asarray(self.box.exterior.xy)[:,:4], axis=1)
        rotation = np.asarray([[np.cos(self.angle), -np.sin(self.angle)],
                               [np.sin(self.angle), np.cos(self.angle)]])
        point0 = center + np.matmul(rotation, np.asarray([0,  self.height/2.]))
        point1 = center + np.matmul(rotation, np.asarray([0, -self.height/2.]))
        vec    = point1 - point0
        #ax.plot(*np.transpose([point0, point1]), 'k-')

        base_points, height_points = self.get_width_height_pairs()
        #baseplot, = ax.plot(base_points[:,0], base_points[:,1], 'y--')
        #heightplot, = ax.plot(height_points[:,0], height_points[:,1], 'k--')

        arrowplot = ax.arrow(*point0, vec[0], vec[1], color='white', length_includes_head=True, head_width=25)

        return [boxplot, startplot, endplot, startextplot, endextplot, *boxextplots, arrowplot]

    def autorotate(self):
        '''
            Find the rotation of the jet wrt to solar north and 
            find the base width and height of the box
        '''
        box_points   = np.transpose(self.box.exterior.xy)[:4,:]

        # find the distance between each point and the starting base
        dists        = [np.linalg.norm((point - self.start)) for point in box_points]
        sorted_dists = np.argsort(dists)

        # the base points are the two points closest to the start
        base_points = np.array([box_points[sorted_dists[0]], box_points[sorted_dists[1]]])

        # the height points are the next two
        rolled_points = np.delete(np.roll(box_points, -sorted_dists[0],
                                          axis=0), 0, axis=0)

        # we want to make sure that the order of the points 
        # is in such a way that the point closest to the base
        # comes first -- this will ensure that the next point is
        # along the height line
        if np.linalg.norm(rolled_points[0,:]-base_points[1,:])==0:
            height_points = rolled_points[:2]
        else:
            height_points = rolled_points[::-1][:2]

        self.base_points   = base_points
        self.height_points = height_points

        # also figure out the angle and additional size metadata
        # the angle is the angle between the height points and the base
        dh = self.height_points[1] - self.height_points[0]
        self.angle = np.arctan2(dh[0], -dh[1])

        self.height = np.linalg.norm(dh)
        self.width  = np.linalg.norm(self.base_points[1] - self.base_points[0])

    def get_width_height_pairs(self):
        '''
            Returns the base points and the height line segment
            points 
        '''

        return self.base_points, self.height_points
