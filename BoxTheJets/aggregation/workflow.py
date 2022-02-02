from multiprocessing.sharedctypes import Value
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import json
from panoptes_client import Panoptes, Subject, Workflow
from skimage import io
import getpass

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

def get_subject_image(subject): 
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    try:
        frame0_url = subjecti.raw['locations'][7]['image/png']
    except KeyError:
        frame0_url = subjecti.raw['locations'][7]['image/jpeg']
    
    img = io.imread(frame0_url)

    return img

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
            self.box_data[col].fill_value    = 'None'

        for col in self.points_data.colnames:
            self.points_data[col].fill_value    = 'None'


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

    def plot_subject(self, subject, task, ax=None):
        '''
            Plot the data for a given subject/task 
        '''
        points_datai = self.points_data[:][(self.points_data['subject_id']==subject)&(self.points_data['task']==task)]
        # convert the data from the csv into an array
        try:
            x0_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_points_x'][0])
            y0_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_points_y'][0])
        except ValueError:
            x0_i = y0_i = []

        try:
            x1_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_points_x'][0])
            y1_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_points_y'][0])
        except ValueError:
            x1_i = y1_i = []
        
        # find the row corresponding to this subject in the box data for this task
        box_datai = self.box_data[:][(self.box_data['subject_id']==subject)&(self.box_data['task']==task)]
        try:
            x_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_rotateRectangle_x'][0])
            y_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_rotateRectangle_y'][0])
            w_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_rotateRectangle_width'][0])
            h_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_rotateRectangle_height'][0])
            a_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_rotateRectangle_angle'][0])
        except ValueError:
            x_i = y_i = w_i = h_i = a_i = []

        try:
            cx0_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_clusters_x'][0])
            cy0_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_clusters_y'][0])
            p0_i  = ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_cluster_probabilities'][0])
        except (ValueError, KeyError) as e:
            cx0_i = cy0_i = []
            p0_i  = np.ones_like(x0_i)

        try:
            cx1_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_clusters_x'][0])
            cy1_i = ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_clusters_y'][0])
            p1_i  = ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_cluster_probabilities'][0])
        except ValueError:
            cx1_i = cy1_i = []
            p1_i  = np.ones_like(x1_i)

        plot_box = False
        try:
            cx_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_x'][0])
            cy_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_y'][0])
            cw_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_width'][0])
            ch_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_height'][0])
            ca_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_angle'][0])
            pb_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_cluster_probabilities'][0])

            plot_box = True
        except (ValueError, KeyError) as e:
            pb_i = np.ones_like(x_i)
        
        img = get_subject_image(subject)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=150)
            plot = True
        else:
            plot = False
        ax.imshow(img)
        
        # plot the raw classifications using a .
        try:
            alphai = np.asarray(p0_i)*0.5 + 0.5
            ax.scatter(x0_i, y0_i, 5.0, marker='.', color='blue', alpha=alphai)
        except ValueError:
            pass

        try:
            alphai = np.asarray(p0_i)*0.5 + 0.5
            ax.scatter(x1_i, y1_i, 5.0, marker='.', color='yellow', alpha=alphai)
        except ValueError:
            pass
        
        # plot the clustered start/end with an x
        ax.scatter(cx0_i, cy0_i, 10.0, marker='x', color='blue')
        ax.scatter(cx1_i, cy1_i, 10.0, marker='x', color='yellow')
        
        # plot the raw boxes with a gray line
        for j in range(len(x_i)):
            points = get_box_edges(x_i[j], y_i[j], w_i[j], h_i[j], np.radians(a_i[j]))
            linewidthi = 0.2*pb_i[j]+0.05
            ax.plot(points[:,0], points[:,1], '-', color='#aaa', linewidth=linewidthi)
            
        # plot the clustered box in blue
        if plot_box:
            for j in range(len(cx_i)):
                clust = get_box_edges(cx_i[j], cy_i[j], cw_i[j], ch_i[j], np.radians(ca_i[j]))
                ax.plot(clust[:,0], clust[:,1], '-', linewidth=0.5, color='blue')
        
        ax.axis('off')

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

        for i, subject in enumerate(subjects):
            print("\r [%-20s] %d/%d"%(int(i/len(subjects)*20)*'=', i+1, len(subjects)), end='')
            if not hasattr(self, 'classification_data'): 
                raise KeyError("Please load the classification data using load_classification_data")
            
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

        # get the cluster and points information from the reduced data
        points_datai = self.points_data[:][(self.points_data['subject_id']==subject)&(self.points_data['task']==task)]
        try:
            cx0_i = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_clusters_x'][0]))
            cy0_i = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_clusters_y'][0]))
            x0_i  = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_points_x'][0]))
            y0_i  = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_points_y'][0]))
            p0_i  = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool0_cluster_probabilities'][0]))
        except (ValueError, KeyError) as e:
            return

        try:
            cx1_i = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_clusters_x'][0]))
            cy1_i = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_clusters_y'][0]))
            x1_i  = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_points_x'][0]))
            y1_i  = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_points_y'][0]))
            p1_i  = np.asarray(ast.literal_eval(points_datai[f'data.frame0.{task}_tool1_cluster_probabilities'][0]))
        except (ValueError, KeyError) as e:
            return

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
        
        # get the number of classifications for each point in time
        npoints_start = [len(starti) for starti in start_frames]
        npoints_end   = [len(endi) for endi in end_frames]

        # the score is the sum of the probabilities at each frame
        start_score = [np.sum(starti) for starti in start_weights]
        end_score   = [np.sum(endi) for endi in end_weights]

        # plot this "histogram"
        fig, ax = plt.subplots(1,1, dpi=150)
        ax.plot(npoints_start, 'r-', label='Start')
        ax.plot(start_score, 'r--', label='Score')
        ax.plot(npoints_end, 'b-', label='End')
        ax.plot(end_score, 'b--', label='Score')
        ax.plot(np.argmax(start_score), np.max(start_score), 'rx', label='Best')
        ax.plot(np.argmax(end_score), np.max(end_score), 'bx', label='Best')

        ax.set_ylabel("# of classifications")
        ax.set_xlabel("Frame #")
        ax.set_xlim((-1, 15))
        ax.set_title(fr'{subject}')
        
        plt.legend(loc='upper center')
        plt.show()