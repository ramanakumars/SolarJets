#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:01:58 2022

@author: pjol
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from panoptes_client import Panoptes, Subject, Workflow
from dateutil.parser import parse
import ast
import os
import datetime
from matplotlib.dates import DateFormatter
import matplotlib.animation as animation
from .workflow import Aggregator, get_subject_image
from sklearn.cluster import OPTICS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import hdbscan
import tqdm

def get_h_w_clusterbox(subject,aggregator, task='T1'):
    '''
    Get the dimensions of the cluster box for a given subject
    This function will be replaced when the find_unique_jets function is ready
    '''
    box_datai = aggregator.box_data[:][(aggregator.box_data['subject_id']==subject)&(aggregator.box_data['task']==task)]
    cw_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_width'][0])
    ch_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_height'][0])
    ca_i = ast.literal_eval(box_datai[f'data.frame0.{task}_tool2_clusters_angle'][0])
    for i in [x for x in range(len(ca_i)) if ca_i[x] > 200]: #if angle is larger than 200 reverse width and height
        ch_i[i], cw_i[i] = cw_i[i],ch_i[i]
    return cw_i,ch_i


class SOL:
    '''
        Single data class to handle all function related to a HEK/SOL_event
    '''
    def __init__(self, SOL_stats_file, aggregator):
        '''
            Inputs
            ------
            SOL_event : str
                Date and time of a read in event
                Solar Object Locator of HEK database
                format: 'SOLyyyy-mm-ddThh:mm:ssL000C000'

        '''
        self.SOL_small, self.SOL_subjects, self.filenames0, self.times, self.Num, \
            self.start, self.end, self.notes = \
            np.loadtxt(SOL_stats_file, delimiter=',',unpack=True,dtype=str)
        self.aggregator = aggregator
    
    def event_bar_plot(self, SOL_event, task='Tc'):
        '''   
        Show the bar plot, indicating locations of jets for a given SOL event
        Produced by SOL_analytics.ipynb
        
        task : str 
            the task key (from Zooniverse)
            default Tc (combined results of task T0 and T3)
        '''
        fig = Image(filename=('SOL/Agreement_SOL_'+task+'/'+SOL_event.replace(':','-')+'.png'))
        display(fig)
        
    def get_subjects(self, SOL_event):
        '''
        Get the subjects that correspond to a given SOL event
        
        SOL_small: list of the SOL events used in Zooniverse
        SOL_subjects: list of subjects corresponding to a given SOL event
        
        Read in using 
        SOL_small,SOL_subjects,times,Num,start,end,notes=np.loadtxt('path/SOL/SOL_{}_stats.csv'.format('Tc'),delimiter=',',unpack=True,dtype=str)
        Num=Num.astype(float)
        '''
        i=np.argwhere(self.SOL_small==SOL_event)[0][0]
        subjects=np.fromstring(self.SOL_subjects[i], dtype=int, sep=' ')
        
        return subjects   
    
    def get_obs_time(self, SOL_event):
        '''
        Get the observation times of a given SOL event
        
        times: start observation times for subjects in a SOL event
        saved in SOL_Tc_stats.csv
        '''
        i=np.argwhere(self.SOL_small==SOL_event)[0][0]
        T=[a + 'T'+ b for a, b in zip(self.times[i].split(' ')[::2],self.times[i].split(' ')[1::2])]
        obs_time=np.array([parse(T[t]) for t in range(len(T))],dtype='datetime64')
        return obs_time
    
    def plot_subjects(self, SOL_event):  
        '''
        Plot all the subjects with aggregation data of a given SOL event
        '''
        subjects=self.get_subjects(SOL_event)
        obs_time=self.get_obs_time(SOL_event)

        for subject in subjects:
            ## check to make sure that these subjects had classification
            subject_rows = self.aggregator.points_data[:][self.aggregator.points_data['subject_id']==subject]
            nsubjects = len(subject_rows['data.frame0.T1_tool0_points_x'])
            if nsubjects > 0:
                #aggregator.plot_subject(subject, task='T1')
                self.aggregator.plot_frame_info(subject, task='T1')

        
    def get_start_end_time(self, SOL_event):
        '''
        Get the start and end times of jet clusters in given SOL event
        
        start: start time subject with jet
        end: end time subject with jet
        saved in SOL_Tc_stats.csv
        '''
        i=np.argwhere(self.SOL_small==SOL_event)[0][0]
        S=[a + 'T'+ b for a, b in zip(self.start[i].split(' ')[::2], self.start[i].split(' ')[1::2])]
        start_time=np.array([parse(S[t]) for t in range(len(S))],dtype='datetime64')
        E=[a + 'T'+ b for a, b in zip(self.end[i].split(' ')[::2], self.end[i].split(' ')[1::2])]
        end_time=np.array([parse(E[t]) for t in range(len(E))],dtype='datetime64')    
        return start_time, end_time
    
    def get_notes_time(self, SOL_event):
        '''
        Get the notes of jet clusters in given SOL event
        
        notes: flags given to subjects 
            100 means an event of less than 6 minutes
            010 means an event where 2 event are closely after eachother
        saved in SOL_Tc_stats.csv
        '''
        i=np.argwhere(self.SOL_small==SOL_event)[0][0]
        flag=np.array(self.notes[i].split(' ')[1::3])
        N= [a + 'T'+ b for a, b in zip(self.notes[i].split(' ')[2::3],self.notes[i].split(' ')[3::3])]
        notes_time=np.array([parse(N[t]) for t in range(len(N))],dtype='datetime64')
        return notes_time, flag
    
    def get_filenames0(self, SOL_event):
        '''
        Get the filenames of the first image for each subjects. 
        filenames0: name of first image in the subject
        '''
        i=np.argwhere(self.SOL_small==SOL_event)[0][0]
        files=np.array(self.filenames0[i].split(' '))
        return files
    
    def get_box_dim(self,SOL_event, p=True):  
        '''
        Get the height and width arrays of the subjects inside a given SOL event
        '''
        subjects=self.get_subjects(SOL_event)
        obs_time=self.get_obs_time(SOL_event)
        W=np.array([])
        H=np.array([])
    
        for subject in subjects:
            ## check to make sure that these subjects had classification
            subject_rows = self.aggregator.points_data[:][self.aggregator.points_data['subject_id']==subject]
            nsubjects = len(subject_rows['data.frame0.T1_tool0_points_x'])
            if nsubjects > 0:
                try:
                    width,height=get_h_w_clusterbox(subject,self.aggregator, task='T1') #Function will be replaced by find_unique_jets 
                    W=np.append(W,width[0]) #Only one box is chosen if more clusters are present
                    H=np.append(H,height[0])
                    #print(width,height)
                    
                except:
                    if p==True:
                        print('No box size available')
                    W=np.append(W,0)
                    H=np.append(H,0)
    
            else:
                if p==True:
                    print(f"{subject} has no classification data!")
                W=np.append(W,0)
                H=np.append(H,0)
                
        return H,W    

    def box_height_width_plot(self,SOL_event, height,width,save=False):
        '''
        Plot the heigth and width evolution over time in a SOL event
        
        height: np.array with height box per subject
        width: np.array with width box per subject
        '''
        obs_time=self.get_obs_time(SOL_event)
        fig, (ax1, ax2) = plt.subplots(2,dpi=150,figsize=(4.8,4),sharex=True)
        x1, y1 = zip(*sorted(zip(obs_time, height)))
        x2, y2 = zip(*sorted(zip(obs_time, width)))
        ax1.plot(x1,y1,color='red')
        ax2.plot(x2,y2,color='red')
        date_form = DateFormatter("%H:%M")
        ax2.xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45)
        ax1.set_ylabel('Height (pix)')
        ax1.set_title(SOL_event)
        ax2.set_ylabel('Width (pix)')
        ax1.set_ylim(0,np.max(np.maximum(height,width))+30)
        ax2.set_ylim(0,np.max(np.maximum(height,width))+30)
        if save==True:
            path = 'SOL/SOL_Box_size/'
            #check if folder for plots exists
            isExist = os.path.exists(path)
            if not isExist: 
              os.makedirs(path)
              print("SOL_Box directory is created")
        
            plt.savefig('SOL/SOL_Box_size'+'/'+SOL_event.replace(':','-')+'.png')
        
        plt.show()
        
        
    def event_box_plot(self, SOL_event):
        fig = Image(filename=('SOL/SOL_Box_size/'+SOL_event.replace(':','-')+'.png'))
        display(fig)
            
            
            
    def filter_jet_clusters(self, SOL_event, eps=2., time_eps=2.):
        # first, get a list of subjects for
        # this event
        subjects  = self.get_subjects(SOL_event)
        times_all = self.get_obs_time(SOL_event)

        event_jets        = []
        jet_starts        = []
        times             = []
        start_confidences = []
        # go through the subjects, and find
        # the jets in each subject
        for j, subject in enumerate(subjects):
            try:
                # find all the jets in this subject
                jets = self.aggregator.filter_classifications(subject)

                # add it to the list
                event_jets.extend(jets)

                # and also keep track of the base positions
                jet_starts.extend([jet.start for jet in jets])

                start_dist = []
                for jet in jets:
                    start_dist.extend(np.linalg.norm(jet.get_extract_starts() - jet.start, axis=0))

                start_confidences.extend(start_dist)
                times.extend([times_all[j] for n in range(len(jets))])

            except (ValueError, IndexError):
                continue

        jets       = np.asarray(event_jets)
        jet_starts = np.asarray(jet_starts)
        times      = np.asarray(times)

        box_metric   = np.zeros((len(jets), len(jets)))
        time_metric  = np.zeros((len(jets), len(jets)))
        point_metric = np.zeros((len(jets), len(jets)))

        dtime = (times_all[-1] - times_all[0]).astype('timedelta64[s]').astype(float)

        for j, jetj in enumerate(jets):
            for k, jetk in enumerate(jets):
                if j==k:
                    point_metric[k,j] = 0
                    box_metric[k,j]   = 0
                    time_metric[k,j]  = 0
                elif jetj.subject==jetk.subject:
                    #print(j, k, jetj.subject, jetk.subject, jetj.start, jetk.start)
                    point_metric[k,j] = np.nan#jetj.subject*np.linalg.norm(jetj.start)*np.linalg.norm(jetk.start)
                    box_metric[k,j]   = np.nan
                    time_metric[k,j]  = np.nan
                else:
                    point_dist        = np.linalg.norm((jetj.start - jetk.start))
                    box_ious          = jetj.box.intersection(jetk.box).area/jetj.box.union(jetk.box).area
                    point_metric[k,j] = point_dist/np.mean([start_confidences[j], start_confidences[k]])
                    box_metric[k,j]   = 1. - box_ious

                    # we will limit to 2 frames (each frame is 5 min)
                    time_metric[k,j]  = np.abs( (times[j] - times[k]).astype('timedelta64[s]')\
                                               .astype(float))/(5*60+12)

        distance_metric = point_metric/np.percentile(point_metric[np.isfinite(point_metric)&(point_metric>0)], 90) + \
                            2.*box_metric


        distance_metric[~np.isfinite(distance_metric)] = np.nan

        indices  = np.arange(len(jets))
        labels   = -1.*np.ones(len(jets))
        subjects = np.asarray([jet.subject for jet in jets])

        print(f"Using eps={eps} and time_eps={time_eps*30} min")

        while len(indices) > 0:
            ind = indices[0]
            # this is the jet we will compare against
            j0 = jets[ind]

            # find all the jets that fall within a distance 
            # eps for this jet and those that are not 
            # already clustered into a jet
            mask = (distance_metric[ind,:] < eps)&(labels==-1)

            unique_subs = np.unique(subjects[mask])

            # make sure that all the jets belong to different subjects
            # two jets in the same subject should be treated differently
            if len(unique_subs) != sum(mask):
                # in this case, there are duplicates
                # we will choose the best subject from each duplicate
                count = [sum(subjects[mask]==subject) for subject in unique_subs]

                # loop through the unique subs
                for sub in unique_subs:
                    # find the indices that correspond to this
                    # jet in the mask
                    inds_sub = np.where((subjects==sub)&mask)[0]
                    # and the corresponding distances
                    dists    = distance_metric[ind,inds_sub]

                    # remove all the other subjects
                    mask[inds_sub] = False

                    # set the lowest distance index to True
                    mask[inds_sub[np.argmin(dists)]] = True

            # next make sure that there is a reachability in time
            # jets should be connected to each other to within 1-2 frames
            if sum(mask) > 1: # only do this when there are more than 1 jet
                rem_inds  = np.where(mask)[0]
                for j, indi in enumerate(rem_inds):
                    # if this is the first index we don't 
                    # have an idea of past reachability
                    if j==0: 
                        continue

                    # get the reachability in time
                    time_disti = time_metric[indi,mask]
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

                    # find the smallest interval between this jet and any other 
                    # jet. then remove this if it more than eps frames away
                    if time_disti[time_disti>0.].min() > time_eps:
                        mask[indi] = False
                

            # assign a new value to these
            labels[mask] = labels.max() + 1

            rem_inds = [np.where(indices==maski)[0][0] for maski in np.where(mask)[0]]

            indices = np.delete(indices, rem_inds)

        # get the list of jets found
        njets = len(np.unique(labels[labels > -1]))

        assert njets > 0, "No jet clusters found!"

        jet_clusters = []

        for j in range(njets):
            mask_j  = labels==j
            # subset the list of jets that correspond to this label
            jets_j  = jets[mask_j]
            times_j = times[mask_j]

            # for each jet, append the time information
            for k, jet in enumerate(jets_j):
                jet.time = times_j[k]

            clusteri = JetCluster(jets_j)

            jet_clusters.append(clusteri)

        return jet_clusters, distance_metric, point_metric, box_metric


class JetCluster:
    def __init__(self, jets):
        self.jets  = jets

    def create_gif(self, output):
        '''
            Create a gif of the jet objects showing the 
            image and the plots from the `Jet.plot()` method
        '''
        fig, ax = plt.subplots(1,1, dpi=250)
        
        # create a temp plot so that we can get a size estimate
        subject0 = self.jets[0].subject
        
        ax.imshow(get_subject_image(subject0, 0))
        ax.axis('off')
        fig.tight_layout(pad=0)

        # loop through the frames and plot
        ims = []
        for jet in tqdm.tqdm(self.jets):
            subject = jet.subject
            for i in range(15):
                img = get_subject_image(subject, i)

                # first, plot the image
                im1 = ax.imshow(img)

                # for each jet, plot all the details
                # and add each plot artist to the list
                jetims = jet.plot(ax)
                
                # combine all the plot artists together
                ims.append([im1, *jetims])

        # save the animation as a gif
        ani = animation.ArtistAnimation(fig, ims)
        ani.save(output, writer='imagemagick')

