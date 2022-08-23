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
from .workflow import Aggregator, Jet
from .workflow import get_subject_image, get_box_edges
from sklearn.cluster import OPTICS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from shapely.geometry import Polygon, Point
import hdbscan
import json
import tqdm

def remove_last_char(string):
    final=string[0:len(string)-1]
    return final

def json_export_list(clusters,output):
    '''
        export the list of JetCluster objects to the output.json file. 
        Inputs
            ------
            clusters : list
                list with JetCluster objects to be exported
            output : str
                name of the exported json file
    '''
    string='['
    for i in range(len(clusters)):
        string+='{ "ID": "'+str(clusters[i].ID)+'", "SOL": "'+str(clusters[i].SOL)+'", "obs_time":"'+str(clusters[i].obs_time)
        string+='", "Duration":'+str(clusters[i].Duration)  
        string+=', "Bx":'+str(clusters[i].Bx) + ', "std_Bx":'+str(clusters[i].std_Bx)
        string+= ', "By":'+str(clusters[i].By) + ', "std_By":'+str(clusters[i].std_By)
        string+= ', "Lat":'+str(clusters[i].Lat) + ', "Lon":'+str(clusters[i].Lon)
        if np.isnan(clusters[i].std_maxH[0]):
            std_H=''
        else:
            std_H=',"std_maxH":{ "upper":'+str(clusters[i].std_maxH[0])+',"lower":'+str(clusters[i].std_maxH[1])+'}'
        string+=', "Max_Height":'+str(clusters[i].Max_Height) +std_H
        string+=', "Width":'+str(clusters[i].Width) + ', "std_W":'+str(clusters[i].std_W)
        if np.isnan(clusters[i].Velocity):
            Vel=''
        else:
            Vel=', "Velocity":'+str(clusters[i].Velocity)
        string+= Vel + ', "sigma":'+str(clusters[i].sigma)     
        jetstring=', "Jets": ['
        for j in clusters[i].jets:
            jetstring+='{ "subject": '+str(j.subject)
            jetstring+=', "sigma": '+str(j.sigma)
            jetstring+=', "time": "'+str(j.time)
            jetstring+='", "start": { "x":'+str(j.start[0]) +',"y":'+str(j.start[1])+'}'
            jetstring+=', "end": { "x":'+str(j.end[0]) +',"y":'+str(j.end[1])+'}'
            jetstring+=', "cluster_values": { "x":'+str(j.cluster_values[0]) +',"y":'+str(j.cluster_values[1])+',"w":'+str(j.cluster_values[2])+',"h":'+str(j.cluster_values[3])+',"a":'+str(j.cluster_values[4])+'}'
            jetstring+= '},'
        subjson=remove_last_char(jetstring) 
        string+=subjson+']},'
    Json=remove_last_char(string)+']'
    
    text_file = open(f"{str(output)}.json", "w")

    text_file.write(Json)

    text_file.close()
    
    print(f'The {len(clusters)} JetCluster objects are exported to {output}.json.')
    
    return 


def json_import_list(input_file):
    '''
        import a list of JetCluster objects from the input_file file. 
        Inputs
            ------
            input_file : string
                path or filename to the json file with JetCluster objects
        Outputs
            ------
            clusters : list
                list of JetCluster objects 
    '''
    file = open(input_file)
    lists=json.load(file)
    file.close()
    clusters=np.array([])
    for k in range(len(lists)):
        json_obj=lists[k]
        jets_subjson=json_obj['Jets']
        jets_list=np.array([])
        for J in jets_subjson:
            subject=J['subject']
            best_start=np.array([J['start'][i] for i in ['x','y']])
            best_end=np.array([J['end'][i] for i in ['x','y']])
            jet_params=np.array([J['cluster_values'][i] for i in ['x','y','w','h','a']])
            jeti=Polygon(get_box_edges(*jet_params))
            jet_obj = Jet(subject, best_start, best_end, jeti, jet_params)
            jet_obj.time = J['time'] 
            jet_obj.sigma = J['sigma'] 
            jets_list=np.append(jets_list,jet_obj)
        cluster_obj = JetCluster(jets_list)
        cluster_obj.ID= json_obj['ID']
        cluster_obj.SOL= json_obj['SOL']
        cluster_obj.Duration= json_obj['Duration']
        cluster_obj.obs_time= json_obj['obs_time']
        cluster_obj.Bx= json_obj['Bx']
        cluster_obj.std_Bx= json_obj['std_Bx']
        cluster_obj.By= json_obj['By']
        cluster_obj.std_By= json_obj['std_By']
        cluster_obj.Lat= json_obj['Lat']
        cluster_obj.Lon= json_obj['Lon']
        cluster_obj.Max_Height= json_obj['Max_Height']
        try:
            cluster_obj.std_maxH= np.array([json_obj['std_maxH'][i] for i in ['upper','lower']])
        except:
            cluster_obj.std_maxH= np.array([np.nan,np.nan])

        cluster_obj.Width= json_obj['Width'] 
        cluster_obj.std_W= json_obj['std_W']
        cluster_obj.sigma= json_obj['sigma']
        try:
            cluster_obj.Velocity= json_obj['Velocity']
        except:
            cluster_obj.Velocity= np.nan  

        clusters=np.append(clusters,cluster_obj)
    
    print(f'The {len(clusters)} JetCluster objects are imported from {input_file}.')
    
    return clusters



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
        if task=='T0':
            fig = Image(filename=('JetOrNot/SOL/Agreement_SOL_'+task+'/'+SOL_event.replace(':','-')+'.png'))
        elif task=='T3':
            fig = Image(filename=('BoxTheJets/SOL/Agreement_SOL_'+task+'/'+SOL_event.replace(':','-')+'.png'))
        else:
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
    
    def get_filenames0(self, SOL_event):
        '''
        Get the filenames of the first image for each subjects. 
        filenames0: name of first image in the subject
        '''
        i=np.argwhere(self.SOL_small==SOL_event)[0][0]
        files=np.array(self.filenames0[i].split(' '))
        return files
    
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
        
        
    def event_box_plot(self, SOL_event): 
        '''
        Show the evolution of the box sizes of the different jets in one SOL event
        '''
        fig = Image(filename=('BoxTheJets/SOL/SOL_Box_size/'+SOL_event.replace(':','-')+'.png'))
        
        display(fig)
            
            
            
    def filter_jet_clusters(self, SOL_event, eps=1., time_eps=2.):
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
        
    def adding_new_attr(self, name_attr,value_attr):
        setattr(self, name_attr, value_attr)

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
        
    def json_export(self,output):
        '''
            export one single jet cluster to output.json file 
            Inputs
            ------
            output : str
                name of the exported json file
        '''
        json_export_list([self],output)
        
