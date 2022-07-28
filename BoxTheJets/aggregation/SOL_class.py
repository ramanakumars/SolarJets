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
from aggregation import Aggregator

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
            
            
            
    
    
