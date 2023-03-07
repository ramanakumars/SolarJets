import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from dateutil.parser import parse
import matplotlib.animation as animation
from .workflow import Jet
from .workflow import get_subject_image, get_box_edges
from shapely.geometry import Polygon
import json
import tqdm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def json_export_list(clusters, output):
    '''
        export the list of JetCluster objects to the output.json file.
        Inputs
            ------
            clusters : list
                list with JetCluster objects to be exported
            output : str
                name of the exported json file
    '''

    outdata = []
    for cluster in clusters:
        ci = {}

        ci['id'] = cluster.ID
        ci['SOL'] = cluster.SOL
        ci['obs_time'] = str(cluster.obs_time)
        ci['duration'] = cluster.Duration

        ci['lat'] = cluster.Lat
        ci['lon'] = cluster.Lon

        ci['Bx'] = {'mean': cluster.Bx, 'std': cluster.std_Bx}
        ci['By'] = {'mean': cluster.By, 'std': cluster.std_By}

        ci['max_height'] = {'mean': cluster.Max_Height,
                            'std_upper': cluster.std_maxH[0],
                            'std_lower': cluster.std_maxH[1]}

        ci['width'] = {'mean': cluster.Width, 'std': cluster.std_W}
        ci['height'] = {'mean': cluster.Height, 'std': cluster.std_H}

        ci['velocity'] = cluster.Velocity

        ci['sigma'] = cluster.sigma

        if hasattr(cluster, 'flag'):
            ci['flag'] = cluster.flag

        ci['jets'] = []
        for jet in cluster.jets:
            ji = {}

            ji['subject'] = jet.subject
            ji['sigma'] = jet.sigma
            ji['time'] = str(jet.time)

            # these are in solar coordinates
            ji['solar_H'] = jet.solar_H
            ji['solar_H_sig'] = {
                'upper': jet.solar_H_sig[0], 'lower': jet.solar_H_sig[1]}
            ji['solar_W'] = jet.solar_W
            ji['solar_start'] = {
                'x': jet.solar_start[0], 'y': jet.solar_start[1]}
            ji['solar_end'] = {'x': jet.solar_end[0], 'y': jet.solar_end[1]}

            # these are in the frame of the image not in solar coords
            ji['start'] = {'x': jet.start[0], 'y': jet.start[1]}
            ji['end'] = {'x': jet.end[0], 'y': jet.end[1]}

            ji['cluster_values'] = {'x': jet.cluster_values[0],
                                    'y': jet.cluster_values[1],
                                    'w': jet.cluster_values[2],
                                    'h': jet.cluster_values[3],
                                    'a': jet.cluster_values[4]}

            ci['jets'].append(ji)
        outdata.append(ci)

    with open(f"{str(output)}.json", "w") as outfile:
        json.dump(outdata, outfile, cls=NpEncoder)

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
    with open(input_file, 'r') as file:
        lists = json.load(file)

    clusters = []

    for k in range(len(lists)):
        json_obj = lists[k]
        jets_subjson = json_obj['jets']

        jets_list = []

        for J in jets_subjson:
            subject = J['subject']
            best_start = np.array([J['start'][i] for i in ['x', 'y']])
            best_end = np.array([J['end'][i] for i in ['x', 'y']])
            jet_params = np.array([J['cluster_values'][i]
                                  for i in ['x', 'y', 'w', 'h', 'a']])
            jeti = Polygon(get_box_edges(*jet_params))
            jet_obj = Jet(subject, best_start, best_end, jeti, jet_params)
            jet_obj.time = np.datetime64(J['time'])
            jet_obj.sigma = J['sigma']
            jet_obj.solar_H = J['solar_H']
            jet_obj.solar_H_sig = np.array(
                [J['solar_H_sig'][i] for i in ['upper', 'lower']])
            jet_obj.solar_W = J['solar_W']
            jet_obj.solar_start = np.array(
                [J['solar_start'][i] for i in ['x', 'y']])
            jet_obj.solar_end = np.array(
                [J['solar_end'][i] for i in ['x', 'y']])
            jets_list.append(jet_obj)

        jets_list = np.asarray(jets_list)

        cluster_obj = JetCluster(jets_list)
        cluster_obj.ID = json_obj['id']
        cluster_obj.SOL = json_obj['SOL']
        cluster_obj.Duration = json_obj['duration']
        cluster_obj.obs_time = np.datetime64(json_obj['obs_time'])

        cluster_obj.Bx = json_obj['Bx']['mean']
        cluster_obj.std_Bx = json_obj['Bx']['std']

        cluster_obj.By = json_obj['By']['mean']
        cluster_obj.std_By = json_obj['By']['std']

        cluster_obj.Lat = json_obj['lat']
        cluster_obj.Lon = json_obj['lon']

        cluster_obj.Max_Height = json_obj['max_height']['mean']
        try:
            cluster_obj.std_maxH = np.array(
                [json_obj['max_height'][i] for i in ['std_upper', 'std_lower']])
        except Exception as e:
            print(e)
            cluster_obj.std_maxH = np.array([np.nan, np.nan])

        cluster_obj.Width = json_obj['width']['mean']
        cluster_obj.std_W = json_obj['width']['std']

        cluster_obj.sigma = json_obj['sigma']

        if 'velocity' in json_obj:
            cluster_obj.Velocity = json_obj['velocity']
        else:
            cluster_obj.Velocity = np.nan

        if 'flag' in json_obj:
            cluster_obj.flag = json_obj['flag']

        clusters.append(cluster_obj)

    clusters = np.asarray(clusters)

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
            np.loadtxt(SOL_stats_file, delimiter=',', unpack=True, dtype=str)
        self.aggregator = aggregator

    def event_bar_plot(self, SOL_event, task='Tc'):
        '''
            Show the bar plot, indicating locations of jets for a given SOL event
            Produced by SOL_analytics.ipynb
            Inputs
            ------
                SOL_event: str
                    name of the SOL event used in Zooniverse
                task : str
                    the task key (from Zooniverse)
                    default Tc (combined results of task T0 and T3)
        '''
        if task == 'T0':
            fig = Image(filename=('JetOrNot/SOL/Agreement_SOL_' +
                        task + '/' + SOL_event.replace(':', '-') + '.png'))
        elif task == 'T3':
            fig = Image(filename=('BoxTheJets/SOL/Agreement_SOL_' +
                        task + '/' + SOL_event.replace(':', '-') + '.png'))
        else:
            fig = Image(filename=('SOL/Agreement_SOL_' + task +
                        '/' + SOL_event.replace(':', '-') + '.png'))

        display(fig)

    def get_subjects(self, SOL_event):
        '''
        Get the subjects that correspond to a given SOL event
        Inputs
        ------
            SOL_event: str
                name of the SOL event used in Zooniverse

        Outputs
        -------
            subjects : np.array
                list of the subjects in the SOL event

        Read in using
        SOL_small,SOL_subjects,times,Num,start,end,notes=np.loadtxt('path/SOL/SOL_{}_stats.csv'.format('Tc'),delimiter=',',unpack=True,dtype=str)
        Num=Num.astype(float)
        '''
        i = np.argwhere(self.SOL_small == SOL_event)[0][0]
        subjects = np.fromstring(self.SOL_subjects[i], dtype=int, sep=' ')

        return subjects

    def get_obs_time(self, SOL_event):
        '''
        Get the observation times of a given SOL event
        Inputs
        ------
            SOL_event: str
                name of the SOL event used in Zooniverse

        Outputs
        -------
            obs_time : starting time of the subjects of the SOL event

        saved in SOL_Tc_stats.csv
        '''
        i = np.argwhere(self.SOL_small == SOL_event)[0][0]
        T = [a + 'T' + b for a,
             b in zip(self.times[i].split(' ')[::2], self.times[i].split(' ')[1::2])]
        obs_time = np.array([parse(T[t])
                            for t in range(len(T))], dtype='datetime64')
        return obs_time

    def plot_subjects(self, SOL_event):
        '''
        Plot all the subjects with aggregation data of a given SOL event
        Inputs
        ------
            SOL_event: str
                name of the SOL event used in Zooniverse
        '''
        subjects = self.get_subjects(SOL_event)

        for subject in subjects:
            # check to make sure that these subjects had classification
            subject_rows = self.aggregator.points_data[:
                                                       ][self.aggregator.points_data['subject_id'] == subject]
            nsubjects = len(subject_rows['data.frame0.T1_tool0_points_x'])
            if nsubjects > 0:
                self.aggregator.plot_frame_info(subject, task='T1')

    def get_start_end_time(self, SOL_event):
        '''
        Get the start and end times of subjects in given SOL event

        Inputs
        ------
            SOL_event : str
                name of the SOL event used in Zooniverse
        Output
        ------
            start_time : np.array(dtype=datetime64)
                start times of the subjects
            end_time : np.array(dtype=datetime64)
                end time of the subjects
        saved in SOL_Tc_stats.csv
        '''
        i = np.argwhere(self.SOL_small == SOL_event)[0][0]
        S = [a + 'T' + b for a,
             b in zip(self.start[i].split(' ')[::2], self.start[i].split(' ')[1::2])]
        start_time = np.array([parse(S[t])
                              for t in range(len(S))], dtype='datetime64')
        E = [a + 'T' + b for a,
             b in zip(self.end[i].split(' ')[::2], self.end[i].split(' ')[1::2])]
        end_time = np.array([parse(E[t])
                            for t in range(len(E))], dtype='datetime64')
        return start_time, end_time

    def get_filenames0(self, SOL_event):
        '''
        Get the filenames of the first image for each subjects.
        Inputs
        ------
            SOL_event : str
                name of the SOL event used in Zooniverse
        Output
        ------
            files : np.array
                get an array of the filenames of the subject in the SOL event
        '''
        i = np.argwhere(self.SOL_small == SOL_event)[0][0]
        files = np.array(self.filenames0[i].split(' '))
        return files

    def get_notes_time(self, SOL_event):
        '''
        Get the notes of jet event (sequential jet subjects) in given SOL event

        Inputs
        ------
            SOL_event : str
                name of the SOL event used in Zooniverse

        Outputs
        -------

            notes_time: str
                flags given to subjects, revised after jet clusters are formed
                100 means an event of less than 6 minutes
                010 means an event where 2 event are closely after eachother
                saved in SOL_Tc_stats.csv
        '''
        i = np.argwhere(self.SOL_small == SOL_event)[0][0]
        flag = np.array(self.notes[i].split(' ')[1::3])
        N = [a + 'T' + b for a,
             b in zip(self.notes[i].split(' ')[2::3], self.notes[i].split(' ')[3::3])]
        notes_time = np.array([parse(N[t])
                              for t in range(len(N))], dtype='datetime64')
        return notes_time, flag

    def event_box_plot(self, SOL_event):
        '''
        Show the evolution of the box sizes of the different jets in one SOL event
        Inputs
        ------
            SOL_event : str
                name of the SOL event used in Zooniverse
        '''
        fig = Image(filename=('BoxTheJets/SOL/SOL_Box_size/' +
                    SOL_event.replace(':', '-') + '.png'))

        display(fig)

    def filter_jet_clusters(self, SOL_event, eps=1., time_eps=2.):
        '''
        For the inputted SOL event search for jet objects that are within the eps in space and the time_eps in time from eachother.
        Cluster those together and make JetCluster objects.
        Inputs
        ------
            SOL_event : str
                name of the SOL event used in Zooniverse
            eps : float
                space parameter in which the jets should lie
            time_eps : float
                time parameter in which the jets should lie
        '''

        # first, get a list of subjects for
        # this event
        subjects = self.get_subjects(SOL_event)
        times_all = self.get_obs_time(SOL_event)

        event_jets = []
        jet_starts = []
        times = []
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
                    start_dist.extend(np.linalg.norm(
                        jet.get_extract_starts() - jet.start, axis=0))

                start_confidences.extend(start_dist)
                times.extend([times_all[j] for n in range(len(jets))])

            except (ValueError, IndexError):
                continue

        jets = np.asarray(event_jets)
        jet_starts = np.asarray(jet_starts)
        times = np.asarray(times)

        box_metric = np.zeros((len(jets), len(jets)))
        time_metric = np.zeros((len(jets), len(jets)))
        point_metric = np.zeros((len(jets), len(jets)))

        for j, jetj in enumerate(jets):
            for k, jetk in enumerate(jets):
                if j == k:
                    point_metric[k, j] = 0
                    box_metric[k, j] = 0
                    time_metric[k, j] = 0
                elif jetj.subject == jetk.subject:
                    point_metric[k, j] = np.nan
                    box_metric[k, j] = np.nan
                    time_metric[k, j] = np.nan
                else:
                    point_dist = np.linalg.norm((jetj.start - jetk.start))
                    box_ious = jetj.box.intersection(
                        jetk.box).area / jetj.box.union(jetk.box).area
                    point_metric[k, j] = point_dist / \
                        np.mean([start_confidences[j], start_confidences[k]])
                    box_metric[k, j] = 1. - box_ious

                    # we will limit to 2 frames (each frame is 5 min)
                    time_metric[k, j] = np.abs((times[j] - times[k]).astype('timedelta64[s]')
                                               .astype(float)) / (5 * 60 + 12)

        distance_metric = point_metric / np.percentile(point_metric[np.isfinite(point_metric) & (point_metric > 0)], 90) + \
            2. * box_metric

        distance_metric[~np.isfinite(distance_metric)] = np.nan

        indices = np.arange(len(jets))
        labels = -1. * np.ones(len(jets))
        subjects = np.asarray([jet.subject for jet in jets])

        print(f"Using eps={eps} and time_eps={time_eps*30} min")

        while len(indices) > 0:
            ind = indices[0]

            # find all the jets that fall within a distance
            # eps for this jet and those that are not
            # already clustered into a jet
            mask = (distance_metric[ind, :] < eps) & (labels == -1)

            unique_subs = np.unique(subjects[mask])

            # make sure that all the jets belong to different subjects
            # two jets in the same subject should be treated differently
            if len(unique_subs) != sum(mask):
                # in this case, there are duplicates
                # we will choose the best subject from each duplicate

                # loop through the unique subs
                for sub in unique_subs:
                    # find the indices that correspond to this
                    # jet in the mask
                    inds_sub = np.where((subjects == sub) & mask)[0]
                    # and the corresponding distances
                    dists = distance_metric[ind, inds_sub]

                    # remove all the other subjects
                    mask[inds_sub] = False

                    # set the lowest distance index to True
                    mask[inds_sub[np.argmin(dists)]] = True

            # next make sure that there is a reachability in time
            # jets should be connected to each other to within 1-2 frames
            if sum(mask) > 1:  # only do this when there are more than 1 jet
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
            jets_j = jets[mask_j]
            times_j = times[mask_j]

            # for each jet, append the time information
            for k, jet in enumerate(jets_j):
                jet.time = times_j[k]

            clusteri = JetCluster(jets_j)

            jet_clusters.append(clusteri)

        return jet_clusters, distance_metric, point_metric, box_metric


class JetCluster:
    def __init__(self, jets):
        '''
            Initiate the JetCluster with a list of jet objects that are contained by that cluster.
        '''
        self.jets = jets

    def adding_new_attr(self, name_attr, value_attr):
        '''
            Add new attributes to the JetCluster
        Inputs
        ------
            name_attr: str
                name of the to be added property
            value_attr: any
                value of the to be added property
        '''
        setattr(self, name_attr, value_attr)

    def create_gif(self, output):
        '''
            Create a gif of the jet objects showing the
            image and the plots from the `Jet.plot()` method
        Inputs
        ------
            output: str
                name of the exported gif
        '''
        fig, ax = plt.subplots(1, 1, dpi=250)

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
                jetims = jet.plot(ax, plot_sigma=False)

                # combine all the plot artists together
                ims.append([im1, *jetims])

        # save the animation as a gif
        ani = animation.ArtistAnimation(fig, ims)
        ani.save(output, writer='imagemagick')

    def json_export(self, output):
        '''
            export one single jet cluster to output.json file
            Inputs
            ------
            output : str
                name of the exported json file
        '''
        json_export_list([self], output)
