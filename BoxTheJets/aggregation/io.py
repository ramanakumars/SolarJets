import json
import numpy as np
from shapely.geometry import Polygon
from .jet import Jet
from .jet_cluster import JetCluster
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import tqdm
from .zoo_utils import get_subject_image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return str(obj)
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
        cluster_obj.duration = json_obj['duration']
        cluster_obj.obs_time = np.datetime64(json_obj['obs_time'])

        cluster_obj.Bx = json_obj['Bx']['mean']
        cluster_obj.std_Bx = json_obj['Bx']['std']

        cluster_obj.By = json_obj['By']['mean']
        cluster_obj.std_By = json_obj['By']['std']

        cluster_obj.lat = json_obj['lat']
        cluster_obj.lon = json_obj['lon']

        cluster_obj.max_height = json_obj['max_height']['mean']
        try:
            cluster_obj.std_maxH = np.array(
                [json_obj['max_height'][i] for i in ['std_upper', 'std_lower']])
        except Exception as e:
            print(e)
            cluster_obj.std_maxH = np.array([np.nan, np.nan])

        cluster_obj.width = json_obj['width']['mean']
        cluster_obj.std_W = json_obj['width']['std']

        cluster_obj.sigma = json_obj['sigma']

        if 'velocity' in json_obj:
            cluster_obj.velocity = json_obj['velocity']
        else:
            cluster_obj.velocity = np.nan

        if 'flag' in json_obj:
            cluster_obj.flag = json_obj['flag']

        clusters.append(cluster_obj)

    clusters = np.asarray(clusters)

    print(f'The {len(clusters)} JetCluster objects are imported from {input_file}.')

    return clusters


def create_gif(jets):
    '''
        Create a gif of the jet objects showing the
        image and the plots from the `Jet.plot()` method

        Inputs
        ------
        jets : list
            List of `Jet` objects corresponding to the same subject
    '''
    # get the subject that the jet belongs to
    subject = jets[0].subject

    # create a temp plot so that we can get a size estimate
    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.imshow(get_subject_image(subject, 0))
    ax.axis('off')
    fig.tight_layout()

    # loop through the frames and plot
    ims = []
    for i in range(15):
        img = get_subject_image(subject, i)

        # first, plot the image
        im1 = ax.imshow(img)

        # for each jet, plot all the details
        # and add each plot artist to the list
        jetims = []
        for jet in jets:
            jetims.extend(jet.plot(ax, plot_sigma=False))

        # combine all the plot artists together
        ims.append([im1, *jetims])

    # save the animation as a gif
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(f'{subject}.gif', writer='imagemagick')


def create_metadata_jsonfile(filename: str, subjectstoloop: np.array, subjectsdata):
    '''
        Write out the metadata file for a given set of subjectstoloop to filename
        Inputs
        ------
        filename: str
            Filename to where the metadata should be written to
        subjectstoloop: np.array
            List of subjects for which the metadata should be gathered
        subjectsdata: astropy.table.table.Table
            data Table with Zooniverse metadata keys ['subject_id','metadata']

    '''

    # Select keys we want to write to json file
    keysToImport = [
        '#fits_names',  # First image filename in Zooniverse subject
        '#width',  # Width of the image in pixel
        '#height',  # Height of the image in pixel
        '#naxis1',  # Pixels along axis 1
        '#naxis2',  # Pixels along axis 2
        '#cunit1',  # Units of the coordinate increments along naxis1 e.g. arcsec
        '#cunit2',  # Units of the coordinate increments along naxis2 e.g. arcsec
        '#crval1',  # Coordinate value at reference point on naxis1
        '#crval2',  # Coordinate value at reference point on naxis2
        '#cdelt1',  # Spatial scale of pixels for naxis1, i.e. coordinate increment at reference point
        '#cdelt2',  # Spatial scale of pixels for naxis2, i.e. coordinate increment at reference point
        '#crpix1',  # Pixel coordinate at reference point naxis1
        '#crpix2',  # Pixel coordinate at reference point naxis2
        '#crota2',  # Rotation of the horizontal and vertical axes in degrees
        '#im_ll_x',  # Vertical distance in pixels between bottom left corner and start solar image
        '#im_ll_y',  # Horizontal distance in pixels between bottom left corner and start solar image
        '#im_ur_x',  # Vertical distance in pixels between bottom left corner and end solar image
        '#im_ur_y',  # Horizontal distance in pixels between bottom left corner and end solar image,
        '#sol_standard'  # SOL event that this subject corresponds to
    ]

    file = open(filename, 'w')
    file.write('[')
    for i, subject in enumerate(tqdm.tqdm(subjectstoloop, ascii=True, desc='Writing subjects to JSON')):
        subjectDict = {}
        subjectDict['subject_id'] = int(subject)
        subjectDict['data'] = create_subjectinfo(subject, subjectsdata, keysToImport)
        if i != len(subjectstoloop) - 1:
            file.write(json.dumps(subjectDict, indent=3) + ',')
        else:
            file.write(json.dumps(subjectDict, indent=3) + ']')
    file.close()
    print(' ')
    print("succesfully wrote subject information to file " + filename)
