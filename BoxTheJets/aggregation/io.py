import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import tqdm
from .zoo_utils import get_subject_image
from .meta_file_handler import create_subjectinfo


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


def create_metadata_jsonfile(filename: str, subjects: np.array, subjectsdata):
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

    subject_data = []
    for i, subject in enumerate(tqdm.tqdm(subjects, ascii=True, desc='Writing subjects to JSON')):
        subjectDict = {}
        subjectDict['subject_id'] = int(subject)
        subjectDict['data'] = create_subjectinfo(subject, subjectsdata, keysToImport)
        subject_data.append(subjectDict)

    with open(filename, 'w') as outfile:
        json.dump(subject_data, outfile, cls=NpEncoder)

    print(f"Wrote subject information to {filename}")
