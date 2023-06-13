from panoptes_client import Subject
from skimage import io, transform


def get_subject_image(subject, frame=7):
    '''
        Fetch the subject image from Panoptes (Zooniverse database)

        Inputs
        ------
        subject : int
            Zooniverse subject ID
        frame : int
            Frame to extract (between 0-14, default 7)

        Outputs
        -------
        img : numpy.ndarray
            RGB image corresponding to `frame`
    '''
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
