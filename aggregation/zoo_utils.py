from panoptes_client import Subject
import imageio


def get_subject_image(subject, time=0.5):
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
    subject = Subject(int(subject))
    vid = imageio.get_reader(subject.locations[0]['video/mp4'], 'ffmpeg')
    num_frames = vid.count_frames()
    frameid = int(time * num_frames)
    img = vid.get_data(frameid)

    return img
