import matplotlib.pyplot as plt
from panoptes_client import Subject
from skimage import transform, io
from matplotlib import animation


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


def create_gif(subject, outfile):
    '''
        Create a gif of the jet objects showing the
        image and the plots from the `Jet.plot()` method

        Inputs
        ------
        jets : list
            List of `Jet` objects corresponding to the same subject
    '''
    # create a temp plot so that we can get a size estimate
    fig, ax = plt.subplots(1, 1, dpi=150)
    im1 = ax.imshow(get_subject_image(subject, 0))
    ax.axis('off')
    fig.tight_layout()

    # loop through the frames and plot
    def animate(i):
        img = get_subject_image(subject, i)

        # plot the image
        #im1 = ax.imshow(img)
        im1.set_array(img)
        
        # combine all the plot artists together
        return [im1]

    # save the animation as a gif
    ani = animation.FuncAnimation(fig, animate, frames=15, interval=200, blit=True)
    ani.save(outfile, writer='imagemagick')

    plt.clf()
    plt.close('all')
