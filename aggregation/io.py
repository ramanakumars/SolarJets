import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
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
