import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .zoo_utils import get_subject_image
from .jet import Jet
import tqdm
from dataclasses import dataclass, field
import datetime


@dataclass
class JetCluster:
    jets: list[Jet]
    start_time: datetime.datetime = field(init=False)
    end_time: datetime.datetime = field(init=False)

    def __post_init__(self):
        '''
            Initiate the JetCluster with a list of jet objects that are contained by that cluster.
        '''
        self.start_time = self.jets[0].time_info['start']
        self.end_time = self.jets[-1].time_info['end']

    @classmethod
    def from_dict(cls, data):
        jets = []
        for jet_dict in data:
            jets.append(Jet.from_dict(jet_dict))

        return cls(jets)

    def to_dict(self):
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'jets': [jet.to_dict() for jet in self.jets]
        }

    def create_gif(self, output):
        '''
            Create a gif of the jet objects showing the
            image and the plots from the `Jet.plot()` method
        Inputs
        ------
            output: str
                name of the exported gif
        '''
        fig, ax = plt.subplots(1, 1, dpi=150)

        # create a temp plot so that we can get a size estimate
        subject0 = self.jets[0].subject

        im1 = ax.imshow(get_subject_image(subject0, 0))
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
        ani.save(output, writer='ffmpeg')
        plt.close('all')
