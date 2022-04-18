from panoptes_client import Workflow, SubjectSet, Subject
from astropy.io import ascii
from astropy.table import Table
from skimage import io
import numpy as np
from multiprocessing import Pool
import tqdm
import signal
import time

FETCH_FROM_PANOPTES = False

def initializer():
    '''
        Ignore CTRL+C in the worker process
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def get_subject_scale(subject_id):
    '''
        Get the scale for all frames for a given subject.
        Loads the subject from Panoptes and gets the image 
        sizes by directly opening the image
    '''
    try:
        # create the subject object 
        subject = Subject(subject_id)
        widths  = np.zeros(15)
        heights = np.zeros(15)
        
        # loop through the frames
        for frame in range(15):
            # get the image URL on panoptes
            try:
                frame_url = subject.raw['locations'][frame]['image/png']
            except KeyError:
                frame_url = subject.raw['locations'][frame]['image/jpeg']

            # read the image from the url 
            img = io.imread(frame_url)
            # and get its shape
            ny, nx, _ = img.shape

            widths[frame]  = nx
            heights[frame] = ny

        # the standard size is 1920x1440 so 
        # we will scale everything else to that size
        scale = widths/1920.

        # add this info to the table
        data = [int(subject.id), *scale]
        
        return data
    except Exception as e:
        return None

def get_scales_set(save=True):
    '''
        Process data for all subjects in the subject set
        and retrieve the corresponding scale wrt. the 1920x1440
        standard
    '''
    # create the column names and associated datatypes
    names  = ['subject_id']
    dtypes = ['i4']
    for i in range(15):
        names.append(f'frame_{i}_scale')
        dtypes.append('f4')

    # and initialize the empty table so we can add 
    # rows later on
    table = Table(names=names, dtype=dtypes)

    if FETCH_FROM_PANOPTES:
        workflow = Workflow(19650)
        subject_set = workflow.links.subject_sets[0]

        subjects = []
        for subject in subject_set:
            subjects.append(subject.id)
    else:
        subject_data = ascii.read('reductions/point_reducer_hdbscan_box_the_jets.csv')
        subjects     = list(np.unique(subject_data['subject_id']))

    # run this process in parallel since there is a lot of 
    # waiting for the API callback
    run = 0
    with Pool(initializer=initializer) as pool:
        print(f"Running with {pool._processes} threads")
        while len(subjects) > 0:
            print(f"Pass {run+1}")
            try:
                r = tqdm.tqdm(pool.imap_unordered(get_subject_scale, subjects), total=len(subjects))

                pool.close()

                ninpt = len(subjects)

                nfailed = 0
                for result in r:
                    # add the result to the table
                    if result is not None:
                        # the subject was downloaded successfully
                        # so we can extract the scales
                        table.add_row(result)

                        # remove this subject from the queue
                        subjects.remove(result[0])
                    else:
                        # there was an issue with the download
                        # we will queue this subject to try again
                        nfailed += 1
                    r.set_postfix({'errors': nfailed})

            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            except Exception as e:
                raise(e)

            pool.join()

            run += 1

    if save:
        table.write('configs/subject_scales.csv', format='csv', overwrite=True)
    
    return table

if __name__=='__main__':
    get_scales_set(False)
