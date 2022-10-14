import subprocess as sub
import os
import sys
import json
from os.path import join as pjoin
from glob import glob
from natsort import natsorted
from os.path import join as pjoin
import numpy as np
from tifffile import TiffFile,imread,imsave
from multiprocessing import Pool
from functools import partial
from skimage import img_as_ubyte

deeptrace_path = pjoin(os.path.expanduser('~'), 'DeepTraCE')
deeptrace_preferences_file = pjoin(deeptrace_path,'DeepTraCE_preferences.json')

defaults = dict(trailmap = dict(path = pjoin(os.path.expanduser('~'),'trailmap'),
                                environment = 'trailmap_env',
                                models_folder = pjoin(deeptrace_path,'models')),
                fiji = dict(path = pjoin(os.path.expanduser('~'),'Fiji.app')),
                elastix = dict(path = pjoin(os.path.expanduser('~'),'Elastix')),
                downsample_factor = [0.40625,0.40625,0.3])


def get_preferences(prefpath = None):
    ''' Reads the user parameters from the home directory.

    pref = get_preferences(filename)

    '''
    if prefpath is None:
        prefpath = deeptrace_preferences_file
    preffolder = os.path.dirname(prefpath)
    if not os.path.exists(preffolder):
        os.makedirs(preffolder)
    if not os.path.isfile(prefpath):
        with open(prefpath, 'w') as outfile:
            json.dump(defaults, 
                      outfile, 
                      sort_keys = True, 
                      indent = 4)
            print('Saving default preferences to: ' + prefpath)
    with open(prefpath, 'r') as infile:
        pref = json.load(infile)
    for k in defaults:
        if not k in pref.keys():
            pref[k] = defaults[k] 
    return pref

deeptrace_preferences = get_preferences()


# to read files
class BrainStack():
    def __init__(self, ch0,ch1=None):
        self.ch0files = natsorted(glob(pjoin(ch0,'*.tif')))
        self.ch1files = []
        if not ch1 is None:
            self.ch1files = natsorted(glob(pjoin(ch1,'*.tif')))
        self.nframes = len(self.ch0files)
        self.ibuf = -1
        self.buf = None
        self.channel = 0
        self.output_rgb = False
    def get(self,i):
        if not self.ibuf == i: 
            if self.output_rgb and len(self.ch1files) :
                ch0 = TiffFile(self.ch0files[i]).pages.get(0).asarray()
                ch1 = TiffFile(self.ch1files[i]).pages.get(0).asarray()
                self.buf = np.stack([ch1,
                                     ch0,
                                     ch0*0]).transpose([1,2,0])
                
            elif self.channel == 0: 
                self.buf = TiffFile(self.ch0files[i]).pages.get(0).asarray()
            elif self.channel == 1: 
                self.buf = TiffFile(self.ch1files[i]).pages.get(0).asarray()
            else:
                self.buf = np.stack([TiffFile(self.ch0files[i]).pages.get(0).asarray(),
                                    TiffFile(self.ch1files[i]).pages.get(0).asarray()])
        return self.buf
    def __len__(self):
        return len(self.ch0files)
    
    def __getitem__(self,*args):
        index = args[0]
        if type(index) is tuple: # then look for 2 channels
            index = index[0]
        if type(index) is slice:
            idx1 = range(*index.indices(self.nframes))#start, index.stop, index.step)
        elif type(index) in [int,np.int32, np.int64]: # just a frame
            idx1 = [index]
        else: # np.array?
            idx1 = index
            
        img = []
        for i in idx1:
            img.append(self.get(i))
        return np.stack(img).squeeze()
    
def chunk_indices(nframes, chunksize = 512, min_chunk_size = 16):
    '''
    Gets chunk indices for iterating over an array in evenly sized chunks
    http://github.com/jcouto/wfield
    '''
    chunks = np.arange(0,nframes,chunksize,dtype = int)
    if (nframes - chunks[-1]) < min_chunk_size:
        chunks[-1] = nframes
    if not chunks[-1] == nframes:
        chunks = np.hstack([chunks,nframes])
    return [[chunks[i],chunks[i+1]] for i in range(len(chunks)-1)]

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from skimage import img_as_ubyte
from scipy import ndimage


def downsample_stack(stack,scales):
    downsampled = []
    with Pool() as pool:        
        for s,e in tqdm(chunk_indices(len(stack),256),desc='Downsampling stack.'):
            ss = stack[s:e]
            ss = img_as_ubyte(ss)
            downsampled.extend(pool.map(partial(ndimage.zoom, zoom=scales[:2]), [s for s in ss]))
    downsampled = np.stack(downsampled)
    return ndimage.zoom(downsampled,[scales[-1],1,1])

from scipy.ndimage import rotate
def rotate_stack(stack, anglex = 0, angley = 0, anglez = 0):
    
    if anglez != 0:
        tt = rotate(stack, angle = anglez, axes = [1,2], reshape = False)
        print(anglez)
    else:
        tt = stack.copy()
    if angley != 0.0:
        tt = rotate(tt, angle = angley, axes = [0,2], reshape = False)
    if anglex != 0.0:
        tt = rotate(tt, angle = anglex, axes = [0,1], reshape = False)
    return tt
