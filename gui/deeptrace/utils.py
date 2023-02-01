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
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from skimage import img_as_ubyte
from scipy import ndimage
from scipy.ndimage import rotate

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
    def __init__(self, channel_folders ,extension = 'tif'):
        '''
        +++++++++++++++++++++++++
        BrainStack(ch0, ch1=None)
        +++++++++++++++++++++++++
        Access a set of tiffstack folders as if they were a numpy array.

        It expects one page per tif file; this can be adapted in the future
        
        Inputs:
           - ch0: string; a folder that contains tif files (usually the 488nm channel)
           - ch1: string; a folder that contains tif files (usually the 640nm channel)

        Outputs:
           - a BrainStack object that can be indexed line a numpy array
        (Warning: don't try to load the whole array to memory)


        Example:
        
        '''
        if type(channel_folders) is str:
            # then it must be made a list
            channel_folders = [channel_folders]
        self.channel_files = []
        self.channel_folders = []

        for folder in channel_folders:
            files = natsorted(glob(pjoin(folder,'*.' + extension)))
            if len(files):
                self.channel_files.append(files)
                self.channel_folders.append(folder)
            else:
                raise(OSError('[DeepTraCE] Could not find {0} files in folder {1}'.format(extension,folder)))
        sz = [len(a) for a in self.channel_files]
        if not np.sum(np.array(sz)==sz[0]) == len(sz):
            # then there is a size missmatch in the folders
            raise(OSError(
                '[DeepTraCE] There is a missmatch in the number of {0} files in the folder.'.format(
                    extension)))
        
        # this function expects one image per folder so the nframes is the number of files
        self.nframes = len(self.channel_files[0])
        self.nchannels = len(self.channel_files)
        self.ibuf = -1
        self.buf = None
        self.active_channels = [0]
        self.get(0) # load the buffer for the first frame
        self.dims = self.buf.squeeze().shape
        self.shape = [self.nframes,self.nchannels,*self.dims]
        self.active_channels = [i for i in range(self.nchannels)]

        # try to get one frame 
    def get(self,i):
        if not self.ibuf == i:
            d = []
            for ich in self.active_channels:
                d.append(TiffFile(self.channel_files[ich][i]).pages.get(0).asarray())
            self.buf = np.stack(d)
        return self.buf

    def __len__(self):
        return self.nframes
    
    def __getitem__(self,*args):
        index = args[0]
        if type(index) is tuple: # then look for nchannels
            ch = index[1]
            if type(index[1]) is slice:
                ch = list(range(*index[1].indices(self.nchannels)))
            self.set_active_channels(ch)
            index = index[0]
        if type(index) is slice:
            idx1 = range(*index.indices(self.nframes))
        elif type(index) in [int,np.int32, np.int64]: # just one frame
            idx1 = [index]
        else: # np.array?
            idx1 = index

        img = []
        for i in idx1:
            img.append(self.get(i))
        return np.stack(img).squeeze()

    def set_active_channels(self,channel_idx):
        '''
        Sets the current active channel.
        Use this to skip reading all channels when working with large datasets.
        '''
        if not type(channel_idx) is list:
            channel_idx = [channel_idx]
        self.active_channels = []
        for ichan in channel_idx:
            if ichan not in [i for i in range(self.nchannels)]:
                raise(ValueError('Channel {0} not available'.format(ichan)))
            self.active_channels.append(ichan)

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


def downsample_stack(stack,scales):
    downsampled = []
    with Pool() as pool:        
        for s,e in tqdm(chunk_indices(len(stack),256),desc='Downsampling stack.'):
            ss = stack[s:e]
            ss = img_as_ubyte(ss)
            downsampled.extend(pool.map(partial(ndimage.zoom, zoom=scales[:2]), [s for s in ss]))
    downsampled = np.stack(downsampled)
    return ndimage.zoom(downsampled,[scales[-1],1,1])


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

def frame_to_rgb(frame):
    '''
    Frame needs to have 3 dims (channel,H,W)
    '''
    tmp = []
    for i in range(3):
        if i > len(frame)-1:
            tmp.append(np.zeros_like(frame[0]))
        else:
            tmp.append(frame[i])
    tmp = np.stack(tmp).transpose(1,2,0)
    if tmp.dtype in [np.uint16]:
        tmp = img_as_ubyte(tmp).astype('uint8')
    return tmp 
