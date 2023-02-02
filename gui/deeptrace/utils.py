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

import sys
if hasattr(sys.modules["__main__"], "get_ipython"):
    from tqdm import notebook as tqdm
else:
    import tqdm

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
    def __init__(self, channel_folders ,extension = 'tif',
                 downsample_suffix = pjoin('_scaled','10um.tif'),
                 downsample = True,
                 pbar = None):
        '''
        +++++++++++++++++++++++++
        BrainStack(ch0, ch1=None)
        +++++++++++++++++++++++++
        Access a set of tiffstack folders as if they were a numpy array.

        It expects one page per tif file; this can be adapted in the future
        
        Inputs:
           - channel_folders: list of strings (folder paths that contains tif files - usually the 488nm channel and another for the 640nm stacks)
           - extension: tif, the extension used to read the raw data
           - downsample_suffix: suffix to add to the downsampled folder; default is _scalled/10um.tif
           - downsample: load or compute the dowsampled stacks (default: False)
           - pbar: progress bar to monitor progress.
        
        Outputs:
           - a BrainStack object that can be indexed line a numpy array
        (Warning: don't try to load the whole array to memory)

        Example:

        
        from deeptrace import BrainStack
        stack = BrainStack(channel_folders=['../../sampledata/210723_NAc326F_488_s3_0_8x_13-31-25/',
                   '../../sampledata/210723_NAc326F_640_s3_0_8x_11-50-51/'],
                   downsample = True)

        
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
        self.dims_original = self.buf.squeeze().shape
        self.shape = [self.nframes,self.nchannels,*self.dims_original]
        self.active_channels = [i for i in range(self.nchannels)]

        self.pbar = pbar
        # check for downsampled data
        self.downsample_suffix = downsample_suffix
        self.downsampled_stack = []
        if downsample:
            self.downsample(pbar = pbar)
        
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

    def downsample(self,scales = None,save = True,pbar = None):
        '''
        open the downsampled data (read from tif or save to tif)
        Control the saving directory with the downsample_suffix option of BrainStack
        Does not return anything - self.downsampled_data has the stacks.
        '''
        if scales is None:
            scales = deeptrace_preferences['downsample_factor']
        self.downsampled_data = []
        # try to read it
        for ichan in range(self.nchannels):
            folder = self.channel_folders[ichan]
            fname = os.path.abspath(folder) # in case there are slashes
            fname = pjoin(fname,self.downsample_suffix)
            if os.path.exists(fname):
                # load it
                stack = imread(fname)
            else:
                if not pbar is None:
                    pbar.reset() # reset the bar
                    pbar.set_description('Downsampling stack for channel {0}'.format(ichan))
                    pbar.total = self.nframes
                self.set_active_channels(ichan)
                stack = downsample_stack(self,scales,pbar = pbar)
                # save it
                if save:
                    if not os.path.exists(os.path.dirname(fname)):
                        os.makedirs(os.path.dirname(fname))
                    imsave(fname,stack)
            # saved stack
            self.downsampled_data.append(stack)
            

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


def run_trailmap_segment_brain_on_model(code_path, model_path, input_folders,trailmap_env):
    code = '''
import os
import sys
sys.path.append('{code_path}')
from inference import *
from models import *
import shutil

if __name__ == "__main__":
    input_batch = sys.argv[1:]
    # Verify each path is a directory
    for input_folder in input_batch:
        if not os.path.isdir(input_folder):
            raise Exception(input_folder + " is not a directory. Inputs must be a folder of files. Please refer to readme for more info")
    # Load the network
    weights_path = '{model_path}'

    model = get_net()
    model.load_weights(weights_path)

    from tqdm import tqdm
    for input_folder in tqdm(input_batch):
        # Remove trailing slashes
        input_folder = os.path.normpath(input_folder)
        # Output folder name
        output_name = "{model}_seg-" + os.path.basename(input_folder)
        output_dir = os.path.dirname(input_folder)
        output_folder = os.path.join(output_dir, output_name)
        # Create output directory. Overwrite if the directory exists
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten")
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        # Segment the brain
        print('The results will be stored in:'+output_f)
        segment_brain(input_folder, output_folder, model)
''' # this is from trailmap, if the environment could be the same (imagej would work) we could skip this

    tmpf = pjoin(deeptrace_path,'run_trailmap.py')
    with open(tmpf,'w') as fd:
        fd.write(code.format(model_path=model_path,
                             model=os.path.splitext(os.path.basename(model_path))[0],
                             code_path = code_path))
    import subprocess as sub
    cmd = r'cd {0} & conda activate {1} & python run_trailmap.py {2}'.format(
        deeptrace_path, trailmap_env, ' '.join(input_folders))
    print(cmd)
    #sub.call(cmd,shell = True)


def run_trailmap(folder):
    '''
    '''
    return

def downsample_stack(stack,scales,chunksize = 50, pbar=None):
    '''
    Downsample a stack

    Inputs:
       - stack, 3d stack or BrainStack (single channel)
       - scales [dx,dy,dz] look at deeptrace_preferences['downscale_factor'] for reference

    Outputs:
       - downsampled stack as uint8

    Example:
    
from tqdm.notebook import tqdm
scales = deeptrace_preferences['downsample_factor']
pbar = tqdm(total = len(stack))
stack.set_active_channels(0)
downsample_stack(stack,scales,pbar = pbar,chunksize = 256)
pbar.close()
    
    '''
    downsampled = []
    with Pool() as pool:        
        for s,e in chunk_indices(len(stack),chunksize):
            ss = stack[s:e]
            ss = img_as_ubyte(ss)
            downsampled.extend(pool.map(partial(ndimage.zoom, zoom=scales[:2]), [s for s in ss]))
            if not pbar is None:
                pbar.update(len(ss))
    downsampled = np.stack(downsampled)
    return ndimage.zoom(downsampled,[scales[-1],1,1])


def rotate_stack(stack, anglex = 0, angley = 0, anglez = 0):
    if anglez != 0:
        tt = rotate(stack, angle = anglez, axes = [1,2], reshape = False)
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
