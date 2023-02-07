
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
from scipy import ndimage
from scipy.ndimage import rotate
import sys

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

def read_ome_tif(file):
    return TiffFile(file).pages.get(0).asarray()

# to read files
class BrainStack():
    def __init__(self, channel_folders ,extension = 'tif',
                 downsample_suffix = pjoin('_scaled','10um.tif'),
                 downsample = False,
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
                d.append(read_ome_tif(self.channel_files[ich][i]))
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
        if pbar is None:
            pbar = self.pbar
        if scales is None:
            scales = deeptrace_preferences['downsample_factor']
        self.downsampled_data = []
        # try to read it
        for ichan in range(self.nchannels):
            folder = self.channel_folders[ichan]
            fname = os.path.abspath(folder) # in case there are slashes
            fname = fname + self.downsample_suffix
            if os.path.exists(fname):
                # load it
                stack = imread(fname)
            else:
                print('Downsampling channel {0}.'.format(ichan))
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
    if not pbar is None:
        pbar.total = len(stack)+1
    with Pool() as pool:        
        for s,e in chunk_indices(len(stack),chunksize):
            ss = stack[s:e]
            ss = img_as_ubyte(ss)
            downsampled.extend(pool.map(partial(ndimage.zoom, zoom=scales[:2]), [s for s in ss]))
            if not pbar is None:
                pbar.update(len(ss))
    downsampled = np.stack(downsampled)
    downsampled = ndimage.zoom(downsampled,[scales[-1],1,1])
    if not pbar is None:
        pbar.update(1)

    return downsampled


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

def trailmap_list_models():
    return glob(pjoin(deeptrace_preferences['trailmap']['models_folder'],'*.hdf5'))

def get_normalized_padded_input_array(files,chunk):
    arr = []
    for i in range(chunk[0],chunk[1]):
        if i < 0:
            arr.append(read_ome_tif(files[0]))
        elif i > len(files)-1:
            arr.append(read_ome_tif(files[len(files)-1]))
        else:
            arr.append(read_ome_tif(files[i]))
            
    return np.stack(arr).astype(np.float32)/(2**16 -1)
    
def trailmap_segment_tif_files(model_path, files,
                               batch_size = 15,
                               threshold = 0.01,
                               pbar = None):
    '''
    trailmap_segment_tif_files - Run a segmentation TRAILMAP model on a brain volume
    Inputs:
       - model_path: path to the hdf5 model to use
       - files: list of sorted files to use as input
       - batch_size: dictates the size of the batch that is loaded to the GPU (default 16 - increase if you have enough memory)
       - threshold: fragments threshold are not considered (default 0.01)
       - pbar: progress bar to monitor progress (default None)
    Outputs:
       - segmented masks stored in a folder {model_name}_seg-{dataset_name}
    Example:
    
files = stack.channel_files[1]
from tqdm.notebook import tqdm
pbar = tqdm()
models = trailmap_list_models()
model_path = models[4]
trailmap_segment_tif_files(model_path, files,pbar = pbar)    

    '''
    print('Loading TRAILMAP network and model.')
    from .trailmap_models import get_net,input_dim, output_dim,trailmap_apply_model
    model = get_net()
    model.load_weights(model_path)
    print('Using model weight {0}'.format(os.path.basename(model_path)))

    # create the output folders
    input_folder = os.path.dirname(os.path.abspath(files[0]))
    output_folder = pjoin(os.path.dirname(input_folder),
                          "{0}_seg-{1}".format(
                              os.path.splitext(os.path.basename(model_path))[0], 
                              os.path.basename(input_folder)))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print('Created {0}'.format(output_folder))

    # parse the data in chunks of size output_dim
    dim_offset = (input_dim - output_dim) // 2
    chunks = np.arange(-dim_offset,len(files)-input_dim+dim_offset,output_dim)
    chunks = np.zeros([2,len(chunks)])+chunks
    chunks = chunks.T.astype(int)
    chunks[:,1] += input_dim
    chunks += [len(files)-input_dim + dim_offset,len(files) + dim_offset]    # last chunk
    if not pbar is None:
        pbar.reset()
        pbar.total = len(chunks)
        pbar.set_description('[TRAILMAP] Segmenting...')
        
    def write_res(res,chunk):
        for i in range(dim_offset, input_dim - dim_offset):
            fname = os.path.basename(files[chunk[0]+i])
            out_fname = pjoin(output_folder,'seg-' + fname)
            imsave(out_fname,res[i])

    for ichunk,chunk in enumerate(chunks):
        # get data from stack
        arr = get_normalized_padded_input_array(files,chunk)
        # run the model
        res = trailmap_apply_model(model,arr)
        # save the array
        write_res(res,chunk)
        if not pbar is None:
            pbar.update(1)
    if not pbar is None:
        pbar.set_description('[TRAILMAP] Completed')
