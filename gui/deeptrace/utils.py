
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
import shutil

deeptrace_path = pjoin(os.path.expanduser('~'), 'DeepTraCE')
deeptrace_preferences_file = pjoin(deeptrace_path,'DeepTraCE_preferences.json')

defaults = dict(trailmap = dict(models_folder = pjoin(deeptrace_path,'models')),
                elastix = dict(path = None,
                               temporary_folder = pjoin(deeptrace_path,'temp'),
                               registration_template = pjoin(deeptrace_path, 'registration',
                                                             'average_template_lsfm_10_crop_flip.tif')),
                downsample_factor = [0.40625,0.40625,0.3])

def create_folder_if_no_filepath(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def get_preferences(prefpath = None):
    ''' Reads the user parameters from the home directory.

    pref = get_preferences(filename)

    '''
    if prefpath is None:
        prefpath = deeptrace_preferences_file
    create_folder_if_no_filepath(prefpath)        
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


def downsample_stack(stack,scales, convert = True, chunksize = 50, pbar=None):
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
            if convert:
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
                               analysis_path = None,
                               batch_size = 128,
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
       - segmented masks stored in a folder {model_name}_seg_{dataset_name}
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
    if analysis_path is None:
        analysis_path = os.path.dirname(input_folder)
    output_folder = pjoin(analysis_path,
                          "{0}_seg_{1}".format(
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
    chunks = np.vstack([chunks,[len(files)-input_dim + dim_offset,len(files) + dim_offset]])    # last chunk
    if not pbar is None:
        pbar.reset()
        pbar.total = len(chunks)
        pbar.set_description('[TRAILMAP] Segmenting...')
        
    def write_res(res,chunk):
        for i in range(dim_offset, input_dim - dim_offset):
            fname = os.path.basename(files[chunk[0]+i])
            out_fname = pjoin(output_folder,'seg_' + fname)
            imsave(out_fname,res[i])
    for ichunk,chunk in enumerate(chunks):
        # get data from stack
        arr = get_normalized_padded_input_array(files,chunk)
        # run the model
        res = trailmap_apply_model(model,arr,batch_size = batch_size, threshold = threshold)
        # save the array
        write_res(res,chunk)
        if not pbar is None:
            pbar.update(1)
    if not pbar is None:
        pbar.set_description('[TRAILMAP] Completed')
