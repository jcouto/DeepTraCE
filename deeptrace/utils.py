
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
import pandas as pd

deeptrace_path = pjoin(os.path.expanduser('~'), 'DeepTraCE')
deeptrace_preferences_file = pjoin(deeptrace_path,'DeepTraCE_preferences.json')

defaults = dict(trailmap = dict(models_folder = pjoin(deeptrace_path,'models')),
                elastix = dict(path = None,
                               temporary_folder = pjoin(deeptrace_path,'temp'),
                               registration_template = pjoin(deeptrace_path, 'registration',
                                                             'average_template_lsfm_10_crop_flip.tif')),
                atlas = pjoin(deeptrace_path, 'registration',
                              'annotation_10_lsfm_collapse_crop_flip_newf.nrrd'),
                ontology = pjoin(deeptrace_path, 'registration','aba_ontology.csv'),
                model_selection = pjoin(deeptrace_path, 'registration','model_selection.csv'),
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

def _read_files_and_downsample(files,scales,convert = True):
    stack = []
    for f in files:
        s = read_ome_tif(f)
        if convert:
            s = img_as_ubyte(s)
        stack.append(ndimage.zoom(s, zoom=scales[:2]))
    return stack
    
    
def downsample_stack(stack,scales, convert = True, chunksize = 50, pbar=None,njobs = 8):
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
    
    Joao Couto - deeptrace 2023

    '''
    downsampled = []
    if not pbar is None:
        pbar.total = len(stack)+1
        
    with Pool(processes = njobs) as pool:
        if hasattr(stack,"active_channels"):
            #then run a pool on the files to read in parallel.
            files = np.array(stack.channel_files[stack.active_channels[0]])
            chunks = chunk_indices(len(files),chunksize)
            file_chunks = []
            for s,e in chunk_indices(len(stack),chunksize):
                file_chunks.append(files[s:e])
            from tqdm import tqdm
            downsampled = pool.map(partial(_read_files_and_downsample,scales = scales,convert = convert),tqdm(file_chunks,desc='Downsampling'))
            downsampled = np.concatenate(downsampled,axis=0)
            
        else:
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


def rotate_stack(stack, anglez = 0, angley = 0, anglex = 0, flip_x = False, flip_y = False):
    '''
    Rotate a stack in 3d.

    Joao Couto - deeptrace 2023
    '''
    
    if anglex != 0.0:
        tt = rotate(stack, angle = anglex, axes = [1,2], reshape = False)
    else:
        tt = stack.copy()
    if angley != 0.0:
        tt = rotate(tt, angle = angley, axes = [0,2], reshape = False)
    if anglez != 0.0:
        tt = rotate(tt, angle = anglez, axes = [0,1], reshape = False)
    if flip_x:
        tt = tt[:,:,::-1]
    if flip_y:
        tt = tt[:,::-1,:]
    return tt

def frame_to_rgb(frame):
    '''
    Frame needs to have 3 dims (channel,H,W)

    Joao Couto - deeptrace 2023
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

