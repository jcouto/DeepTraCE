from .utils import *


class BrainStack():
    def __init__(self, channel_folders ,extension = 'tif',
                 analysis_folder = None,
                 downsample_suffix = 'scaled',
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
        self.pathdict = dict()
        for i,f in enumerate(self.channel_folders):
            self.pathdict['ch{0}folder'.format(i)] = os.path.abspath(f)
            self.pathdict['ch{0}'.format(i)] = os.path.basename(os.path.abspath(f))

        if analysis_folder is None:
            self.analysis_folder = pjoin('{ch0folder}','..','deeptrace_analysis','{ch0}')
        else:
            self.analysis_folder = analysis_folder
        self.analysis_folder = os.path.abspath(self.analysis_folder.format(**self.pathdict))
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
            self.downsample(channel_indices = [0], pbar = pbar)
        
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
                raise(ValueError('[DeepTraCE] - Channel {0} not available'.format(ichan)))
            self.active_channels.append(ichan)

    def downsample(self,scales = None, channel_indices = [], save = True, pbar = None):
        '''
        open the downsampled data (read from tif or save to tif)
        Control the saving directory with the downsample_suffix option of BrainStack
        Does not return anything - self.downsampled_data has the stacks.
        '''
        if pbar is None:
            pbar = self.pbar
        if scales is None:
            scales = deeptrace_preferences['downsample_factor']
        self.downsampled_stack = []
        # try to read it
        if not len(channel_indices):
            channel_indices = list(range(self.nchannels))
        downsample_files = []
        for ichan in channel_indices:
            folder = self.channel_folders[ichan]
            fname = os.path.basename(os.path.abspath(folder)) # in case there are slashes
            fname = pjoin(self.analysis_folder, self.downsample_suffix,'{0}.tif'.format(fname))
            downsample_files.append(fname)
            if os.path.exists(fname):
                # load it
                stack = imread(fname)
            else:
                print('Downsampling channel {0} to {1}.'.format(ichan,fname))
                if not pbar is None:
                    pbar.reset() # reset the bar
                    pbar.set_description('Downsampling stack for channel {0}'.format(ichan))
                    pbar.total = self.nframes
                self.set_active_channels(ichan)
                stack = downsample_stack(self,scales,pbar = pbar)
                # save it
                if save:
                    create_folder_if_no_filepath(fname)
                    imsave(fname,stack)
                    
            # saved stack
            self.downsampled_stack.append(stack)
        return downsample_files

    def deeptrace_analysis(self, angles = None,
                           flip_x = None,
                           flip_y = None,
                           scales = None,
                           trailmap_models = None,
                           trailmap_channel = None,
                           pbar = None):
        params = dict()
        fname = pjoin(self.analysis_folder,'deeptrace_parameters.json')
        if os.path.exists(fname):
            with open(fname,'r') as f:
                params = json.load(f)
        if trailmap_models is None:
            if 'trailmap_models' in params.keys():
                trailmap_models = params['trailmap_models']
            else:
                trailmap_models = trailmap_list_models()  # use all models
            
        if trailmap_channel is None:
            if 'trailmap_channel' in params.keys():
                trailmap_channel = params['trailmap_channel']
            else:
                trailmap_channel = 1

        if scales is None:
            scales = deeptrace_preferences['downsample_factor']
        if not flip_x is None:
            params['flip_x'] = flip_x
        if not flip_x is None:
            params['flip_y'] = flip_y
        if angles is None:
            if 'angles' in params.keys():
                angles = params['angles']
            else:
                print('[DeepTraCE] Rotation angles are not set, please select at least one angle.')
                if not len(self.downsampled_stack):
                    self.downsample(channel_indices = [0],
                                    pbar = pbar) # downsample the first channel
                from deeptrace.plotting import interact_find_angles
                res = interact_find_angles(self.downsampled_stack[0])
                return res
        params['trailmap_channel'] = trailmap_channel
        params['angles'] = angles
        if scales is None:
            params['scales'] = deeptrace_preferences['downsample_factor']
        trailmap_models = [t for t in trailmap_models] # convert to a list if not
        params['trailmap_models'] = trailmap_models
        params['analysis_folder'] = self.analysis_folder
        create_folder_if_no_filepath(fname)
        # Run trailmap models
        for model_path in trailmap_models:
            model_folder = pjoin(self.analysis_folder,
                                 "{0}_seg_{1}".format(
                                     os.path.splitext(os.path.basename(model_path))[0], 
                                     os.path.basename(os.path.abspath(self.channel_folders[trailmap_channel]))))
            if not 'trailmap_segmentation' in params.keys():
                params['trailmap_segmentation'] = []
            if os.path.isdir(model_folder):
                trailmapfiles = glob(pjoin(model_folder,'*.tif'))
                if not len(trailmapfiles) == len(self.channel_files[trailmap_channel]):
                    raise(ValueError("[DeepTraCE] - File missmatch. Need to recompute trailmap, delete the folder: {0}".format(model_folder)))
            else:
                trailmap_segment_tif_files(model_path,
                                           self.channel_files[trailmap_channel],
                                           analysis_path = self.analysis_folder,
                                           pbar = pbar)
            # todo: check if all the files are there.
            if not model_folder in params['trailmap_segmentation']:
                params['trailmap_segmentation'].append(model_folder)

        # downsample the first channel and all the model data
        folder =  pjoin(self.analysis_folder,'rotated')
        filename = pjoin(folder,'align_channel.tiff')
        create_folder_if_no_filepath(filename)
        # downsample the first channel
        if not os.path.exists(filename):
            print('[DeepTraCE] Downsampling the alignment channel to {0}'.format(filename))
            self.set_active_channels(0)
            stack = downsample_stack(self, scales, pbar = pbar)
            print('[DeepTraCE] Rotating the alignment channel')
            to_elastix = rotate_stack(stack,
                                      *angles,
                                      flip_x = params['flip_x'],
                                      flip_y = params['flip_y'])
            imsave(filename,to_elastix)
        else:
            to_elastix = imread(filename)
        # run elastix on the first channel
        aligned_file = pjoin(self.analysis_folder,'aligned.tiff')
        transform_path = pjoin(self.analysis_folder,'transformix')
        if not os.path.exists(transform_path):
            print('[DeepTraCE] Fitting the alignment channel with elastix')
            from .elastix_utils import elastix_fit
            aligned,transformpaths = elastix_fit(to_elastix, pbar = pbar)
            os.makedirs(transform_path,exist_ok = True)
            for f in transformpaths:
                shutil.copyfile(f, pjoin(transform_path,os.path.basename(f)))
            imsave(aligned_file, aligned)
        else:
            aligned = imread(aligned_file)
        
        # downsample all the model stacks and apply transforms
        from .elastix_utils import elastix_apply_transform
        if 'trailmap_segmentation' in params.keys():
            modeldata = BrainStack(params['trailmap_segmentation'])
            for ichan in range(len(params['trailmap_segmentation'])):
                # save downsampled and rotated
                modelname = os.path.basename(params['trailmap_segmentation'][ichan])
                filename = pjoin(folder,modelname+'.tiff')
                if not os.path.exists(filename):
                    print('[DeepTraCE] Downsampling the {0} model'.format(modelname))
                    modeldata.set_active_channels(ichan)
                    stack = downsample_stack(modeldata,scales,pbar = pbar, convert = True) # convert to uint8
                    print('[DeepTraCE] Rotating the model')
                    to_elastix = rotate_stack(stack,
                                              *angles,
                                              flip_x = params['flip_x'],
                                              flip_y = params['flip_y'])
                    imsave(filename,to_elastix)
                else:
                    to_elastix = imread(filename)

                aligned_file = pjoin(self.analysis_folder,modelname+'_aligned.tiff')
                if not os.path.exists(aligned_file):
                    aligned_res = elastix_apply_transform(to_elastix,
                                                          transform_path,
                                                          pbar = pbar)
                    imsave(aligned_file,aligned_res)

        # then apply transformix on raw images
        if self.nchannels>1:
            if not 'aligned_channels' in params.keys():
                params['aligned_channels'] = []
                for ichan in range(1,self.nchannels):
                    channelname = os.path.basename(os.path.abspath(self.channel_folders[ichan]))
                    if not os.path.exists(aligned_file):
                        filename = pjoin(folder,channelname+'.tiff')
                        self.set_active_channels(ichan)
                        stack = downsample_stack(self,scales,pbar = pbar, convert = True) # convert to uint8
                        print('[DeepTraCE] Rotating the {0}'.format(channelname))
                        to_elastix = rotate_stack(stack,
                                                  *angles,
                                                  flip_x = params['flip_x'],
                                                  flip_y = params['flip_y'])
                        imsave(filename,to_elastix)
                        aligned_file = pjoin(self.analysis_folder,channelname+'_aligned.tiff')
                        aligned_res = elastix_apply_transform(to_elastix,
                                                              transform_path,
                                                              pbar = pbar)
                        imsave(aligned_file,aligned_res)
                    params['aligned_channels'].append(channelname)
        # save the parameters
        with open(fname,'w') as f:
            json.dump(params,f,
                      sort_keys = True, 
                      indent = 4)

        return params
        
def trailmap_list_models():
    '''
    Lists trailmap models in the models_folder
    '''
    return natsorted(np.array(glob(
        pjoin(deeptrace_preferences['trailmap']['models_folder'],'*.hdf5'))))

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

    This is adapted from the trailmap examples.
    JC - 01/2023

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
        res = trailmap_apply_model(model, arr, batch_size = batch_size, threshold = threshold)
        # save the array
        write_res(res,chunk)
        if not pbar is None:
            pbar.update(1)
    if not pbar is None:
        pbar.set_description('[TRAILMAP] Completed')


from skimage.morphology import  skeletonize_3d
from skimage import measure
from multiprocessing import Pool

def skeletonize_multithreshold_uint8_stack(stack, nbins = 8):
    '''
    This runs skeletonize_3d for the threshold values in parallel. 
    Thin each connected component single-pixel wide skeleton for different thresholds (nbins).
    It needs a bit of RAM.
    skel_threshold is from 0-255

    JC - 03/2023
    '''
    thresh = ((np.linspace(3,nbins+2,nbins)-1.)/(nbins+2) * 255).astype('uint8')
    with Pool() as pool:
        bws = pool.map(skeletonize_3d,[(stack>th).astype(bool) for th in thresh])
    res = (np.stack(bws).astype(bool).T*np.arange(len(bws))).T.sum(axis = 0)
    return res

def refine_connected_components(stack,nbins = 8, skel_threshold = 5, area_threshold = 90):
    '''
    refined_components = refine_connected_components(stack,nbins = 8, skel_threshold = 5, area_threshold = 90)

    This:
     - runs skeletonize_3d for the "nbins" threshold values in parallel. 
     - thins each connected component single-pixel wide skeleton for different thresholds (nbins).  
     - combines all thresholds into a weighted average, giving more weight to the more restrictive bins.
     - finds connected components that have more than "skel_threshold" value
     - finally, removes connected components that have less than "area_threshold" pixels

    
    JC - 03/2023
    '''

    res = skeletonize_multithreshold_uint8_stack(stack,nbins = nbins)
    labels = measure.label(res>skel_threshold)
    regions = measure.regionprops(labels)
    nregions = list(filter(lambda x: x.area > area_threshold, regions))
    
    threshres = np.zeros_like(stack,dtype=bool)
    idx = []
    for r in nregions:
        idx.append(r.coords)
    idx = np.vstack(idx)
    threshres[idx[:,0],idx[:,1],idx[:,2]] = True
    res[~threshres] = 0
    return res # return the refined model.

def read_atlas(atlas_path = deeptrace_preferences['atlas'],
               ontology_path = deeptrace_preferences['ontology']):
    '''
    Loads the atlas and the ontology.

    atlas,ontology = read_atlas()
    
    JC - 03/2023
    '''
    import pandas as pd
    import nrrd
    ontology = pd.read_csv(ontology_path)

    atlas, header = nrrd.read(atlas_path)
    ontology = pd.read_csv(ontology_path)
    atlas = atlas.transpose([2,1,0])
    return atlas, ontology, header

def load_deeptrace_models(path):
    '''
    Loads aligned data to memory
    params, aligned, aligned_models, aligned_other = load_deeptrace_models()
    
    '''
    cfg_file = pjoin(path,'deeptrace_parameters.json')
    if not os.path.exists(cfg_file):
        raise(OSError('Path should be to a deeptrace analysis folder {0}'.format(cfg_file)))
    with open(cfg_file,'r') as f:
        params = json.load(f)
        
    aligned_file = pjoin(path,'aligned.tiff')
    if not os.path.exists(aligned_file):
        raise(OSError('[DeepTraCE] Analysis results not found in {0}'.format(path)))
    aligned = imread(aligned_file)
    aligned_models = []
    aligned_other = []
    if not 'trailmap_models' in params.keys():
        raise(OSError('[DeepTraCE] No models found in {0}'.format(path)))
    else:
        for m in params['trailmap_models']:
            modelname = os.path.splitext(os.path.basename(m))[0]
            files = glob(pjoin(path,'*'+modelname+'*aligned*tiff'))
            if len(files):
                aligned_models.append(imread(files[0]))
            else:
                raise(OSError('[DeepTraCE] No models {0} in {1}'.format(modelname,path)))
    if 'aligned_channels' in params.keys():
        for m in params['aligned_channels']:
            aligned_file = pjoin(path,'{0}_aligned.tiff'.format(m))
            if os.path.exists(aligned_file):
                aligned_other.append(imread(aligned_file))

    return params,aligned,aligned_models,aligned_other

def combine_models(models, default_model, model_correspondence=None, ontology = None, atlas = None):
    '''
    Combines multiple models in one.

    Models are refered by number - 1 based (first model is model 1).
    combined_model, model_correspondence = combine_models(models,model_correspondence, default_model, ontology, atlas)

    models: List of models
    default_model: which model to use by default
    model_correspondence: pandas dataframe with fields ("atlas_name" and "model") or a string to a cvs table
    ontology: the atlas ontology to use (default will load from the preferences)
    atlas: the atlas to use (default will load from the preferences)
    '''
    atlas = None
    ontology = None
    model_correspondence = None
    if atlas is None or ontology is None:
        atlas,ontology,header = read_atlas()
    if default_model is None:
        default_model = int(len(models)/2)+1
    # load from the preferences
    model_correspondence = load_model_selection(model_correspondence)
    # Make the combined model                     
    combined = models[default_model-1].copy()
    for i,o in model_correspondence.iterrows():
        if not o.model == default_model:
            for ii in o.atlas_ids:
                combined[atlas == ii] = models[o.model-1][atlas == ii]
    
    return combined, model_correspondence

def load_model_selection(model_correspondence  = None, atlas = None,ontology = None):
    if atlas is None or ontology is None:
        atlas,ontology,header = read_atlas()    
    if model_correspondence is None:
        model_correspondence = deeptrace_preferences['model_selection']    
    if type(model_correspondence) is str:
        model_correspondence = pd.read_csv(model_correspondence)
    # find out the unique atlas numbers
    unique_atlas_ids = np.unique(atlas)
    model_correspondence['atlas_ids'] = [[] for i in range(len(model_correspondence))]
    # find the atlas IDs from the model selections
    for i,m in model_correspondence.iterrows():
        for j,o in ontology.iterrows():
            if m.atlas_name.strip("'") in o['name']:
                if o['id'] in unique_atlas_ids:
                    m['atlas_ids'].append(o['id'])
    return model_correspondence
    
def count_labeling_density(refined_model, model_selection, ontology = None, atlas = None):

    if ontology is None or atlas is None:
        atlas, ontology, header = read_atlas()
    model_selection['volume_pixels'] = np.nan
    model_selection['count_pixels'] = np.nan
    for i,o in model_selection.iterrows():
        model_selection.loc[i,'volume_pixels'] = 0
        model_selection.loc[i,'count_pixels'] = 0
        for ii in model_selection.loc[i,'atlas_ids']:
            mask = (atlas==ii)
            model_selection.loc[i,'volume_pixels'] += np.sum(mask)
            model_selection.loc[i,'count_pixels'] +=  np.sum(refined_model[mask])
    model_selection['density'] = model_selection.count_pixels/model_selection.volume_pixels
    # get colors from the ontology
    model_selection['color'] = '#FFFFFF'
    for i,m in model_selection.iterrows():
        idx = np.where(model_selection.loc[i,'atlas_name'].strip("'") == ontology['name'])[0]
        c = ontology.iloc[idx[0]]['color_hex_triplet']
        if len(c) != 6:
            c += 'e' # some triplets are not 
        model_selection.loc[i,'color'] = '#' + c
    return model_selection
