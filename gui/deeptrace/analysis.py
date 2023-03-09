from .utils import *

# to read files and run the analysis
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
            if 'scales' in params.keys():
                scales = deeptrace_preferences['downsample_factor']
        
        if angles is None:
            if 'angles' in params.keys():
                angles = params['angles']
            else:
                print('[DeepTraCE] Rotation angles are not set, please select at least one angle.')
                if not len(self.downsampled_stack):
                    self.downsample(channel_indices = [0]) # downsample the first channel
                from deeptrace.plotting import interact_find_angles
                res = interact_find_angles(self.downsampled_stack[0])
                return res
        params['trailmap_channel'] = trailmap_channel
        params['angles'] = angles
        params['scales'] = scales
        params['trailmap_models'] = trailmap_models
        params['analysis_folder'] = self.analysis_folder
        create_folder_if_no_filepath(fname)
        with open(fname,'w') as f:
            json.dump(params,f,
                      sort_keys = True, 
                      indent = 4)
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
                                      *angles)
            imsave(filename,to_elastix)
        else:
            to_elastix = imread(filename)
        # run elastix on the first channel
        aligned_file = pjoin(self.analysis_folder,'aligned.tiff')
        transform_path = pjoin(self.analysis_folder,'transformix.txt')
        if not os.path.exists(transform_path):
            print('[DeepTraCE] Fitting the alignment channel with elastix')
            from .elastix_utils import elastix_fit
            aligned,transformpath = elastix_fit(to_elastix,pbar = pbar)
            shutil.copyfile(transformpath,transform_path)
            imsave(aligned_file,aligned)
        else:
            aligned = imread(aligned_file)
        
        # downsample all the model stacks and apply transforms
        if 'trailmap_segmentation' in params.keys():
            modeldata = BrainStack(params['trailmap_segmentation'])
            for ichan in range(len(params['trailmap_segmentation'])):
                # save downsampled and rotated
                modelname = os.path.basename(params['trailmap_segmentation'][ichan])
                filename = pjoin(folder,modelname+'.tiff')
                if not os.path.exists(filename):
                    print('[DeepTraCE] Downsampling the {0} model'.format(modelname))
                    modeldata.set_active_channels(ichan)
                    stack = downsample_stack(modeldata,scales,pbar = pbar)
                    print('[DeepTraCE] Rotating the model')
                    to_elastix = rotate_stack(stack,
                                              *angles)
                    imsave(filename,to_elastix)
                else:
                    to_elastix = imread(filename)
                from .elastix_utils import elastix_apply_transform
                aligned_file = pjoin(self.analysis_folder,modelname+'_aligned.tiff')
                if not os.path.exists(aligned_file):
                    aligned_res = elastix_apply_transform(to_elastix,transform_path,pbar = pbar)
                    imsave(aligned_file,aligned_res)
        
        print(params)
        
