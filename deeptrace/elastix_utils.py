from .utils import *

################################################################
######### Transform parameters passed to ELASTIX ###############
################################################################

elastixpar0 = '''//Affine Transformation - updated May 2012

// Description: affine, MI, ASGD

//ImageTypes
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")

//Components
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "AdaptiveStochasticGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "AffineTransform")

(ErodeMask "true" )

(NumberOfResolutions 4)

(HowToCombineTransforms "Compose")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "false")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false") 
(ShowExactMetricValue "false")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations 500 ) 

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins 32 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

//Number of spatial samples used to compute the mutual information in each resolution level:
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 3)
(UseRandomSampleRegion "false")
(NumberOfSpatialSamples 4000 )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)

//Order of B-Spline interpolation used in each resolution level:
(BSplineInterpolationOrder 3)

//Order of B-Spline interpolation used for applying the final deformation:
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

//SP: Param_A in each resolution level. a_k = a/(A+k+1)^alpha
(SP_A 20.0 )
'''

elastixpar1 = '''//Bspline Transformation - updated May 2012

//ImageTypes
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")

//Components
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "StandardGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "BSplineTransform")

(ErodeMask "false" )

(NumberOfResolutions 3)
(FinalGridSpacingInVoxels 25.000000 25.000000 25.000000)

(HowToCombineTransforms "Compose")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(ResultImageFormat "tiff")
//unsigned char gives issues when values are very close to the max range (i.e. for 255)
//(ResultImagePixelType "unsigned char")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false")
(ShowExactMetricValue "false")
(WriteDiffusionFiles "true")

// Option supported in elastix 4.1:
(UseFastAndLowMemoryVersion "true")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations 5000)

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins 32 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

//Number of spatial samples used to compute the mutual information in each resolution level:
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 1 )
(UseRandomSampleRegion "true")
(SampleRegionSize 150.0 150.0 150.0)
(NumberOfSpatialSamples 10000 )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)

//Order of B-Spline interpolation used in each resolution level:
(BSplineInterpolationOrder 3)

//Order of B-Spline interpolation used for applying the final deformation:
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

//SP: Param_a in each resolution level. a_k = a/(A+k+1)^alpha
(SP_a 10000.0 )

//SP: Param_A in each resolution level. a_k = a/(A+k+1)^alpha
(SP_A 100.0 )

//SP: Param_alpha in each resolution level. a_k = a/(A+k+1)^alpha
(SP_alpha 0.6 )
'''

def elastix_fit(stack,
                elastix_path = deeptrace_preferences['elastix']['path'],
                registration_template = deeptrace_preferences['elastix']['registration_template'],
                par0 = elastixpar0,
                par1 = elastixpar1,
                outpath = None,
                pbar = None):
    # TODO: Adjust so it takes a variable number of parameters.
    
    # make that it works with registration template being an array!
    if not type(registration_template) is str:
        raise(ValueError("[ELASTIX] - not working without a tif template. Pass a filename."))
    # First clean-up the folder. You can only run one brain at a time in a computer.
    shutil.rmtree(deeptrace_preferences['elastix']['temporary_folder'])
    stack_path = pjoin(deeptrace_preferences['elastix']['temporary_folder'],'temporary_brain.tif')
    create_folder_if_no_filepath(stack_path)
    # prepare the file
    imsave(stack_path,stack)
    # prepare the parameters
    
    p0 = pjoin(deeptrace_preferences['elastix']['temporary_folder'],'elastix_p0.txt')
    create_folder_if_no_filepath(p0)
    with open(p0,'w') as fd:
        fd.write(par0)
    p1 = pjoin(deeptrace_preferences['elastix']['temporary_folder'],'elastix_p1.txt')
    with open(p1,'w') as fd:
        fd.write(par1)
    if outpath is None:
        outpath = deeptrace_preferences['elastix']['temporary_folder']
    create_folder_if_no_filepath(outpath)
    if elastix_path is None:
        # assume that it is in path
        elastix_path = "elastix"
        
    elastixcmd = r'{elastix_path} -f "{template}" -m "{stack_path}" -out "{outpath}" -p "{p0}" -p "{p1}"'.format(
        elastix_path = elastix_path,
        template = registration_template,
        stack_path = stack_path,
        outpath = outpath,
        p0 = p0,
        p1 = p1)
    proc = sub.Popen(elastixcmd,
                 shell=True,
                 stdout=sub.PIPE)
                 # preexec_fn=os.setsid) # does not work on windows?
    if not pbar is None:
        pbar.set_description('Running elastix')
        pbar.reset()
    while True:
        out = proc.stdout.readline()
        ret = proc.poll()
        if not pbar is None:
            pbar.update(1)
        if ret is not None:
            break

    transformix_parameters_paths = (pjoin(outpath,'TransformParameters.0.txt'),
                                    pjoin(outpath,'TransformParameters.1.txt')) # needs to be both for transformix, copy these together
    # The output filename will depend on the transforms..
    # clip to uint8
    return np.clip(imread(pjoin(outpath,'result.1.tiff')),0,255).astype('uint8'), transformix_parameters_paths

def elastix_apply_transform(stack,transform_path,
                    elastix_path = deeptrace_preferences['elastix']['path'],
                    outpath = None,
                    pbar = None):
    '''
    transform path is a folder with 2 files
    '''
    
    # make that it works with registration template being an array!
    shutil.rmtree(deeptrace_preferences['elastix']['temporary_folder'])
    if not type(stack) is str:
        stack_path = pjoin(deeptrace_preferences['elastix']['temporary_folder'],'temporary_brain.tif')
        create_folder_if_no_filepath(stack_path)
        # prepare the file
        imsave(stack_path,stack)
    else:
        stack_path = stack
    transformfiles = glob(pjoin(transform_path,'TransformParameters.*'))
    if not len(transformfiles):
        raise(ValueError('Could not find transform files.'))
    for f in transformfiles:
        shutil.copyfile(f, pjoin(deeptrace_preferences['elastix']['temporary_folder'],os.path.basename(f)))
    transform_path = glob(pjoin(deeptrace_preferences['elastix']['temporary_folder'],'TransformParameters.1.*'))
    if len(transform_path):
        transform_path = transform_path[0]
    else:
        raise(ValueError('Could not find TransformParameters.1.txt'))
    if outpath is None:
        outpath = deeptrace_preferences['elastix']['temporary_folder']
    create_folder_if_no_filepath(outpath)
    if elastix_path is None:
        # assume that it is in path
        elastix_path = "transformix"
    else:
        elastix_path = elastix_path.replace('elastix',"transformix")
        
    elastixcmd = r'{elastix_path} -in "{stack}" -out "{outpath}" -tp "{t1}"'.format(
        elastix_path = elastix_path,
        stack = stack_path,
        outpath = outpath,
        t1 = transform_path)
    proc = sub.Popen(elastixcmd,
                 shell=True,
                 stdout=sub.PIPE)
                 #preexec_fn=os.setsid)  # does not work on windows
    if not pbar is None:
        pbar.set_description('Running transformix')
        pbar.reset()
    while True:
        out = proc.stdout.readline()
        ret = proc.poll()
        if not pbar is None:
            pbar.update(1)
        if ret is not None:
            break
    # clip to uint8 because output is Elastix was overflowing.
    return np.clip(imread(pjoin(outpath,'result.tiff')),0,255).astype('uint8')
