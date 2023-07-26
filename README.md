## DeepTraCe

### Instalation

We recoment using a python distribution like [Anaconda Python](https://www.anaconda.com/).
[Trailmap](https://github.com/AlbertPun/TRAILMAP) uses [Tensorflow 2.1](https://www.tensorflow.org/) to segment brains and has been tested with Python 3.7; we include the necessary files for trailmap so no need to install it.


The installation steps assume you have installed Anaconda Python.

##### Installation steps:
1. clone the git repository: ``git clone <repository address>``
2. go into the repository folder with the terminal and create the conda environement ``conda env create -f conda-environment.yml``
3. run ``conda activate deeptrace`` followed by ``python setup.py develop``
The first time the package is imported it create a ``DeepTraCe`` folder (see below). You'll need to unzip the contents of the models and atlas to the ``DeepTraCE`` folder.



#### Note:

The first time deeptrace is imported, it will create a folder in the user folder called ``DeepTraCE`` (i.e. in ``C:\Users\USERNAME\DeepTraCE`` on windows, ``\home\USERNAME\DeepTraCE`` on linux or ``\Users\USERNAME\DeepTraCE`` on mac).
You can retrieve the path running ``import deeptrace;print(deeptrace.utils.deeptrace_path)``

The ``DeepTraCE`` folder contains:
- **DeepTraCE_preferences.json** - A file with the paths and preferences
- **models** - a folder with all the models to be ran in hdf5 format
- **registration** - a folder with the files required to register the brainbrains to the reference, this will include:
   - *average_template_lsfm_10_crop_flip.tif*   - reference brain
   - *annotation_10_lsfm_collapse_crop_flip_newf.nrrd* - annotated atlas
   - *aba_ontology.csv* - table with the region names
   - *model_selection.csv* - table to select which model gets used for each area

**Important** Download the models and the atlas from [here](https://drive.google.com/file/d/1-TpVhovErZYMHRs4vbum6FPOJ0JKTGTD/view?usp=sharing) and **unzip** it to the ``DeepTraCE`` folder.



### Analysis steps:

1) Select the stack (we tipically use 2 channels, 488 and 640 - the brains are registered to 488). Load the brains to the ``BrainStack`` - as in cell 1.
2) Correct the rotation angles of the brain. When you run the first cell, the 488 stack will be downsampled and opened on a GUI. **Use this gui to select 2 locations of the frame that are in the same plane.** To do this:
    - find the blood vessels in medial sections and place the mouse over a vessel. Press the ``z`` key to mark the first coordinates/plane.
    - Use the slider or the arrows to find a **more medial** plane that has the same vessel and place the mouse over the same vessel (x and y coords of the vessel are different from the first point). Press the ``shift z`` key combination to record the second point.
    
3) Once the points have been selected, run the second cell. That will run all segmentation models; register the stacks to the average. 
4) Run the third cell for combining the models and preparing the resuts.

All results are saved to a folder named ``deeptrace_analysis`` one level up from the raw stacks.

Examples in the [notebooks folder](notebooks).


