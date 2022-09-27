import subprocess as sub
import os
import sys
import json
from os.path import join as pjoin

deeptrace_path = pjoin(os.path.expanduser('~'), 'DeepTraCE')
deeptrace_preferences_file = pjoin(deeptrace_path,'DeepTraCE_preferences.json')

defaults = dict(trainmap = dict(path = pjoin(os.path.expanduser('~'),'trailmap'),
                                environment = 'trailmap_env',
                                model_folder = pjoin(deeptrace_path,'models')),
                fiji = dict(path = pjoin(os.path.expanduser('~'),'FIJI','fiji.app')))


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
