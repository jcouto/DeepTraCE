#!/usr/bin/env python
# Install script for the DeepTraCE gui.
# Joao Couto - September 2022
import os
from os.path import join as pjoin
from setuptools import setup
from setuptools.command.install import install

requirements = []
with open("requirements.txt","r") as f:
    requirements = f.read().splitlines()
    
data_path = pjoin(os.path.expanduser('~'), 'DeepTraCE')

setup(
    name = 'DeepTraCE',
    version = '0.0.1',
    author = 'DeNardoLab',
    author_email = 'jpcouto@gmail.com',
    description = 'GUI for the DeepTraCE analysis of whole-brain light sheet microscopy ',
    long_description = '',
    long_description_content_type='text/markdown',
    license = 'GPL',
    install_requires = requirements,
    url = "https://github.com/DeNardoLab/DeepTraCE",
    packages = ['deeptrace'],
    entry_points = {
        'console_scripts': [
            'deeptrace = deeptrace.gui:main',
        ]
    },
)


