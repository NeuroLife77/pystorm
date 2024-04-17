from setuptools import setup, find_packages
from os.path import join as os_join

# Get the current version number from inside the module
with open(os_join('pystorm', 'version.py')) as version_file:
    exec(version_file.read())

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
   name='pystorm3',
   version=__version__,
   description='Python implementation of some Brainstorm functions',
   license="GPL-3.0 license",
   long_description=long_description,
   long_description_content_type = 'text/markdown',
   author='Dominic Boutet',
   author_email='dominic.boutet@mail.mcgill.ca',
   maintainer='Dominic Boutet',
   maintainer_email='dominic.boutet@mail.mcgill.ca',
   url='https://github.com/NeuroLife77/pystorm',
   packages=find_packages(),
   install_requires=[
                        'torch>=2.2.0',
                        'numpy>=1.26',
                        'scipy>=1.13.0',
                        'torchaudio>=2.2.0',
                        "numba>=0.59"
    ],
   python_requires='>=3.9',
   keywords = [
                    'neuroscience',
                    'neuroimaging',
                    'neural oscillations',
                    'time series analysis',
                    'time frequency analysis',
    ],
   classifiers=[
                'Development Status :: 1 - Planning',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Operating System :: Unix',
                'Programming Language :: Python',
   ]
   #scripts=[
   #        ]
)