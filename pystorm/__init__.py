"""
Pystorm main module

Pystorm is a package that aims to provide a Python implementations for various functions from the Brainstorm repository, often written for GPU compatibility and/or jit compilation.

Pystorm does not handle interactions with Brainstorm or matlab and, in its current state, does not handle loading of data files. 

It mainly operates as a library of functions to perform operations on data directly (e.g., numpy array of sensor signals and imaging kernel). 

It aims to be lightweight, fast, and hopefully easily scalable to parallel computing and high performance computing.

This module contains the main set of function wrappers that are meant to be used directly. Other functions, such as direct backend implementations, must be imported through the sub-modules.

Currently contains: Welch PSD (sensor and source space), band pass filter, hilbert transform, minitorch (also available as 'mnt').
"""

from .utils import *
from .utils import minitorch as mnt
from .signal_processing import *
from .timefreq import *
from .connectivity import *

from .version import __version__