"""
Connectivity module

Contains the complete set of functions that pertain to connectivity metrics.

Currently contains: get_AEC

Most of the important (wrapper) functions are imported directly into pystorm, but some utility functions, such as direct backend implementations and functions that generate the filter's impulse response, are present here.

In general it is not necessary to import through this module, but if you want to use a specific backend and want to avoid some overhead from multiple function calls it would be possilbe to import them directly through this module.
"""

__all__ = ["get_AEC","get_source_AEC"]
from .amplitude_envelope_correlation import *