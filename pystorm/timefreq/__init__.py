"""
TimeFreq module

Contains the complete set of functions that pertain to frequency and time-frequency analysis.

Currently contains: welch PSD functions and hilbert transform

Most of the important (wrapper) functions are imported directly into pystorm, but some utility functions, such as direct backend implementations, are present here.

In general it is not necessary to import through this module, but if you want to use a specific backend and want to avoid some overhead from multiple function calls it would be possilbe to import them directly through this module.
"""

__all__ = [
    "get_hilbert_torch","hilbert","band_pass_hilbert",
    "welch_psd_source_space", "welch_psd_sensor_space"
]
from .welch_psd import *
from .analytical_signal import *