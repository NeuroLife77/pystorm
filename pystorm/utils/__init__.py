"""
Utils module

Contains the complete set of utility functions that improve quality of life.

Currently contains: minitorch (also available as 'mnt')

Minitorch is already available through pystorm import (along with its mini-name 'mnt') so it is not necessary to import through here unless you want to import some of its functions directly without importing minitorch.
"""

__all__=[
        "minitorch",
        "time_series_utils",
        "get_sign_flip_mask",
        "get_scout_time_series",
]
from .time_series_utils import *
