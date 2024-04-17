# Pystorm
Python implementation of various functions from the [Brainstorm](https://neuroimage.usc.edu/brainstorm/) [repository](https://github.com/brainstorm-tools/brainstorm3), often written for GPU compatibility and/or jit compilation.

## Note on the current scope of Pystorm
Pystorm does not handle interactions with Brainstorm or matlab and, in its current state, does not handle loading of data files. 

It mainly operates as a library of functions to perform operations on data directly (e.g., numpy array of sensor signals and imaging kernel). 

It aims to be lightweight, fast, and hopefully easily scalable to parallel computing and high performance computing.

It can be installed using pip through pypi where it is named **pystorm3**: 

`python -m pip install pystorm3` 

or

`pip install pystorm3`.

## Pystorm functions

### Currently implemented
- Power spectral density estimation ([PSD Welch method](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/timefreq/bst_psd.m), physical units only) in source and sensor space (can operate on GPU)
- Band-pass filtering (equivalent to ["bst-hfilter-2019"](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/math/bst_bandpass_hfilter.m))
- Hilbert transform
- Amplitude envelope correlation (equivalent to ["penv" and "oenv"](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/connectivity/bst_henv.m)). Can only be applied on signal directly for now: does not work with (sensor, kernel) input.

### Coming soon
- Sign flip for parcellated source signal
- [tPAC](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/process/functions/process_pac_dynamic.m) (time-resolved Phase-Amplitude Coupling) with [surrogate data z-score](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/process/functions/process_pac_dynamic_sur2.m)

### Currently not implemented
- Handling of unconstrained sources