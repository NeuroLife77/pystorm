![Pystorm logo](https://github.com/NeuroLife77/pystorm/blob/main/pystorm_logo.png?raw=true)
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
- Time-resolved PSD (same as above function, but returns all windows instead of the average)
- FFT (only implemented with pytorch backend) that returns either the complex values or the power and phase values.
- Band-pass filtering (equivalent to ["bst-hfilter-2019"](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/math/bst_bandpass_hfilter.m))
- Hilbert transform
- Amplitude envelope correlation (equivalent to ["penv" and "oenv"](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/connectivity/bst_henv.m)). Can only be applied on signal directly for now: does not work with (sensor, kernel) input.
- Sign flip for parcellated source signal
- PAC and [tPAC](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/process/functions/process_pac_dynamic.m) (time-resolved Phase-Amplitude Coupling) with [surrogate data z-score](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/process/functions/process_pac_dynamic_sur2.m)

### Coming (very) soon
- TBD

### Currently not implemented
- Handling of unconstrained sources