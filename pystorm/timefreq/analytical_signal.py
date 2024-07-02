from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from pystorm.signal_processing import band_pass
import pystorm.signal_processing.band_filter as bf
from pystorm import minitorch as mnt
from sys import stderr 
from numba import njit, objmode
from scipy.fft import fft as _spfft
from scipy.fft import ifft as _spifft
from scipy.fft import fftfreq as _spfftfreq
from numpy import concatenate as _npcat
from numpy import zeros as _npzeros
from numpy import arange as _nparange
from numpy import abs as _npabs
from pystorm.utils.time_series_utils import get_scout_time_series as _get_scout_time_series
__all__ = ["get_hilbert_torch","get_hilbert_scipy","hilbert","band_pass_hilbert","get_multiple_band_pass_hilbert","get_source_hilbert_torch"]

def get_hilbert_torch(
                        signal, fs : int,
                        pad_size = None,
                        window_mask = None,
                        device = "cpu",
                        return_numba_compatible = False,
                        return_torch = False, return_on_CPU = True
    ):
    """ This function applies the hilbert transformation using pytorch functions. Can be applied to signals of any shapes but only applies the transform over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            pad_size : float
                The size (in seconds) of the padding to add before applying the hilbert transform. 
            window_mask : str
                Specifies which part of the input signal to transform. [Only useful when applying windowed transform on signal that was previously filtered on its full length.]
            device: str
                Specifies the device in which to apply the filtering.
            return_numba_compatible: bool
                Specifies if the signal should be returned as numpy array (required for numba)
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            analytical_signal: numpy array (or torch tensor)                
                The analytical signal.
    """

    signal = ensure_torch(signal).to(device)
    
    if window_mask is not None: # Foreshadowing Amplitude Envelope Correlation implementation
        window_mask = ensure_torch(window_mask)
        windowed_signal = signal[..., window_mask]
    else:
        windowed_signal = signal
    
    if pad_size is not None:
        pad_size = int(pad_size*fs)
        padded_signal = mnt.cat([mnt.zeros(*signal.shape[:-1],pad_size, device=device),windowed_signal,mnt.zeros(*signal.shape[:-1],pad_size, device=device)], dim = -1).to(device)
    else:
        padded_signal = windowed_signal
        pad_size = 0
    
    recover_signal_mask = mnt.arange(windowed_signal.shape[-1])+pad_size
    freqs = mnt.fftfreq(padded_signal.shape[-1],d=1/fs)
    signal_fft = mnt.fft(padded_signal)
    signal_fft[...,freqs<0] = 0
    signal_fft[...,freqs>0] = signal_fft[...,freqs>0]*2
    analytical_signal = mnt.ifft(signal_fft)
    analytical_signal = analytical_signal[...,recover_signal_mask]
    if return_numba_compatible or not return_torch:
        return mnt.ensure_numpy(analytical_signal)
    return mnt.ensure_torch(analytical_signal, move_to_CPU = return_on_CPU)

def get_hilbert_scipy(
                        signal, fs : int,
                        pad_size = None,
                        window_mask = None,
                        return_numba_compatible = False,
                        return_torch = False
    ):
    """ This function applies the hilbert transformation using scipy functions. Can be applied to signals of any shapes but only applies the transform over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            pad_size : float
                The size (in seconds) of the padding to add before applying the hilbert transform. 
            window_mask : str
                Specifies which part of the input signal to transform. [Only useful when applying windowed transform on signal that was previously filtered on its full length.]
            return_numba_compatible: bool
                Specifies if the signal should be returned as numpy array (required for numba)
            return_torch: bool
                Specifies if the output should be a torch tensor.
        Returns: 
            analytical_signal: numpy array (or torch tensor)                
                The analytical signal.
    """

    signal = ensure_numpy(signal)
    
    if window_mask is not None: # Foreshadowing Amplitude Envelope Correlation implementation
        window_mask = ensure_numpy(window_mask)
        windowed_signal = signal[..., window_mask]
    else:
        windowed_signal = signal
    
    if pad_size is not None:
        pad_size = int(pad_size*fs)
        padded_signal = _npcat([_npzeros((*signal.shape[:-1],pad_size)),windowed_signal,_npzeros((*signal.shape[:-1],pad_size))], axis = -1)
    else:
        padded_signal = windowed_signal
        pad_size = 0
    
    recover_signal_mask = _nparange(windowed_signal.shape[-1])+pad_size
    freqs = _spfftfreq(padded_signal.shape[-1],d=1/fs)
    signal_fft = _spfft(padded_signal)
    signal_fft[...,freqs<0] = 0
    signal_fft[...,freqs>0] = signal_fft[...,freqs>0]*2
    analytical_signal = _spifft(signal_fft)
    analytical_signal = analytical_signal[...,recover_signal_mask]
    if return_numba_compatible or not return_torch:
        return ensure_numpy(analytical_signal)
    return ensure_torch(analytical_signal)

def hilbert(
                signal, fs : int,
                pad_size = None,
                window_mask = None,
                return_envelope = False,
                backend="torch", device = "cpu", return_torch = False, return_on_CPU = True
    ):
    """ This function applies the hilbert transformation. Can be applied to signals of any shapes but only applies the transform over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            pad_size : float
                The size (in seconds) of the padding to add before applying the hilbert transform. 
            window_mask : list/numpy array/torch tensor 
                Specifies which part of the input signal to transform. [Only useful when applying windowed transform on signal that was previously filtered on its full length.]
            return_envelope: bool
                Specifies whether to return only then envelope of the analytical signal.
            backend: str
                Specifies which backend to use. [Currently 'torch' and 'scipy' are available.]
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            analytical_signal: numpy array (or torch tensor)                 
                The analytical signal.
    """
    _return_torch_here = backend=="torch"
    if backend == "torch":
        signal = mnt.ensure_torch(signal)
        if device == "cuda" and signal.nelement() * signal.element_size() > mnt._check_available_memory():
            stderr.write(f'Resource Warning [band_pass()]: Your signal (of size {signal.nelement() * signal.element_size()*1e-6}MB) is too big to be moved to your GPU. Consider splitting the job into blocks. The process will likely crash now. \n')
        analytical_signal = get_hilbert_torch(signal, fs, pad_size = pad_size, window_mask=window_mask, device=device, return_torch=_return_torch_here, return_on_CPU=False)
    else: # For future use
        analytical_signal = get_hilbert_scipy(signal, fs, pad_size = pad_size, window_mask=window_mask, return_torch=_return_torch_here)
    if return_envelope:
        if return_torch:
            return ensure_torch(analytical_signal, move_to_CPU = return_on_CPU).abs()
        return _npabs(ensure_numpy(analytical_signal))
    if return_torch:
            return ensure_torch(analytical_signal, move_to_CPU = return_on_CPU)
    return ensure_numpy(analytical_signal)

def band_pass_hilbert(
                        signal, fs : int,
                        band, ripple = 60, width = 1.0,
                        keep_pad_percent_for_hilbert = 0.2,
                        return_envelope = False,
                        convolve_type = "auto",
                        return_filtered_signal = False,
                        backend = "torch", device="cpu",
                        verbose = 1, return_torch = False, return_on_CPU = True
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). 
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            return_envelope: bool
                Specifies whether to return only then envelope of the analytical signal.
            convolve_type : str
                Specifies which method to use for convolution.
            return_filtered_signal: bool
                Specifies if the filtered signal should be returned too
            backend: str
                Specifies which backend to use. [Currently 'torch' and 'scipy' are available.]
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            analytical_signal: numpy array (or torch tensor)               
                The analytical signal of the filtered signal.
            filtered_signal: numpy array (or torch tensor)               
                If return_filtered_signal=True
    """
    _return_torch_here = backend=="torch"
    filtered_signal, signal_mask = band_pass(
                                                signal,fs,
                                                band,ripple=ripple,width=width,
                                                keep_pad_percent=keep_pad_percent_for_hilbert,
                                                convolve_type=convolve_type,
                                                return_with_pad=True,
                                                backend="torch",device=device, # Forcince torch backend for band_pass until scipy version is fixed
                                                verbose = verbose, return_torch=_return_torch_here, return_on_CPU = False
                                    )
    if backend == "torch":
        signal = mnt.ensure_torch(signal)
        if device == "cuda" and signal.nelement() * signal.element_size() > mnt._check_available_memory():
            stderr.write(f'Resource Warning [band_pass()]: Your signal (of size {signal.nelement() * signal.element_size()*1e-6}MB) is too big to be moved to your GPU. Consider splitting the job into blocks. The process will likely crash now. \n')
        signal_mask = ensure_torch(signal_mask==1.0)
        pad_size = mnt.argwhere(signal_mask)[0]//fs
        analytical_signal = get_hilbert_torch(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device, return_torch=_return_torch_here, return_on_CPU = return_on_CPU)
    else: # For future use
        signal_mask = ensure_torch(signal_mask==1.0)
        pad_size = mnt.argwhere(signal_mask)[0]//fs
        analytical_signal = get_hilbert_scipy(signal, fs, pad_size = pad_size, window_mask=None, return_torch=True)

    
    if return_envelope and return_torch and not return_filtered_signal:
            return ensure_torch(analytical_signal, move_to_CPU = return_on_CPU).abs()
    if return_envelope and not return_torch and not return_filtered_signal:
        return _npabs(ensure_numpy(analytical_signal))
    if not return_envelope and return_torch and not return_filtered_signal:
            return ensure_torch(analytical_signal, move_to_CPU = return_on_CPU)
    if not return_envelope and not return_torch and not return_filtered_signal:
        return ensure_numpy(analytical_signal)

    if return_envelope and return_torch:
            return ensure_torch(analytical_signal, move_to_CPU = return_on_CPU).abs(), ensure_torch(filtered_signal, move_to_CPU = return_on_CPU), ensure_torch(signal_mask, move_to_CPU = return_on_CPU)
    if return_envelope and not return_torch:
        return _npabs(ensure_numpy(analytical_signal)), ensure_numpy(filtered_signal), ensure_numpy(signal_mask)
    if not return_envelope and return_torch:
            return ensure_torch(analytical_signal, move_to_CPU = return_on_CPU), ensure_torch(filtered_signal, move_to_CPU = return_on_CPU), ensure_torch(signal_mask, move_to_CPU = return_on_CPU)
    if not return_envelope and not return_torch:
        return ensure_numpy(analytical_signal), ensure_numpy(filtered_signal), ensure_numpy(signal_mask)

def get_multiple_band_pass_hilbert(
                        signal, fs : int,
                        bands, ripple = 60, width = 1.0,
                        keep_pad_percent_for_hilbert = 0.2,
                        return_envelope = False,
                        convolve_type = "auto",
                        return_filtered_signal = False,
                        backend = "torch", device="cpu",
                        verbose = 1, return_torch = False, return_on_CPU = True
    ):

    """ This function is a wrapper for applying the band pass filtering then hilbert transform on multiple bands. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). 
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            bands: list/numpy array/torch tensor 
                Contains a list of frequency ranges of each filters.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            return_envelope: bool
                Specifies whether to return only then envelope of the analytical signal.
            convolve_type : str
                Specifies which method to use for convolution.
            return_filtered_signal: bool
                Specifies if the filtered signal should be returned too
            backend: str
                Specifies which backend to use. [Currently 'torch' and 'scipy' are available.]
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            filtered_signals: numpy array (or torch tensor)  
                The (possibly still padded) filtered signals in each bands: if return_filtered_signal=True
            signal_mask: numpy array (or torch tensor)  
                The mask of the signals (specifying the location of the signal within the padded signal): if return_filtered_signal=True
            analytical_signal: numpy array (or torch tensor)               
                The analytical signal of the filtered signals in each bands
    """
    filtered_signals = []
    signal_masks = []
    analytical_signals = []
    for band in bands:
        output =  band_pass_hilbert(
                    signal,fs,band,
                    ripple = ripple, width = width, 
                    keep_pad_percent_for_hilbert = keep_pad_percent_for_hilbert,
                    convolve_type = convolve_type,  return_filtered_signal=return_filtered_signal, return_envelope = return_envelope, 
                    backend = backend, device=device, verbose = verbose, return_torch = True, return_on_CPU = False
        )
        if return_filtered_signal:
            analytical_signal, filtered_signal, signal_mask = output
            filtered_signals.append(filtered_signal.unsqueeze(0))
            signal_masks.append(signal_mask.unsqueeze(0))
        else:
            analytical_signal = output
        analytical_signals.append(analytical_signal.unsqueeze(0))
        
    if return_filtered_signal:
        filtered_signals = mnt.cat(filtered_signals, dim = 0)
        signal_masks = mnt.cat(signal_masks, dim = 0)
    analytical_signals = mnt.cat(analytical_signals, dim = 0)
    
    if return_torch and not return_filtered_signal:
            return ensure_torch(analytical_signals, move_to_CPU = return_on_CPU)
    if not return_torch and not return_filtered_signal:
        return ensure_numpy(analytical_signals)
    if return_torch:
            return ensure_torch(analytical_signals, move_to_CPU = return_on_CPU), ensure_torch(filtered_signals, move_to_CPU = return_on_CPU), ensure_torch(signal_masks, move_to_CPU = return_on_CPU)
    if not return_torch:
        return ensure_numpy(analytical_signals), ensure_numpy(filtered_signals), ensure_numpy(signal_masks)



def get_source_hilbert_torch(
                        kernels, signal, fs : int,
                        collapse_function = None,
                        pad_size = None,
                        window_mask = None,
                        device = "cpu",
                        return_numba_compatible = False,
                        return_torch = False, return_on_CPU = True, **kwargs
    ):
    """ This function applies the hilbert transformation on source signals using pytorch functions. It implements the extraction of parcel time series before computing the hilbert transform. Can be applied to signals of any shapes but only applies the transform over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            kernels: list/numpy array/torch tensor
                The kernel or list of kernels to project the sensor data onto the source space (or parcel-specific source space)
                    -If the collapse function is none the kernels should be of shape [nSources, nSensors].
                    -Otherwise it should be a list of kernels (or an array of objects containing kernels) of length k with each element in the list being a matrix of shape [nSource_in_parcel, nSensors]. Given the variable number of sources per parcel it cannot be a single tensor/array of numbers. It can be a list of matrices or it can be a tensor/array of objects which elements are the matrices.
            signal: list/numpy array/torch tensor 
                The sensor space signal
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            collapse_function: str
                The function used to collapse the multiple signals from all sources within a parcel into a single signal.
            pad_size : float
                The size (in seconds) of the padding to add before applying the hilbert transform. 
            window_mask : str
                Specifies which part of the input signal to transform. [Only useful when applying windowed transform on signal that was previously filtered on its full length.]
            device: str
                Specifies the device in which to apply the filtering.
            return_numba_compatible: bool
                Specifies if the signal should be returned as numpy array (required for numba)
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            kwargs: Expect to potentially receive the following additional arguments
                reference_PC: numpy array (or torch tensor) 
                    An array that aims to serve as reference to align the PCs across windows (or trials).
        Returns: 
            analytical_signal: numpy array (or torch tensor)                
                The analytical signal.
            reference_PC: numpy array (or torch tensor) 
                Will return an array that aims to serve as reference to align the PCs across windows (or trials). Only happens if there is no 'reference_PC' key in the kwargs dict. 
                    -In an iterative process (looping over time windows or trials): Collect the reference_PC on the first call of the function and pass it as keyword argument over the next function calls.
    """

    signal = mnt.ensure_torch(signal).to(device)
    
    if window_mask is not None: # Foreshadowing Amplitude Envelope Correlation implementation
        window_mask = mnt.ensure_torch(window_mask)
        windowed_signal = signal[..., window_mask]
    else:
        windowed_signal = signal


    windowed_signal_source = _get_scout_time_series(kernels, windowed_signal, collapse_function=collapse_function, device = device, **kwargs)
    if isinstance(windowed_signal_source, tuple):
        windowed_signal_source, reference_PC = windowed_signal_source
    if pad_size is not None:
        pad_size = int(pad_size*fs)
        padded_signal = mnt.cat([mnt.zeros(*windowed_signal_source.shape[:-1],pad_size, device=device),windowed_signal_source,mnt.zeros(*windowed_signal_source.shape[:-1],pad_size, device=device)], dim = -1).to(device)
    else:
        padded_signal = windowed_signal_source
        pad_size = 0
    
    recover_signal_mask = mnt.arange(windowed_signal_source.shape[-1])+pad_size
    freqs = mnt.fftfreq(padded_signal.shape[-1],d=1/fs)
    signal_fft = mnt.fft(padded_signal)
    signal_fft[...,freqs<0] = 0
    signal_fft[...,freqs>0] = signal_fft[...,freqs>0]*2
    analytical_signal = mnt.ifft(signal_fft)
    analytical_signal = analytical_signal[...,recover_signal_mask]
    try: 
        reference_PC = mnt.ensure_torch(reference_PC, move_to_CPU = return_on_CPU)
        if return_numba_compatible or not return_torch:
            return mnt.ensure_numpy(analytical_signal), reference_PC
        return mnt.ensure_torch(analytical_signal, move_to_CPU = return_on_CPU), reference_PC
    except:
        if return_numba_compatible or not return_torch:
            return mnt.ensure_numpy(analytical_signal)
        return mnt.ensure_torch(analytical_signal, move_to_CPU = return_on_CPU)


########################################- Numba compatible functions [Not used yet] - ########################################


# This is likely bloated from multiple functions that do the same thing but with different signal dimensions and floating point precisions, but that's my current best solution to this technical problem. 


@njit
def _numba_hilbert_torch_complex64_1d(signal, fs, pad_size=None, window_mask = None, device = "cpu"):
    with objmode(analytical_signal='complex64[:]'):
        analytical_signal = get_hilbert_torch(signal, fs, pad_size=pad_size, window_mask = window_mask, device = device, return_numba_compatible=True)
    return analytical_signal
@njit
def _numba_hilbert_torch_complex128_1d(signal, fs, pad_size=None, window_mask = None, device = "cpu"):
    with objmode(analytical_signal='complex128[:]'):
        analytical_signal = get_hilbert_torch(signal, fs, pad_size=pad_size, window_mask = window_mask, device = device, return_numba_compatible=True)
    return analytical_signal

@njit
def _numba_hilbert_torch_complex64_2d(signal, fs, pad_size=None, window_mask = None, device = "cpu"):
    with objmode(analytical_signal='complex64[:,:]'):
        analytical_signal = get_hilbert_torch(signal, fs, pad_size=pad_size, window_mask = window_mask, device = device, return_numba_compatible=True)
    return analytical_signal
@njit
def _numba_hilbert_torch_complex128_2d(signal, fs, pad_size=None, window_mask = None, device = "cpu"):
    with objmode(analytical_signal='complex128[:,:]'):
        analytical_signal = get_hilbert_torch(signal, fs, pad_size=pad_size, window_mask = window_mask, device = device, return_numba_compatible=True)
    return analytical_signal

@njit
def _numba_hilbert_torch_complex64_3d(signal, fs, pad_size=None, window_mask = None, device = "cpu"):
    with objmode(analytical_signal='complex64[:,:,:]'):
        analytical_signal = get_hilbert_torch(signal, fs, pad_size=pad_size, window_mask = window_mask, device = device, return_numba_compatible=True)
    return analytical_signal
@njit
def _numba_hilbert_torch_complex128_3d(signal, fs, pad_size=None, window_mask = None, device = "cpu"):
    with objmode(analytical_signal='complex128[:,:,:]'):
        analytical_signal = get_hilbert_torch(signal, fs, pad_size=pad_size, window_mask = window_mask, device = device, return_numba_compatible=True)
    return analytical_signal

@njit
def _numba_band_pass_hilbert_float64_1d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float64_1d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex128_1d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_hilbert_float64_2d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float64_2d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex128_2d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_hilbert_float64_3d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float64_3d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex128_3d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal



@njit
def _numba_band_pass_hilbert_float32_1d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float32_1d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex64_1d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_hilbert_float32_2d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float32_2d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex64_2d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_hilbert_float32_3d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float32_3d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex64_3d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal



####

@njit
def _numba_band_pass_window_hilbert_float64_1d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2, sliding_window_size = 5000,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float64_1d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex128_1d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_window_hilbert_float64_2d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2, sliding_window_size = 5000,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float64_2d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex128_2d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_window_hilbert_float64_3d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2, sliding_window_size = 5000,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float64_3d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex128_3d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal



@njit
def _numba_band_pass_window_hilbert_float32_1d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2, sliding_window_size = 5000,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float32_1d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex64_1d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_window_hilbert_float32_2d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2, sliding_window_size = 5000,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float32_2d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex64_2d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal

@njit
def _numba_band_pass_window_hilbert_float32_3d(
                                        signal,fs:int,
                                        band,ripple = 60, width = 1.0,
                                        keep_pad_percent_for_hilbert = 0.2, sliding_window_size = 5000,
                                        convolve_type = "auto",
                                        backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering then hilbert transform on the full signal. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1). Future implementation of a similar function for windowed hilbert transform on full-length filtered signal (e.g., for AEC implementation) will be distinct from this one.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = bf._numba_band_pass_float32_3d(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type,backend=backend,device=device)
    if backend == "torch":
        signal_mask = signal_mask==1.0
        pad_size = signal_mask.argmax()//fs
        analytical_signal = _numba_hilbert_torch_complex64_3d(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    return analytical_signal