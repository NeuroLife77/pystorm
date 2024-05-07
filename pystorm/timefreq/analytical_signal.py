from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from pystorm.signal_processing import band_pass
import pystorm.signal_processing.band_filter as bf
from pystorm import minitorch as mnt
from sys import stderr 
from numba import njit, objmode
__all__ = ["get_hilbert_torch","hilbert","band_pass_hilbert"]

def get_hilbert_torch(
                        signal, fs : int,
                        pad_size = None,
                        window_mask = None,
                        device = "cpu",
                        return_numba_compatible = False
    ):
    """ This function applies the hilbert transformation using pytorch functions. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
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
        Returns: 
            analytical_signal: torch tensor                       
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
    if return_numba_compatible:
        return mnt.ensure_numpy(analytical_signal)
    return analytical_signal

def hilbert(
                signal, fs : int,
                pad_size = None,
                window_mask = None,
                return_envelope = False,
                backend="torch", device = "cpu"
    ):
    """ This function applies the hilbert transformation. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
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
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal.
    """

    if backend == "torch":
        signal = mnt.ensure_torch(signal)
        if device == "cuda" and signal.nelement() * signal.element_size() > mnt._check_available_memory():
            stderr.write(f'Resource Warning [band_pass()]: Your signal (of size {signal.nelement() * signal.element_size()*1e-6}MB) is too big to be moved to your GPU. Consider splitting the job into blocks. The process will likely crash now. \n')
        analytical_signal = get_hilbert_torch(signal, fs, pad_size = pad_size, window_mask=window_mask, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    if return_envelope:
        return ensure_numpy(analytical_signal.abs())
    return ensure_numpy(analytical_signal)


def band_pass_hilbert(
                        signal, fs : int,
                        band, ripple = 60, width = 1.0,
                        keep_pad_percent_for_hilbert = 0.2,
                        return_envelope = False,
                        convolve_type = "auto",
                        backend = "torch", device="cpu",
                        verbose = 1
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
            return_envelope: bool
                Specifies whether to return only then envelope of the analytical signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
        Returns: 
            analytical_signal: numpy array                       
                The analytical signal of the filtered signal.
    """

    filtered_signal, signal_mask = band_pass(
                                                signal,fs,
                                                band,ripple=ripple,width=width,
                                                keep_pad_percent=keep_pad_percent_for_hilbert,
                                                convolve_type=convolve_type,
                                                return_with_pad=True,
                                                backend=backend,device=device,
                                                verbose = verbose
                                    )
    if backend == "torch":
        signal = mnt.ensure_torch(signal)
        if device == "cuda" and signal.nelement() * signal.element_size() > mnt._check_available_memory():
            stderr.write(f'Resource Warning [band_pass()]: Your signal (of size {signal.nelement() * signal.element_size()*1e-6}MB) is too big to be moved to your GPU. Consider splitting the job into blocks. The process will likely crash now. \n')
        signal_mask = ensure_torch(signal_mask==1.0)
        pad_size = mnt.argwhere(signal_mask)[0]//fs
        analytical_signal = get_hilbert_torch(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    if return_envelope:
        return ensure_numpy(analytical_signal.abs())
    return ensure_numpy(analytical_signal)


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