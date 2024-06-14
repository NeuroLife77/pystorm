from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from scipy.signal import firwin as _firwin
from math import ceil as _ceil 
from pystorm import mnt
from scipy.signal import convolve as _sp_convolve
from sys import stderr 
from numba import njit, objmode
__all__ = ["get_fir_window","band_pass_torchaudio","band_pass"]

def get_firwin(ntaps, centered_band, beta, fs):
    return _firwin(ntaps, centered_band, window=("kaiser", beta), scale = True, pass_zero=False,fs=fs)

def get_fir_window(band, ripple:float, width:float, fs:int):
    """ This function computes the impulse response of the band-pass filter to get the exact same filter as bst-hfilter-2019 in brainstorm.
    
        Args: 
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            fs : int
                Sampling rate of the signal.
        Keyword Args: 
            None
        Returns: 
            window:                          
                The impulse response of the filter.


    Shamelessly stolen from 
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/fir_filter_design.py#L85
    and 
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/fir_filter_design.py#L29 
    """
    band = ensure_numpy(band)
    width_norm = width/(0.5*fs)
    a = abs(ripple)  
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    beta = round(beta,4)
    numtaps = (a - 7.95) / 2.285 / (mnt.pi * width_norm) + 1
    ntaps = int(_ceil(numtaps))
    ntaps = ntaps + (1-ntaps%2)
    centered_band = ensure_numpy([1e-5,fs//2 - 1])
    if band[0] is not None:
        centered_band[0] = band[0]-width/2
    if band[1] is not None:
        centered_band[1] = band[1]+width/2

    window = get_firwin(ntaps, centered_band,beta, fs)
    return window



def band_pass_torchaudio(
                            signal, win, fs:int,
                            return_pad = 0.2,
                            convolve_type = "auto", device="cpu",
                            verbose = 1, return_numba_compatible = False, return_torch = False, return_on_CPU = True
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response using torchaudio's convolve or fftconvolve functions. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            win: list/numpy array/torch tensor 
                The impulse response of the filter.
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            return_pad : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
            return_numba_compatible: bool
                Specifies whether to combine the (possibly still padded) filtered signal and the mask of the signal (specifying the location of the signal within the padded signal) into a single array which is required when calling this function through a numba-compiled function. [useful now]
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            filtered_signal: numpy array (or torch tensor)                  
                The (possibly still padded) filtered signal.
            signal_mask: numpy array (or torch tensor) 
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """

    signal = ensure_torch(signal).to(device)
    win = ensure_torch(win)
    
    if convolve_type == "fft" or (convolve_type == "auto" and len(win)*signal.shape[-1]>1e5 and abs(len(win)-signal.shape[-1])>1e2):
        convolve = mnt.fftconvolve
    else:
        convolve = mnt.convolve

    win = win.to(device)

    E = win[win.shape[-1]//2:] ** 2
    E = E.cumsum(dim=-1)
    E = E / E.amax()
    iE99 = ((E-0.99).abs().argmin() / fs).item()
    edge_percent = 2*iE99 / (signal.shape[-1]/fs)
    if edge_percent>0.1:
        if verbose > 0:
            stderr.write(f"Data Warning [band_pass()]: Start up and end transients represent {round(edge_percent*100,2)}% of your data. \n")

    pad_size = win.shape[0]//2
    mean_centered_signal = signal
    if len(mean_centered_signal.shape)<2:
        mean_centered_signal = mean_centered_signal.unsqueeze(0)
    signal_size = mean_centered_signal.shape[-1]
    mean_signal = mean_centered_signal.mean(-1, keepdims=True)
    mean_centered_signal = mean_centered_signal - mean_signal
    padded_signal = mnt.cat([mnt.zeros(*mean_centered_signal.shape[:-1],pad_size, device=device),mean_centered_signal,mnt.zeros(*mean_centered_signal.shape[:-1],pad_size, device=device)], dim = -1).to(device)
    if len(padded_signal.shape)<2:
        padded_signal = padded_signal.unsqueeze(0)
    if return_pad is not None:
        if return_pad > 1:
            return_pad = 1/return_pad
        signal_start = 2*int(pad_size)
        remainin_pad_size = int(pad_size*return_pad)
        remaining_margin_start = signal_start-remainin_pad_size
        remaining_margin_end = signal_start + signal_size + remainin_pad_size
    else:
        signal_start = 2*int(pad_size)
        remainin_pad_size = 2*int(pad_size)
        remaining_margin_start = None
        remaining_margin_end = None
    win = win.view(*[1 for _ in range(len(padded_signal.shape[:-1]))],-1)
    if win.dtype != padded_signal.dtype:
        win = ensure_torch(win,type_float=True)
        padded_signal = ensure_torch(padded_signal,type_float=True)
    filtered_signal = convolve(padded_signal,win, mode = "full")[...,remaining_margin_start:remaining_margin_end]
    signal_mask = mnt.zeros(filtered_signal.shape[-1], dtype=int)
    
    signal_end = signal_size+remainin_pad_size
    signal_indices = mnt.arange(remainin_pad_size,signal_end, dtype=int, device=device) 
    signal_mask[signal_indices] = 1
    if len(signal.shape)<2:
        filtered_signal = filtered_signal
    
    if return_numba_compatible: # For future use
        return_val = mnt.cat([filtered_signal,signal_mask.unsqueeze(0)], dim = 0).cpu()
        return ensure_numpy(return_val)
    else:
        if return_torch:
            return ensure_torch(filtered_signal, move_to_CPU = return_on_CPU), ensure_torch(signal_mask, move_to_CPU = return_on_CPU)
        return ensure_numpy(filtered_signal), ensure_numpy(signal_mask)
    

## TODO Rewrite the code, there seems to be an issue with the filtering here.
def band_pass_scipy(
                            signal, win, fs:int,
                            return_pad = 0.2,
                            convolve_type = "auto",
                            verbose = 1, return_numba_compatible = False, return_torch = False
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response using torchaudio's convolve or fftconvolve functions. Can be applied to signals of any shapes but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            win: list/numpy array/torch tensor 
                The impulse response of the filter.
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            return_pad : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
            return_numba_compatible: bool
                Specifies whether to combine the (possibly still padded) filtered signal and the mask of the signal (specifying the location of the signal within the padded signal) into a single array which is required when calling this function through a numba-compiled function. [not useful yet]
            return_torch: bool
                Specifies if the output should be a torch tensor.
        Returns: 
            filtered_signal: numpy array (or torch tensor)             
                The (possibly still padded) filtered signal.
            signal_mask: numpy array (or torch tensor) 
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    device = "cpu"
    signal = ensure_torch(signal)
    win = ensure_torch(win)
    
    
    convolve = _sp_convolve

    win = win.to(device)

    E = win[win.shape[-1]//2:] ** 2
    E = E.cumsum(dim=-1)
    E = E / E.amax()
    iE99 = ((E-0.99).abs().argmin() / fs).item()
    edge_percent = 2*iE99 / (signal.shape[-1]/fs)
    if edge_percent>0.1:
        if verbose > 0:
            stderr.write(f"Data Warning [band_pass()]: Start up and end transients represent {round(edge_percent*100,2)}% of your data. \n")

    pad_size = win.shape[0]//2
    mean_centered_signal = signal
    if len(mean_centered_signal.shape)<2:
        mean_centered_signal = mean_centered_signal.unsqueeze(0)
    signal_size = mean_centered_signal.shape[-1]
    mean_signal = mean_centered_signal.mean(-1, keepdims=True)
    mean_centered_signal = mean_centered_signal - mean_signal
    padded_signal = mnt.cat([mnt.zeros(*mean_centered_signal.shape[:-1],pad_size, device=device),mean_centered_signal,mnt.zeros(*mean_centered_signal.shape[:-1],pad_size, device=device)], dim = -1).to(device)
    if len(padded_signal.shape)<2:
        padded_signal = padded_signal.unsqueeze(0)
    if return_pad is not None:
        if return_pad > 1:
            return_pad = 1/return_pad
        signal_start = 2*int(pad_size)
        remainin_pad_size = int(pad_size*return_pad)
        remaining_margin_start = signal_start-remainin_pad_size
        remaining_margin_end = signal_start + signal_size + remainin_pad_size
    else:
        signal_start = 2*int(pad_size)
        remainin_pad_size = 2*int(pad_size)
        remaining_margin_start = None
        remaining_margin_end = None
    win = win.view(*[1 for _ in range(len(padded_signal.shape[:-1]))],-1)
    if win.dtype != padded_signal.dtype:
        win = ensure_torch(win,type_float=True)
        padded_signal = ensure_torch(padded_signal,type_float=True)
    

    filtered_signal = ensure_torch(convolve(padded_signal.numpy(),win.numpy(), method = convolve_type, mode = "full")[...,remaining_margin_start:remaining_margin_end])
    signal_mask = mnt.zeros(filtered_signal.shape[-1], dtype=int)
    
    signal_end = signal_size+remainin_pad_size
    signal_indices = mnt.arange(remainin_pad_size,signal_end, dtype=int, device=device) 
    signal_mask[signal_indices] = 1
    if len(signal.shape)<2:
        filtered_signal = filtered_signal
    
    if return_numba_compatible: # For future use
        return_val = mnt.cat([filtered_signal,signal_mask.unsqueeze(0)], dim = 0).cpu()
        return ensure_numpy(return_val)
    else:
        if return_torch:
            return ensure_torch(filtered_signal), ensure_torch(signal_mask)
        return ensure_numpy(filtered_signal), ensure_numpy(signal_mask)

def band_pass(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",  return_with_pad = False, 
                backend = "torch", device="cpu", verbose = 1, return_torch = False, return_on_CPU = True
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            return_with_pad: bool
                Specifies whether to return the (padded) signal with its mask or to return the (unpadded) signal directly.
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
            filtered_signal: numpy array (or torch tensor)          
                The (possibly still padded) filtered signal.
            signal_mask: numpy array (or torch tensor)
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    
    band = ensure_numpy(band)
    win = get_fir_window(band,ripple,width,fs)
    if (convolve_type == "direct" and len(win)*signal.shape[-1]>1e5 and abs(len(win)-signal.shape[-1])>1e2):
        stderr.write('Resource Warning [band_pass()]: Using the "direct" method is VERY slow on signals of that size. Perhaps consider using the "fft" convolve_type. \n')
    total_signal_size = 1
    for dim_size in signal.shape[:-1]:
        total_signal_size *= dim_size
    if backend == "torch":
        signal = ensure_torch(signal)
        if device == "cuda" and signal.nelement() * signal.element_size() > mnt._check_available_memory():
            stderr.write(f'Resource Warning [band_pass()]: Your signal (of size {signal.nelement() * signal.element_size()*1e-6}MB) is too big to be moved to your GPU. Consider splitting the job into blocks. The process will likely crash now. \n')
        filtered_signal, filtered_mask = band_pass_torchaudio(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device=device,verbose=verbose, return_torch=True, return_on_CPU=False)
    # elif backend =="scipy":
        
    #     if convolve_type == "direct" and total_signal_size+signal.shape[-1]//100 > 50:
    #         stderr.write('Resource Warning [band_pass()]: Using the "scipy" backend with the "direct" convolve_type is VERY slow when applied on multiple signals at the same time. Perhaps consider using the "fft" convolve_type or the "torch" backend. \n')
    #     filtered_signal, filtered_mask = band_pass_scipy(signal,win,fs,return_pad=keep_pad_percent,convolve_type = convolve_type, verbose=verbose)
    else: # For future use
        #raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
        raise NotImplementedError('The only backends available for now is "torch".')
    if return_with_pad:
        if return_torch:
            return ensure_torch(filtered_signal, move_to_CPU = return_on_CPU), ensure_torch(filtered_mask, move_to_CPU = return_on_CPU)
        return ensure_numpy(filtered_signal), ensure_numpy(filtered_mask)
    else:
        if return_torch:
            return ensure_torch(filtered_signal[...,filtered_mask==1], move_to_CPU = return_on_CPU)
        return ensure_numpy(filtered_signal[...,filtered_mask==1])







########################################- Numba compatible functions [Not used yet] - ########################################


# This is likely bloated from multiple functions that do the same thing but with different signal dimensions and floating point precisions, but that's my current best solution to this technical problem. 
def _numba_get_firwin(ntaps, centered_band, beta, fs, precision):
    if precision == "float32":
        return _firwin(ntaps, centered_band, window=("kaiser", beta), scale = True, pass_zero=False,fs=fs).astype(mnt._np_float32)
    elif precision == "float64":
        return _firwin(ntaps, centered_band, window=("kaiser", beta), scale = True, pass_zero=False,fs=fs).astype(mnt._np_float64)
    else:
        return _firwin(ntaps, centered_band, window=("kaiser", beta), scale = True, pass_zero=False,fs=fs)
@njit
def _numba_get_fir_window_float32(band, ripple:float, width:float, fs:int):
    """ This function computes the impulse response of the band-pass filter to get the exact same filter as bst-hfilter-2019 in brainstorm.
    
        Args: 
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            fs : int
                Sampling rate of the signal.
        Keyword Args: 
            None
        Returns: 
            window:                          
                The impulse response of the filter.


    Shamelessly stolen from 
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/fir_filter_design.py#L85
    and 
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/fir_filter_design.py#L29 
    """
    width_norm = width/(0.5*fs)
    a = abs(ripple)  
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    beta = round(beta,4)
    numtaps = (a - 7.95) / 2.285 / (mnt.pi * width_norm) + 1
    ntaps = int(_ceil(numtaps))
    ntaps = ntaps + (1-ntaps%2)
    centered_band = mnt._np_array([1e-5,fs//2 - 1])
    if band[0] is not None:
        centered_band[0] = band[0]-width/2
    if band[1] is not None:
        centered_band[1] = band[1]+width/2
    with objmode(window='float32[:]'):
        window = _numba_get_firwin(ntaps, centered_band,beta, fs, precision="float32")
    return window



@njit
def _numba_get_fir_window_float64(band, ripple:float, width:float, fs:int):
    """ This function computes the impulse response of the band-pass filter to get the exact same filter as bst-hfilter-2019 in brainstorm.
    
        Args: 
            band: list/numpy array/torch tensor 
                Contains the frequency range of the filter.
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            fs : int
                Sampling rate of the signal.
        Keyword Args: 
            None
        Returns: 
            window:                          
                The impulse response of the filter.


    Shamelessly stolen from 
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/fir_filter_design.py#L85
    and 
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/fir_filter_design.py#L29 
    """
    width_norm = width/(0.5*fs)
    a = abs(ripple)  
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    beta = round(beta,4)
    numtaps = (a - 7.95) / 2.285 / (mnt.pi * width_norm) + 1
    ntaps = int(_ceil(numtaps))
    ntaps = ntaps + (1-ntaps%2)
    centered_band = mnt._np_array([1e-5,fs//2 - 1])
    if band[0] is not None:
        centered_band[0] = band[0]-width/2
    if band[1] is not None:
        centered_band[1] = band[1]+width/2
    with objmode(window='float64[:]'):
        window = _numba_get_firwin(ntaps, centered_band,beta, fs, precision="float64")
    return window

@njit
def _numba_band_pass_torchaudio_float32_1d(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu"):
    with objmode(bp_signal='float32[:,:]'):
        bp_signal = band_pass_torchaudio(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, device = device, verbose = 0, return_numba_compatible=True)
    return bp_signal[0], bp_signal[1]

@njit
def _numba_band_pass_torchaudio_float64_1d(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu"):
    with objmode(bp_signal='float64[:,:]'):
        bp_signal = band_pass_torchaudio(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, device = device, verbose = 0, return_numba_compatible=True)
    
    return bp_signal[0], bp_signal[1]

@njit
def _numba_band_pass_torchaudio_float32_2d(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu"):
    with objmode(bp_signal='float32[:,:]'):
        bp_signal = band_pass_torchaudio(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, device = device, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]

@njit
def _numba_band_pass_torchaudio_float64_2d(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu"):
    with objmode(bp_signal='float64[:,:]'):
        bp_signal = band_pass_torchaudio(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, device = device, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]

@njit
def _numba_band_pass_torchaudio_float32_3d(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu"):
    with objmode(bp_signal='float32[:,:,:]'):
        bp_signal = band_pass_torchaudio(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, device = device, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]

@njit
def _numba_band_pass_torchaudio_float64_3d(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu"):
    with objmode(bp_signal='float64[:,:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, device = device, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]
## 
@njit
def _numba_band_pass_scipy_float32_1d(signal,win, fs, return_pad = 0.2, convolve_type = "auto"):
    with objmode(bp_signal='float32[:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, verbose = 0, return_numba_compatible=True)
    return bp_signal[0], bp_signal[1]

@njit
def _numba_band_pass_scipy_float64_1d(signal,win, fs, return_pad = 0.2, convolve_type = "auto"):
    with objmode(bp_signal='float64[:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, verbose = 0, return_numba_compatible=True)
    return bp_signal[0], bp_signal[1]
@njit
def _numba_band_pass_scipy_float32_2d(signal,win, fs, return_pad = 0.2, convolve_type = "auto"):
    with objmode(bp_signal='float32[:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]
@njit
def _numba_band_pass_scipy_float64_2d(signal,win, fs, return_pad = 0.2, convolve_type = "auto"):
    with objmode(bp_signal='float64[:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]

@njit
def _numba_band_pass_scipy_float32_3d(signal,win, fs, return_pad = 0.2, convolve_type = "auto"):
    with objmode(bp_signal='float32[:,:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]

@njit
def _numba_band_pass_scipy_float64_3d(signal,win, fs, return_pad = 0.2, convolve_type = "auto"):
    with objmode(bp_signal='float64[:,:,:]'):
        bp_signal = band_pass_scipy(signal,win, fs, return_pad = return_pad, convolve_type = convolve_type, verbose = 0, return_numba_compatible=True)
    return bp_signal[:-1], bp_signal[-1]

@njit
def _numba_band_pass_float64_1d(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",
                backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: ONLY numpy array 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    win = _numba_get_fir_window_float64(band,ripple,width,fs)
    # 1D Torch
    if backend == "torch":
        filtered_signal, filtered_mask = _numba_band_pass_torchaudio_float64_1d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device = device)
    elif backend == "scipy":
        filtered_signal, filtered_mask = _numba_band_pass_scipy_float64_1d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type)
    else: # For future use
        raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
    return filtered_signal, filtered_mask

@njit
def _numba_band_pass_float64_2d(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",
                backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: ONLY numpy array 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    win = _numba_get_fir_window_float64(band,ripple,width,fs)

    # 1D Torch
    if backend == "torch":
        filtered_signal, filtered_mask = _numba_band_pass_torchaudio_float64_2d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device = device)
    elif backend == "scipy":
        filtered_signal, filtered_mask = _numba_band_pass_scipy_float64_2d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type)
    else: # For future use
        raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
    return filtered_signal, filtered_mask


@njit
def _numba_band_pass_float64_3d(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",
                backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: ONLY numpy array 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    win = _numba_get_fir_window_float64(band,ripple,width,fs)

    # 1D Torch
    if backend == "torch":
        filtered_signal, filtered_mask = _numba_band_pass_torchaudio_float64_3d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device = device)
    elif backend == "scipy":
        filtered_signal, filtered_mask = _numba_band_pass_scipy_float64_3d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type)
    else: # For future use
        raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
    return filtered_signal, filtered_mask


@njit
def _numba_band_pass_float32_1d(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",
                backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: ONLY numpy array 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    win = _numba_get_fir_window_float32(band,ripple,width,fs)
    # 1D Torch
    if backend == "torch":
        filtered_signal, filtered_mask = _numba_band_pass_torchaudio_float32_1d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device = device)
    elif backend == "scipy":
        filtered_signal, filtered_mask = _numba_band_pass_scipy_float32_1d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type)
    else: # For future use
        raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
    return filtered_signal, filtered_mask

@njit
def _numba_band_pass_float32_2d(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",
                backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: ONLY numpy array 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    win = _numba_get_fir_window_float32(band,ripple,width,fs)

    # 1D Torch
    if backend == "torch":
        filtered_signal, filtered_mask = _numba_band_pass_torchaudio_float32_2d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device = device)
    elif backend == "scipy":
        filtered_signal, filtered_mask = _numba_band_pass_scipy_float32_2d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type)
    else: # For future use
        raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
    return filtered_signal, filtered_mask


@njit
def _numba_band_pass_float32_3d(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",
                backend = "torch", device="cpu"
    ):

    """ This function applies the band pass filtering by convolving the signal and the impulse response. Can be applied to signals of any shapes (currently, might vary with different backends) but only takes in a single impulse response and applies the convolution over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: ONLY numpy array 
                The signal to filter.
            fs: : int
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
            keep_pad_percent : float
                The percentage of the padding to keep in the returned filtered signal.
            convolve_type : str
                Specifies which method to use for convolution.
            backend: str
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """
    win = _numba_get_fir_window_float32(band,ripple,width,fs)

    # 1D Torch
    if backend == "torch":
        filtered_signal, filtered_mask = _numba_band_pass_torchaudio_float32_3d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device = device)
    elif backend == "scipy":
        filtered_signal, filtered_mask = _numba_band_pass_scipy_float32_3d(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type)
    else: # For future use
        raise NotImplementedError('The only backends available for now are "torch" and "scipy".')
    return filtered_signal, filtered_mask