from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from scipy.signal import firwin as _firwin
from math import ceil as _ceil 
from pystorm import mnt

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
                            verbose = 1, return_numba_compatible = False
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
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """

    signal = ensure_torch(signal)
    win = ensure_torch(win)
    
    if convolve_type == "fft" or (convolve_type == "auto" and len(win)*signal.shape[-1]>1e5 and abs(len(win)-signal.shape[-1])>1e2):
        convolve = mnt.tfftconvolve
    else:
        convolve = mnt.tconvolve

    win = win.to(device)

    E = win[win.shape[-1]//2:] ** 2
    E = E.cumsum(dim=-1)
    E = E / E.amax()
    iE99 = ((E-0.99).abs().argmin() / fs).item()
    edge_percent = 2*iE99 / (signal.shape[-1]/fs)
    if edge_percent>0.1:
        if verbose > 0:
            print(f"Start up and end transients represent {round(edge_percent*100,2)}% of your data.")

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
        return ensure_numpy(filtered_signal), ensure_numpy(signal_mask)

def band_pass(
                signal,fs:int,band,
                ripple = 60.0, width = 1.0, 
                keep_pad_percent = 0.2,
                convolve_type = "auto",  return_with_pad = False, 
                backend = "torch", device="cpu", verbose = 1
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
                Specifies which backend to use. [Currently only 'torch' is available.]
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
        Returns: 
            filtered_signal: numpy array                       
                The (possibly still padded) filtered signal.
            signal_mask: numpy array
                The mask of the signal (specifying the location of the signal within the padded signal) 
    """

    band = ensure_numpy(band)
    win = get_fir_window(band,ripple,width,fs)
    if backend == "torch":
        filtered_signal, filtered_mask = band_pass_torchaudio(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device=device,verbose=verbose)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    if return_with_pad:
        return filtered_signal, filtered_mask
    else:
        return filtered_signal[...,filtered_mask==1]
