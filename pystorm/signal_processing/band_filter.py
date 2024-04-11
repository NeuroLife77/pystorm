from pystorm.utils.type_safety import ensure_numpy, ensure_torch
from scipy.signal import firwin
from math import ceil 
from pystorm import minitorch as mnt

def get_firwin(ntaps, centered_band, beta, fs):
    return firwin(ntaps, centered_band, window=("kaiser", beta), scale = True, pass_zero=False,fs=fs)

def get_fir_window(band, ripple, width, fs):
    """
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
    ntaps = int(ceil(numtaps))
    ntaps = ntaps + (1-ntaps%2)
    centered_band = ensure_numpy([1e-5,fs//2 - 1])
    if band[0] is not None:
        centered_band[0] = band[0]-width/2
    if band[1] is not None:
        centered_band[1] = band[1]+width/2

    window = get_firwin(ntaps, centered_band,beta, fs)
    return window

def band_pass_torchaudio(signal,win, fs, return_pad = 0.2, convolve_type = "auto", device="cpu", verbose = 1, return_numba_compatible = False):
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
        print("correcting")
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

def band_pass(signal,fs,band,ripple = 60, width = 1.0, keep_pad_percent = 0.2,convolve_type = "auto",  return_with_pad = False, backend = "torch", device="cpu"):
    band = ensure_numpy(band)
    win = get_fir_window(band,ripple,width,fs)
    if backend == "torch":
        filtered_signal, filtered_mask = band_pass_torchaudio(signal,win,fs,return_pad=keep_pad_percent, convolve_type = convolve_type, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    if return_with_pad:
        return filtered_signal, filtered_mask
    else:
        return filtered_signal[...,filtered_mask==1]
