from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from pystorm.signal_processing import band_pass
from pystorm import minitorch as mnt

def get_hilbert_torch(signal, fs, pad_size = None, window_mask = None, device = "cpu"):
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

    return analytical_signal

def hilbert(signal, fs, pad_size = None, window_mask = None, return_envelope = False, backend="torch", device = "cpu"):
    if backend == "torch":
        analytical_signal = get_hilbert_torch(signal, fs, pad_size = pad_size, window_mask=window_mask, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    if return_envelope:
        return ensure_numpy(analytical_signal.abs())
    return ensure_numpy(analytical_signal)


def band_pass_hilbert(signal,fs,band,ripple = 60, width = 1.0, keep_pad_percent_for_hilbert = 0.2, return_envelope = False, convolve_type = "auto", backend = "torch", device="cpu"):
    filtered_signal, signal_mask = band_pass(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,convolve_type=convolve_type, return_with_pad=True, backend=backend,device=device)
    if backend == "torch":
        signal_mask = ensure_torch(signal_mask==1.0)
        pad_size = mnt.argwhere(signal_mask)[0]//fs
        analytical_signal = get_hilbert_torch(filtered_signal[...,signal_mask], fs, pad_size = pad_size, window_mask=None, device=device)
    else: # For future use
        raise NotImplementedError('The only backend available for now is "torch".')
    if return_envelope:
        return ensure_numpy(analytical_signal.abs())
    return ensure_numpy(analytical_signal)
