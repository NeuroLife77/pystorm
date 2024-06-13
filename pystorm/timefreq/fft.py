from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from pystorm import minitorch as mnt

__all__ = ["get_fft_torch"]

def get_fft_torch(
                        signal, fs : int,
                        pad_size = None,
                        return_complex_fft = True,
                        device = "cpu",
                        return_torch = False, return_on_CPU = True
    ):
    """ This function applies the fast fourier transformation using pytorch functions. Can be applied to signals of any shapes but only applies the fft over the last dimension of the signal (time series should be along dim=-1).
    
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to tramsform,
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            pad_size : float
                The size (in seconds) of the padding to add before applying the fft transform. 
            window_mask : str
                Specifies which part of the input signal to transform. [Only useful when applying windowed transform on signal that was previously filtered on its full length.]
            device: str
                Specifies the device in which to apply the fft.
            return_numba_compatible: bool
                Specifies if the signal should be returned as numpy array (required for numba)
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            signal_fft: numpy array (or torch tensor)   [could be signal_power and signal_phase if return_complex_fft=False]             
                The fft of the signal signal.
            freqs: numpy array (or torch tensor)
                The frequencies for the fft.
    """

    signal = ensure_torch(signal).to(device)
    
    if pad_size is not None:
        zero_pad_size = int(pad_size*fs)
        padded_signal = mnt.cat([mnt.zeros(*signal.shape[:-1],zero_pad_size, device=device),signal,mnt.zeros(*signal.shape[:-1],zero_pad_size, device=device)], dim = -1).to(device)
    else:
        padded_signal = signal
        zero_pad_size = 0
    #print(signal.shape,padded_signal.shape,zero_pad_size)
    freqs = mnt.fftfreq(padded_signal.shape[-1],d=1/fs)
    signal_fft = mnt.fft(padded_signal)
    if return_complex_fft:
        if return_torch:
            return ensure_torch(signal_fft, move_to_CPU = return_on_CPU), ensure_torch(freqs, move_to_CPU = return_on_CPU)
        return ensure_numpy(signal_fft), ensure_numpy(freqs)
    signal_fft_real = signal_fft[...,freqs>=0]
    signal_power = signal_fft_real.abs()**2
    signal_phase = signal_fft_real.angle()
    #print(signal_power.shape)
    if return_torch:
        return ensure_torch(signal_power, move_to_CPU = return_on_CPU), ensure_torch(signal_phase, move_to_CPU = return_on_CPU), ensure_torch(freqs[freqs>=0], move_to_CPU = return_on_CPU)
    return ensure_numpy(signal_power), ensure_numpy(signal_phase), ensure_numpy(freqs[freqs>=0])