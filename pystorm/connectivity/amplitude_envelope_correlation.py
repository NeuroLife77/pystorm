from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from numpy import zeros as np_zeros
from numpy import abs as np_abs
from numpy import imag as np_imag
from numpy import conj as np_conj
from numpy import diag as np_diag
from numba import njit
from pystorm.signal_processing import band_pass
from pystorm import minitorch as mnt
from pystorm.timefreq.analytical_signal import hilbert

## TODO Documentation

__all__ = ["get_AEC"]
def get_AEC(
                    signal, fs,
                    band, ripple = 60, width = 1, 
                    keep_pad_percent_for_hilbert = 0.2, sliding_window_size = None, overlap = 0.0,
                    convolve_type = "auto",
                    orthogonalize = True, symmetrize = False,
                    backend = "torch", device = "cpu", verbose = 1,
    ):

    filtered_signal, signal_mask = band_pass(signal,fs,band,ripple=ripple,width=width,keep_pad_percent=keep_pad_percent_for_hilbert,return_with_pad = True,convolve_type=convolve_type,backend=backend,device=device, verbose=verbose)
    signal_mask = mnt.ensure_torch(signal_mask==1.0)
    pad_size = mnt.argwhere(signal_mask)[0]//fs
    unpadded_signal = filtered_signal[...,signal_mask]
    if sliding_window_size is None:
        time_sequence = mnt.arange(0,unpadded_signal.shape[-1], dtype = int)
        Nwin = 1
        Lwin = 0
        Loverlap = 0
    else:
        nTime = unpadded_signal.shape[-1]
        Lwin = round(sliding_window_size/1000*fs)
        Loverlap = round(Lwin * overlap)
        Lwin = Lwin - Lwin%2
        Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))
        time_sequence = mnt.arange(0,Lwin, dtype = int)
    connectivity = []
    for i in range(Nwin):
        iTimes =  time_sequence + (i - 1)*(Lwin-Loverlap)
        analytical_signal = hilbert(unpadded_signal, fs, pad_size = pad_size, window_mask=iTimes, backend = backend, device=device)
        analytical_signal = mnt.ensure_numpy(analytical_signal)
        if orthogonalize:
            connectivity.append(mnt.ensure_torch(get_orthogonalized_corr_loop(analytical_signal, symmetrize=symmetrize))[None,...])
        else:
            connectivity.append(mnt.ensure_torch(get_corr_loop(analytical_signal, symmetrize=symmetrize))[None,...])
    return mnt.ensure_numpy(mnt.cat(connectivity, dim = 0).squeeze())


@njit
def normr(x, remove_mean = False):
    if remove_mean:
        x_mean_centered = x-(x.sum(1)/x.shape[1])[:,None]
    else:
        x_mean_centered = x
    n = np_sqrt(np_sum(x_mean_centered**2, axis = 1))
    if (n!=0).sum()>0:
        x_mean_centered[n!=0,:] = x_mean_centered[n!=0,:]/(n[n!=0])[:,None]
    x_mean_centered[n==0,:] = x_mean_centered[n==0,:]/np_sqrt(x_mean_centered.shape[1])
    return x_mean_centered

@njit
def corrn(x,y, remove_mean = False):
    Xc = normr(x,remove_mean=remove_mean)
    Yc = normr(y,remove_mean=remove_mean)
    return Xc @ Yc.conj().T

## TODO Fix the numba warning about @ being faster on contigous arrays
@njit  
def get_orthogonalized_corr_loop(analytical_signal, symmetrize = False):
    n_signals = analytical_signal.shape[0]
    R = np_zeros((n_signals,n_signals))
    for i in range(n_signals):
        R[i,:] = np_abs(corrn(
                            np_abs(analytical_signal[i])[None,:], # abs(X)
                            np_abs(
                                    np_imag(analytical_signal * np_conj(analytical_signal[i]))/np_abs(analytical_signal[i]) # imag(Y * conj(X))/abs(X)
                            ),
                            remove_mean=True
                ))[0]/2 + np_diag(np_abs(corrn(
                            np_abs(
                                    np_imag(analytical_signal[i] * np_conj(analytical_signal))/np_abs(analytical_signal)# imag(X * conj(Y))/abs(Y)
                            ),                                    
                            np_abs(analytical_signal), # abs(Y)
                            remove_mean=True
                )))/2
    if  symmetrize:
        R = (R + R.T)/2
    return R


@njit  
def get_corr_loop(analytical_signal, symmetrize = False):
    n_signals = analytical_signal.shape[0]
    R = np_zeros((n_signals,n_signals))
    for i in range(n_signals):
        R[i,:] = corrn(
                            np_abs(analytical_signal[i])[None,:], # abs(X)
                            np_abs(analytical_signal),  # abs(Y)
                            remove_mean=True
                )[0]
    if  symmetrize:
        R = (R + R.T)/2
    return R
