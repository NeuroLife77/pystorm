from numpy import sqrt as _np_sqrt
from numpy import sum as _np_sum
from numpy import zeros as _np_zeros
from numpy import abs as _np_abs
from numpy import imag as _np_imag
from numpy import conj as _np_conj
from numpy import diag as _np_diag
from numba import njit as _njit
from pystorm.signal_processing import band_pass as _band_pass
from pystorm import minitorch as mnt
from pystorm.timefreq.analytical_signal import hilbert as _hilbert

__all__ = ["get_AEC"]


def get_AEC(
                    signal, fs,
                    band, ripple = 60, width = 1, 
                    keep_pad_percent_for_hilbert = 0.2, sliding_window_size = None, overlap = 0.0,
                    convolve_type = "auto",
                    orthogonalize = True, symmetrize = False,
                    backend = "torch", device = "cpu", verbose = 1,
    ):
    """ This function computes the amplitude envelope correlation of a signal within a specific band. It can only be applied to signal directly and does not handle projection from sensor to source space. It first band pass filters the whole signal and then windows it (if sliding_window_size is not None) before computing the hilbert transform and the correlation coefficients.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the band over which to compute the amplitude envelope correlation.
        Keyword Args: 
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            keep_pad_percent_for_hilbert : float
                The percentage of size of the the band pass padding to use when computing hilbert transform (default is 0.2, same as internally used in brainstorm) 
            sliding_window_size : float 
                Size of the sliding window (in seconds)
            overlap : float
                Overlap percentage between windows (in 0-1 range)
            convolve_type : str
                Specifies which method to use for convolution.
            orthogonalize : bool
                Whether to use 'oenv' (when set to True) or 'henv' (when set to False) to compute the AEC. While this operation is numba compiled 'oenv' take significantly longer to run.
            symmetrize : bool
                Whether to return the symmetric matrix (R + R.T)/2 instead of R.
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
        Returns: 
            connectivity: numpy array of shape (n_signals, n_signals) if not windowed or (Nwin, n_signals, n_signals) if windowed.
    """
    

    filtered_signal, signal_mask = _band_pass(
                                                signal,fs,
                                                band,ripple=ripple,width=width,
                                                keep_pad_percent=keep_pad_percent_for_hilbert,return_with_pad = True,
                                                convolve_type=convolve_type,
                                                backend=backend,device=device,
                                                verbose=verbose
                                    )

    signal_mask = mnt.ensure_torch(signal_mask==1.0)
    pad_size = mnt.argwhere(signal_mask)[0]//fs
    unpadded_signal = filtered_signal[...,signal_mask]
    if sliding_window_size is None:
        time_sequence = mnt.arange(0,unpadded_signal.shape[-1], dtype = int)
        Nwin = 1
        Lwin = 0
        Loverlap = 0
    else:
        Lwin = round(sliding_window_size*fs)
        nTime = unpadded_signal.shape[-1]
        
        Loverlap = round(Lwin * overlap)
        Lwin = Lwin - Lwin%2
        Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))
        time_sequence = mnt.arange(0,Lwin, dtype = int)
    connectivity = []
    for i in range(Nwin):
        iTimes =  time_sequence + (i - 1)*(Lwin-Loverlap)
        analytical_signal = _hilbert(
                                        unpadded_signal, fs,
                                        pad_size = pad_size,
                                        window_mask=iTimes,
                                        backend = backend, device=device
                            )
        analytical_signal = mnt.ensure_numpy(analytical_signal)
        if orthogonalize:
            connectivity.append(mnt.ensure_torch(_get_orthogonalized_corr_loop(analytical_signal, symmetrize=symmetrize))[None,...])
        else:
            connectivity.append(mnt.ensure_torch(_get_corr_loop(analytical_signal, symmetrize=symmetrize))[None,...])
    return mnt.ensure_numpy(mnt.cat(connectivity, dim = 0).squeeze())


@_njit
def _normr(x, remove_mean = False):
    if remove_mean:
        x_mean_centered = x-(x.sum(1)/x.shape[1])[:,None]
    else:
        x_mean_centered = x
    n = _np_sqrt(_np_sum(x_mean_centered**2, axis = 1))
    if (n!=0).sum()>0:
        x_mean_centered[n!=0,:] = x_mean_centered[n!=0,:]/(n[n!=0])[:,None]
    x_mean_centered[n==0,:] = x_mean_centered[n==0,:]/_np_sqrt(x_mean_centered.shape[1])
    return x_mean_centered

@_njit
def _corrn(x,y, remove_mean = False):
    Xc = _normr(x,remove_mean=remove_mean)
    Yc = _normr(y,remove_mean=remove_mean).conj().T
    return Xc @ Yc

## It's not pretty, but it fix the numba slow down from having to add an axis to analytical_signal[i] when passing it in _corrn.
@_njit  
def _get_orthogonalized_corr_loop(analytical_signal, symmetrize = False):
    n_signals = analytical_signal.shape[0]
    i = 0
    R = _np_zeros((n_signals,n_signals))
    for i in range(n_signals-1):
        R[i,:] = _np_abs(_corrn(
                            _np_abs(analytical_signal[i:i+1]), # abs(X)
                            _np_abs(
                                    _np_imag(analytical_signal * _np_conj(analytical_signal[i:i+1]))/_np_abs(analytical_signal[i:i+1]) # imag(Y * conj(X))/abs(X)
                            ),
                            remove_mean=True
                ))[0]/2 + _np_diag(_np_abs(_corrn(
                            _np_abs(
                                    _np_imag(analytical_signal[i:i+1] * _np_conj(analytical_signal))/_np_abs(analytical_signal)# imag(X * conj(Y))/abs(Y)
                            ),                                    
                            _np_abs(analytical_signal), # abs(Y)
                            remove_mean=True
                )))/2
    R[-1,:] = _np_abs(_corrn(
                            _np_abs(analytical_signal[-1:]), # abs(X)
                            _np_abs(
                                    _np_imag(analytical_signal * _np_conj(analytical_signal[-1:]))/_np_abs(analytical_signal[-1:]) # imag(Y * conj(X))/abs(X)
                            ),
                            remove_mean=True
                ))[0]/2 + _np_diag(_np_abs(_corrn(
                            _np_abs(
                                    _np_imag(analytical_signal[-1:] * _np_conj(analytical_signal))/_np_abs(analytical_signal)# imag(X * conj(Y))/abs(Y)
                            ),                                    
                            _np_abs(analytical_signal), # abs(Y)
                            remove_mean=True
                )))/2
    if  symmetrize:
        R = (R + R.T)/2
    return R


@_njit  
def _get_corr_loop(analytical_signal, symmetrize = False):
    n_signals = analytical_signal.shape[0]
    R = _np_zeros((n_signals,n_signals))
    for i in range(n_signals-1):
        R[i,:] = _corrn(
                            _np_abs(analytical_signal[i:i+1]), # abs(X)
                            _np_abs(analytical_signal),  # abs(Y)
                            remove_mean=True
                )[0]
    R[-1,:] = _corrn(
                            _np_abs(analytical_signal[-1:]), # abs(X)
                            _np_abs(analytical_signal),  # abs(Y)
                            remove_mean=True
                )[0]
    if  symmetrize:
        R = (R + R.T)/2
    return R

# Precompiling the functions. Adds a bit of runtime on import, but ensures that it's already compiled when first called.
#_ = _get_orthogonalized_corr_loop(mnt.ensure_numpy(mnt.cos(mnt.arange(3).unsqueeze(0))), symmetrize=True)
#_ = _get_corr_loop(mnt.ensure_numpy(mnt.cos(mnt.arange(3).unsqueeze(0))), symmetrize=True)
