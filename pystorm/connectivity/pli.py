from functools import partial
from numpy import zeros as _np_zeros
from numpy import abs as _np_abs
from numpy import imag as _np_imag
from numpy import conj as _np_conj
from numpy import sign as _np_sign
from numba import njit as _njit
from pystorm.signal_processing import band_pass as _band_pass
from pystorm import minitorch as mnt
from pystorm.timefreq.analytical_signal import hilbert as _hilbert
from pystorm.utils.time_series_utils import get_data_cov as _get_data_cov
from pystorm.timefreq.analytical_signal import get_source_hilbert_torch as _get_source_hilbert_torch 
from pystorm.timefreq.analytical_signal import _get_source_hilbert_torch_sequential 
__all__ = ["get_pli", "get_source_pli"]

pli_dict = {"pli":1,"wpli":2,"dwpli":3}

def get_pli(
                    signal, fs,
                    band, ripple = 60, width = 1, 
                    keep_pad_percent_for_hilbert = 0.2, sliding_window_size = None, overlap = 0.0, return_time_resolved = True,
                    convolve_type = "auto",
                    pli_type = "wpli", use_RAM_efficient = True,
                    backend = "torch", device = "cpu", verbose = 1, return_torch = False
    ):
    """ This function computes the phase lag index (PLI) of a signal within a specific band. It can only be applied to signal directly and does not handle projection from sensor to source space. It first band pass filters the whole signal and then windows it (if sliding_window_size is not None) before computing the hilbert transform and the connectivity values.
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs : int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the band over which to compute the PLI.
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
            return_time_resolved: bool
                Specifies whether to return the connectivity for each sliding window (returns the average if false).
            convolve_type : str
                Specifies which method to use for convolution.
            pli_type: str
                Specifies if 'pli', 'wpli', or 'dwpli' is to be used.
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
            return_torch: bool
                Specifies if the output should be a torch tensor.
        Returns: 
            connectivity: numpy array (or torch tensor) of shape (n_signals, n_signals) if not windowed or (Nwin, n_signals, n_signals) if windowed.
    """
    pli_function  = _get_PLI_loop_RAM_efficient
    pli_selected = pli_dict[pli_type]

    filtered_signal, signal_mask = _band_pass(
                                                signal,fs,
                                                band,ripple=ripple,width=width,
                                                keep_pad_percent=keep_pad_percent_for_hilbert,return_with_pad = True,
                                                convolve_type=convolve_type,
                                                backend=backend, device=device,
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
        iTimes =  time_sequence + i*(Lwin-Loverlap)
        analytical_signal = _hilbert(
                                        unpadded_signal, fs,
                                        pad_size = pad_size,
                                        window_mask=iTimes,
                                        backend = backend, device=device
                            )
        analytical_signal = mnt.ensure_numpy(analytical_signal)
        conn = mnt.ensure_torch(pli_function(analytical_signal, pli_type = pli_selected))
        if return_time_resolved:
            connectivity.append(conn[None,...])
        else:
            if isinstance(connectivity,list):
                connectivity = conn/Nwin 
            else:
                connectivity += conn/Nwin
    if isinstance(connectivity,list):
        connectivity = mnt.cat(connectivity, dim = 0).squeeze()
    if return_torch:
        return mnt.ensure_torch(connectivity)
    return mnt.ensure_numpy(connectivity)


@_njit
def _get_PLI_loop(analytical_signal, pli_type = 1):
    if pli_type < 1 or pli_type>3:
        raise ValueError("pli_type can only take values within [1,2,3]")
    n_signals = analytical_signal.shape[0]
    n_time_points = analytical_signal.shape[-1]
    i = 0
    R = _np_zeros((n_signals,n_signals))
    for i in range(n_signals-1):
        Sab = analytical_signal * _np_conj(analytical_signal[i:i+1])
        if pli_type == 1:
            R[i,:] =_np_sign(_np_imag(Sab)).sum(axis = -1)/n_time_points
        elif pli_type == 2:
            R[i,:] =(_np_imag(Sab).sum(axis = -1)/n_time_points)/(_np_abs(_np_imag(Sab)).sum(axis = -1)/n_time_points)
        elif pli_type == 3:
            sqrd_Sab_imag = (_np_imag(Sab)**2).sum(axis = -1)/n_time_points
            R[i,:] =((_np_imag(Sab).sum(axis = -1)/n_time_points)**2 - sqrd_Sab_imag)/((_np_abs(_np_imag(Sab)).sum(axis = -1)/n_time_points)**2 - sqrd_Sab_imag)
    Sab = analytical_signal * _np_conj(analytical_signal[-1:])
    i = -1
    if pli_type == 1:
        R[i,:] =_np_sign(_np_imag(Sab)).sum(axis = -1)/n_time_points
    elif pli_type == 2:
        R[i,:] =(_np_imag(Sab).sum(axis = -1)/n_time_points)/(_np_abs(_np_imag(Sab)).sum(axis = -1)/n_time_points)
    elif pli_type == 3:
        sqrd_Sab_imag = (_np_imag(Sab)**2).sum(axis = -1)/n_time_points
        R[i,:] =((_np_imag(Sab).sum(axis = -1)/n_time_points)**2 - sqrd_Sab_imag)/((_np_abs(_np_imag(Sab)).sum(axis = -1)/n_time_points)**2 - sqrd_Sab_imag)
    return R


@_njit
def _get_PLI_loop_RAM_efficient(analytical_signal, pli_type = 1):
    if pli_type < 1 or pli_type>3:
        raise ValueError("pli_type can only take values within [1,2,3]")
    n_signals = analytical_signal.shape[0]
    n_time_points = analytical_signal.shape[-1]
    i = 0
    R = _np_zeros((n_signals,n_signals))
    for i in range(n_signals):
        for j in range(i,n_signals):
            Sab = analytical_signal[j] * _np_conj(analytical_signal[i])
            if pli_type == 1:
                R[i,j] =_np_sign(_np_imag(Sab)).sum(axis = -1)/n_time_points
            elif pli_type == 2:
                numerator = (_np_imag(Sab).sum(axis = -1)/n_time_points)
                denominator = (_np_abs(_np_imag(Sab)).sum(axis = -1)/n_time_points)
                if denominator != 0:
                    R[i,j] =numerator/denominator

            elif pli_type == 3:
                sqrd_Sab_imag = (_np_imag(Sab)**2).sum(axis = -1)/n_time_points
                numerator = ((_np_imag(Sab).sum(axis = -1)/n_time_points)**2 - sqrd_Sab_imag)
                denominator = ((_np_abs(_np_imag(Sab)).sum(axis = -1)/n_time_points)**2 - sqrd_Sab_imag)
                if denominator != 0:
                    R[i,j] =numerator/denominator
    if pli_type == 1 or pli_type == 2:
        R = R - R.T
    elif pli_type == 3:
        R = R + R.T
    return R

def get_source_pli(
                    kernels,signal, fs,
                    band, ripple = 60, width = 1, 
                    collapse_function = "pca",
                    keep_pad_percent_for_hilbert = 0.2, sliding_window_size = None, overlap = 0.0,
                    return_time_resolved = True,
                    convolve_type = "auto",
                    pli_type = "wpli", use_RAM_efficient = True, 
                    use_sequential_hilbert = False,
                    backend = "torch", device = "cpu", verbose = 1, return_torch = False, return_everything = False, **kwargs
    ):
    """ This function computes the phase lag index (PLI) of a signal within a specific band. It is applied to the source-space signal by implementing the projection from sensor to source space using a 'collapse_function' applied over all sources within a parcel to give a parcel time series. It first band pass filters the whole signal and then windows it (if sliding_window_size is not None) before computing the hilbert transform and the connectivity values.
    
        Args: 
            kernels: list/numpy array/torch tensor
                The kernel or list of kernels to project the sensor data onto the source space (or parcel-specific source space)
                    -If the collapse function is none the kernels should be of shape [nSources, nSensors].
                    -Otherwise it should be a list of kernels (or an array of objects containing kernels) of length k with each element in the list being a matrix of shape [nSource_in_parcel, nSensors]. Given the variable number of sources per parcel it cannot be a single tensor/array of numbers. It can be a list of matrices or it can be a tensor/array of objects which elements are the matrices.
            signal: list/numpy array/torch tensor 
                The signal to filter
            fs: int
                The signal's sampling rate.
            band: list/numpy array/torch tensor 
                Contains the frequency range of the band over which to compute the PLI.
        Keyword Args: 
            ripple: float
                Positive number specifying maximum ripple in passband (dB) and minimum
                ripple in stopband.
            width: float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            collapse_function: str
                Name of the function used to collapse the sources into parcels.
            keep_pad_percent_for_hilbert: float
                The percentage of size of the the band pass padding to use when computing hilbert transform (default is 0.2, same as internally used in brainstorm) 
            sliding_window_size: float 
                Size of the sliding window (in seconds)
            overlap: float
                Overlap percentage between windows (in 0-1 range)
            return_time_resolved: bool
                Specifies whether to return the connectivity for each sliding window (returns the average if false).
            return_time_resolved: bool
                Specifies whether to return the connectivity for each sliding window (returns the average if false).
            convolve_type: str
                Specifies which method to use for convolution.
            pli_type: str
                Specifies if 'pli', 'wpli', or 'dwpli' is to be used.
            device: str
                Specifies the device in which to apply the filtering.
            verbose: int
                Specifies the verbosity of the function call.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_everything: bool
                Specifies if the output should inlcude all intermediary steps (filtered time series, analytical signal, etc.)
            kwargs: Expect to potentially receive the following additional arguments
                reference_PC: numpy array (or torch tensor) 
                    An array that aims to serve as reference to align the PCs across windows (or trials).
        Returns: 
            connectivity: numpy array (or torch tensor) of shape (n_signals, n_signals) if not windowed or (Nwin, n_signals, n_signals) if windowed. If kernels is a list or a numpy array of objects it will return a list or numpy array of objects of the same length as the kernels with each element being numpy array (or torch tensor) of shape (n_signals, n_signals) if not windowed or (Nwin, n_signals, n_signals) if windowed.
            if 'return_everything':
                unpadded_signal: numpy array (or torch tensor)                
                    The filtered sensor space signal.
                analytical_signal: numpy array (or torch tensor)                
                    The parcellated source space windowed analytical signals used to compute the AEC coefficients.
    """
    pli_function  = _get_PLI_loop_RAM_efficient
    if use_sequential_hilbert:
        hilbert_fun = _get_source_hilbert_torch_sequential
    else:
        hilbert_fun = _get_source_hilbert_torch

    pli_selected = pli_dict[pli_type]

    filtered_signal, signal_mask = _band_pass(
                                                signal,fs,
                                                band,ripple=ripple,width=width,
                                                keep_pad_percent=keep_pad_percent_for_hilbert,return_with_pad = True,
                                                convolve_type=convolve_type,
                                                backend=backend, device=device,
                                                verbose=verbose
                                    )

                                

    signal_mask = mnt.ensure_torch(signal_mask==1.0)

    pad_size = mnt.argwhere(signal_mask)[0]//fs
    unpadded_signal = filtered_signal[...,signal_mask]
    if collapse_function == "pca":
        kwargs["data_cov"] = _get_data_cov(unpadded_signal, device = device)
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
    if return_everything:
        analytical_signals = []
    is_collapsed = True
    if collapse_function is None and ((isinstance(kernels,list) and not mnt._ensure_torch(kernels)[0]) or (isinstance(kernels,mnt._ndarray) and kernels.dtype == 'O')):
        parcel_indices = [mnt.arange(0,kernels[0].shape[0])]
        counter = parcel_indices[0].shape[0]
        for parcel in range(1,len(kernels)):
            parcel_indices.append(mnt.arange(0,kernels[parcel].shape[0])+counter)
            counter += kernels[parcel].shape[0]
        kernels = mnt.cat([mnt.ensure_torch(kernel_parcel) for kernel_parcel in kernels], dim = 0)
        is_collapsed = False
    for i in range(Nwin):
        iTimes =  time_sequence + i*(Lwin-Loverlap)
        
        analytical_signal = hilbert_fun(
                                        kernels,
                                        unpadded_signal, fs,
                                        pad_size = pad_size,
                                        window_mask=iTimes,
                                        backend = backend, device=device,
                                        collapse_function = collapse_function,
                                        return_numba_compatible = True,
                                        **kwargs
                            )
        
        if i == 0 and collapse_function=="pca" and "reference_PC" not in kwargs:
            kwargs["reference_PC"] = analytical_signal[1]
            analytical_signal = analytical_signal[0]
        if return_everything:
            analytical_signals.append(mnt.ensure_torch(analytical_signal[None,...]))
        analytical_signal = mnt.ensure_numpy(analytical_signal)
        conn = mnt.ensure_torch(pli_function(analytical_signal, pli_type = pli_selected))
        if return_time_resolved:
            connectivity.append(conn[None,...])
        else:
            if isinstance(connectivity,list):
                connectivity = conn/Nwin 
            else:
                connectivity += conn/Nwin
    if isinstance(connectivity,list):
        connectivity = mnt.cat(connectivity, dim = 0).squeeze()

    if not return_torch:
        connectivity = mnt.ensure_numpy(connectivity)
        ensure_rest = partial(mnt.ensure_numpy, allow_object_dtype=True)
    else:
        ensure_rest = mnt.ensure_torch
    if return_everything:
        unpadded_signal = ensure_rest(unpadded_signal.squeeze())
        analytical_signals = ensure_rest(mnt.cat(analytical_signals, dim = 0).squeeze())
    if not is_collapsed:
        connectivity = ensure_rest([connectivity[...,parcel,:] for parcel in parcel_indices])
        if return_everything:
            analytical_signals = ensure_rest([analytical_signals[...,parcel,:] for parcel in parcel_indices])
    if return_everything:
        return connectivity, unpadded_signal, analytical_signals
    return connectivity
    