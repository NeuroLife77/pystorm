import pystorm as pst
from pystorm import minitorch as mnt
from torch import where as _twhere
from torch import isclose as _tisclose
from torch import randperm as _trandperm

__all__ = ["pac"]

def pac(
    signal,
    fs,
    fA_bin_width = 5,
    fA_bin_edges = [30,120],
    range_of_fP_search = [4,8],
    window_length = None,
    n_win = None,
    window_offset = 0,
    overlap = 0.5,
    ripple = 40,
    width = 1.0,
    psd_zero_pad = 1.5,
    percentage_of_total_band_power_threshold_signal = 0.0000125,
    percentage_of_total_band_power_threshold_envelope = 0.0000125,
    n_sur = 500,
    n_blocks = 5,
    signal_flip_mask = None,
    backend = 'torch',
    device = 'cpu',
    return_torch = False,
    verbose = 0,
):

    """ This function extracts Phase Amplitude Coupling (PAC) from the signals in addition to z-score values obtained from testing against surrogate data (block-resampling). 
    
    It also implements time-resolved PAC when provided with a window_length and/or n_win. 
    
    It can take multiple signals at the same time and applies the function over the last dimension of the signal (time series should be along dim=-1).
    
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter.
            fs: : int
                The signal's sampling rate.
        Keyword Args: 
            fA_bin_width: float
                Specifies the width of the bins for fA grid search
            fA_bin_edges: list/numpy array/torch tensor 
                Specifies the bounds of the fA grid search
            range_of_fP_search: list/numpy array/torch tensor 
                Specifies the bounds of fP (used as band for band_pass)
            window_length: float
                Specifies the length (in seconds) of the sliding windows (if tPAC)
            n_win: int
                Specifies the number of sliding windows (if tPAC), will default to a computed max_n_win if 'None'
            overlap: float
                Specifies the overlap (between 0 and 1) of the sliding windows (if tPAC)
            ripple : float
                Positive number specifying maximum ripple in passband (dB) and minimum ripple in stopband.
            width : float
                Width of transition region (normalized so that 1 corresponds to pi
                radians / sample).
            psd_zero_pad : float
                Length (in seconds) of padding to add when computing PSDs.
            percentage_of_total_band_power_threshold_signal: float
                Tuning parameter to decide on what constitutes a peak within the original signal's power
            percentage_of_total_band_power_threshold_envelope: float
                Tuning parameter to decide on what constitutes a peak within the signal's envelope power
            n_sur: int
                Specifies the number of surrogates per block to use when computing the z-score.
            n_blocks: int
                Specifies the number of surrogate blocks to use when computing the z-score.
            flip_signal_signs: Bool
                Specifies whether to align the signals using sign flip [mainly for source-space analysis]. 
            backend: str
                Specifies which backend to use. [Currently 'torch' and 'scipy' are available, but band_pass defaults to 'torch' until the scipy version is fixed]
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            verbose: int
                Specifies the verbosity of the function call.
        Returns: 
            sPAC: Dictionary of numpy arrays (or torch tensors)
                //Note: All arrays will ALWAYS have the shape (num_winows, num_fA_bins, num_signals, ...) irrespective of the arguments passed to the function//

                Dictionary content breakdown:
                    -sPAC['PAC']: Coupling coefficients
                    -sPAC['phase']: phase
                    -sPAC['fP']: the fP value for each fA
                    -sPAC['PAC_complex']: Coupling values in the complex form
                    -sPAC['MaxPAC_index']: Index of the fA with max PAC coefficient, it has the shape (num_winows, 1, num_signals, ...) to enable easy 'take_along_axis' or 'take_along_dim'
                    -sPAC['z_score']: z-score values wrt surrogate (if used)
    """
    signals = mnt.ensure_torch(signal).to(device)
    if signal_flip_mask is not None:
        signals = signals * mnt.ensure_torch(signal_flip_mask).to(device).unsqueeze(-1)
    if len(signals.shape) == 1:
        signals = signals.unsqueeze(0)

    if window_length is None:
        window_size = int(signals.shape[-1] - window_offset*fs)
    else:
        window_size = int(window_length*fs)
    if window_size>signals.shape[-1]-window_offset*fs:
        raise ValueError("The value of 'window_length' exceeds the 'signals' length given minus the 'window_offset'.")
    max_n_win = int((signals.shape[-1]-window_offset*fs-window_size*overlap)/(window_size*overlap))
    if n_win is None:
        n_win = max_n_win
    if n_win>max_n_win:
        raise ValueError("The value of 'n_win' exceeds the max number of windows of the length specified by 'window_length' given the 'window_offset' and 'signals' length.")
    n_win = int(n_win)

    bands = mnt.cat([
                    mnt.arange(fA_bin_edges[0],fA_bin_edges[1],fA_bin_width,dtype=mnt.__default_dtype__).view(-1,1)-max(fA_bin_width,range_of_fP_search[1]),
                    mnt.arange(fA_bin_edges[0],fA_bin_edges[1],fA_bin_width,dtype=mnt.__default_dtype__).view(-1,1)+max(fA_bin_width,range_of_fP_search[1])
    ],dim=-1)

    analytical_signals = pst.get_multiple_band_pass_hilbert(signals,fs,bands,ripple=ripple,width=width, device=device, return_envelope=False, return_torch=True, verbose = 0, backend=backend)
    analytical_signals = analytical_signals.transpose(0,1)


    windowed_signals,windowed_envelopes = mnt.zeros(n_win,*signals.shape[:-1],window_size),mnt.zeros(n_win,*analytical_signals.shape[:-1],window_size)
    for window_id in range(n_win):
        window = mnt.arange(window_size)+int(window_id*window_size*(1-overlap)) + int(window_offset*fs)
        windowed_signals[window_id] = signals[...,window]
        windowed_envelopes[window_id] = analytical_signals[...,window].abs()
    windowed_envelopes = windowed_envelopes - windowed_envelopes.mean(-1,keepdims = True)

    psd_env, _, freqs_fft_envelope = pst.get_fft_torch(windowed_envelopes,fs,pad_size=psd_zero_pad, return_complex_fft=False,device = device, return_torch=True)
    psd_signal, _, freqs_fft_signal = pst.get_fft_torch(windowed_signals,fs,pad_size=psd_zero_pad, return_complex_fft=False,device = device, return_torch=True)

    freq_mask_for_peaks_envelope = mnt.logical_and(freqs_fft_envelope>=range_of_fP_search[0],freqs_fft_envelope<=range_of_fP_search[1])
    freq_mask_for_peaks_signal = mnt.logical_and(freqs_fft_signal>=range_of_fP_search[0],freqs_fft_signal<=range_of_fP_search[1])

    peaks_of_envelope, location_of_peaks_of_envelope = _find_peaks_recur(
                                                                        psd_env[...,freq_mask_for_peaks_envelope],
                                                                        freqs_fft_envelope[...,freq_mask_for_peaks_envelope],
                                                                        power_thresh = percentage_of_total_band_power_threshold_envelope
                                                )
    peaks_of_signals, location_of_peaks_of_signals = _find_peaks_recur(
                                                                        psd_signal[...,freq_mask_for_peaks_signal],
                                                                        freqs_fft_signal[...,freq_mask_for_peaks_signal],
                                                                        power_thresh = percentage_of_total_band_power_threshold_signal
                                                )


    max_dist_between_peaks = max(1.5/(window_size/fs), 1.5)
    sPAC = {}
    sPAC['PAC'] = mnt.zeros(n_win,bands.shape[0],len(peaks_of_signals[0]))
    sPAC['phase'] = mnt.zeros(n_win,bands.shape[0],len(peaks_of_signals[0]))
    sPAC['fP'] = mnt.zeros(n_win,bands.shape[0],len(peaks_of_signals[0]))
    sPAC['PAC_complex'] = mnt.zeros(n_win,bands.shape[0],len(peaks_of_signals[0]), dtype=mnt.__default_complex_dtype__)
    if n_sur is not None and n_sur> 0:
        sPAC['z_score'] = mnt.zeros(n_win,bands.shape[0],len(peaks_of_signals[0]))
    for window_index in range(n_win):
        if verbose > 0:
            print("Window", window_index)
        for fA_index, fA_value in enumerate(bands.mean(-1)):
            if verbose > 0:
                print("\t",fA_index,":",fA_value.item(),"Hz")
            for signal_index, signal_peaks in enumerate(peaks_of_signals[window_index]):
                if verbose > 0:
                    print("\t \t","Signal",signal_index)
                if peaks_of_envelope[window_index][signal_index][fA_index] is not None and peaks_of_signals[window_index][signal_index] is not None:
                    distance_between_env_peaks_and_signal_peaks = (peaks_of_envelope[window_index][signal_index][fA_index].unsqueeze(1) - peaks_of_signals[window_index][signal_index].unsqueeze(0)).abs()
                    if distance_between_env_peaks_and_signal_peaks.amin()<max_dist_between_peaks:
                        closest_peak = distance_between_env_peaks_and_signal_peaks.amin(1).argmin(0)
                    else:
                        closest_peak = None
                    fP_freq = peaks_of_envelope[window_index][signal_index][fA_index][closest_peak].squeeze()
                    
                    fP_band = [fP_freq-0.5,fP_freq+0.5]
                    
                    analytical_signal_of_fp_bandpassed_signal = pst.band_pass_hilbert(
                                                                                        windowed_signals[window_index,signal_index], fs=fs,
                                                                                        band=fP_band,ripple=ripple,width=width,
                                                                                        verbose=0,device=device,return_torch=True, backend=backend
                        ).squeeze()
                    phase_of_fP = (analytical_signal_of_fp_bandpassed_signal-analytical_signal_of_fp_bandpassed_signal.mean(-1,keepdims=True)).angle()
                    
                    phase_of_fP_locked_to_t0 = phase_of_fP - phase_of_fP[0]
                    is_cycle_end = _twhere(
                                                _tisclose(
                                                    phase_of_fP_locked_to_t0,
                                                    mnt.zeros_like(phase_of_fP_locked_to_t0),
                                                    atol = phase_of_fP_locked_to_t0.diff().abs().mean()
                                                )
                                )[0]
                    last_cycle_end = is_cycle_end[-1]
                    finer_fA_band = [fA_value-1.1*fP_band[1],fA_value+1.1*fP_band[1]]
                    
                    analytical_signal_of_fA_bandpassed_signal = pst.band_pass_hilbert(
                                                                                        windowed_signals[window_index,signal_index], fs=fs,
                                                                                        band=finer_fA_band,ripple=ripple,width=width,
                                                                                        verbose=0,device=device,return_torch=True, backend=backend
                        ).squeeze()
                    amplitude_of_finer_fA_band = analytical_signal_of_fA_bandpassed_signal.abs()
                    
                    PAC_val = (amplitude_of_finer_fA_band[:last_cycle_end] * (1j*phase_of_fP[:last_cycle_end]).exp()).sum() / (
                        (amplitude_of_finer_fA_band[:last_cycle_end] ** 2).mean()
                        ).sqrt() / last_cycle_end
                    
                    DynamicPhase = PAC_val.angle()
                    if n_sur is not None and n_sur > 0:
                        surrogate_amplitude = analytical_signal_of_fA_bandpassed_signal[:last_cycle_end].abs()
                        num_points = last_cycle_end
                        block_len = num_points//n_blocks
                        amplitude_blocks = surrogate_amplitude[:block_len*n_blocks].view(n_blocks, block_len)
                        sur_PAC_val = mnt.zeros(n_sur,*PAC_val.shape, dtype=mnt.__default_complex_dtype__)
                        for sur in range(n_sur):
                            order = _trandperm(n_blocks)
                            sur_amplitude = amplitude_blocks[order].flatten()
                            sur_PAC_val[sur] = (sur_amplitude * (1j*phase_of_fP[:block_len*n_blocks]).exp()).sum() / (
                            (sur_amplitude ** 2).mean()
                            ).sqrt() / last_cycle_end
                        PAC_val_z_score = (PAC_val.abs() - sur_PAC_val.abs().mean(0))/sur_PAC_val.abs().std(0)
                        sPAC['z_score'][window_index, fA_index, signal_index] = PAC_val_z_score

                    if verbose > 0:
                        if n_sur is not None and n_sur>0:
                            print("\t \t \t","PAC=", round(PAC_val.abs().item(),4),'| fP=', round(fP_freq.item(),2),'| phase=', round(DynamicPhase.item(),4),'| Z-score=',round(PAC_val_z_score.item(),2))
                        else:
                            print("\t \t \t","PAC=", round(PAC_val.abs().item(),4),'| fP=', round(fP_freq.item(),2),'| phase=', round(DynamicPhase.item(),4))

                    sPAC['PAC_complex'][window_index, fA_index, signal_index] = PAC_val
                    sPAC['PAC'][window_index, fA_index, signal_index] = PAC_val.abs()
                    sPAC['phase'][window_index, fA_index, signal_index] = DynamicPhase
                    sPAC['fP'][window_index, fA_index, signal_index] = fP_freq

    sPAC["MaxPAC_index"] = sPAC['PAC'].argmax(1,keepdims=True)

    if not return_torch:
        sPAC['PAC'] = mnt.ensure_numpy(sPAC['PAC'])
        sPAC['phase'] = mnt.ensure_numpy(sPAC['phase'])
        sPAC['fP'] = mnt.ensure_numpy(sPAC['fP'])
        sPAC['MaxPAC_index'] = mnt.ensure_numpy(sPAC['MaxPAC_index'])
        sPAC['PAC_complex'] = mnt.ensure_numpy(sPAC['PAC_complex'])
        if n_sur is not None and n_sur> 0:
            sPAC['z_score'] = mnt.ensure_numpy(sPAC['z_score'])

    return sPAC
                    

def _get_peak_from_diff(is_peak,freqs,peak_freqs, peak_freqs_location, edge_start = None, edge_end = None):

    for i, peaks in enumerate(is_peak):
        peak_freqs.append([])
        peak_freqs_location.append([])
        if len(peaks.shape) == 1:
            if peaks.sum()>0:
                peak_freqs[i].append((freqs[1:-1])[peaks])
                peak_freqs_location[i].append((peaks.argwhere()+1))
            if len(peak_freqs[i])>0:
                peak_freqs[i] = mnt.cat(peak_freqs[i]).view(-1)
                peak_freqs_location[i] = mnt.cat(peak_freqs_location[i]).view(-1)
            else:
                peak_freqs[i] = None
                peak_freqs_location[i] = None
        else:
            if edge_start is not None:
                peak_freqs[i], peak_freqs_location[i] = _get_peak_from_diff(peaks, freqs,peak_freqs[i],peak_freqs_location[i], edge_start[i],edge_end[i])
            else:
                peak_freqs[i], peak_freqs_location[i] = _get_peak_from_diff(peaks, freqs,peak_freqs[i],peak_freqs_location[i])
    return peak_freqs, peak_freqs_location


def _find_peaks_recur(psd, freqs, power_thresh=0.05, use_edges = False):
    
    thresh = (psd.sum(-1,keepdims=True)*power_thresh)
    forward_diff = psd.diff(dim = -1)
    is_peak = mnt.logical_and(forward_diff[...,:-1]>thresh,forward_diff[...,1:]<-thresh)
    if use_edges:
        edge_start, edge_end = forward_diff[...,0]<-thresh.squeeze(), forward_diff[...,-1]>thresh.squeeze()
    peak_freqs = [[],[]]
    if len(psd.shape)>1:
        if use_edges:
            peak_freqs[0],peak_freqs[1] = _get_peak_from_diff(is_peak, freqs, peak_freqs[0], peak_freqs[1], edge_start=edge_start, edge_end=edge_end)
        else:
            peak_freqs[0],peak_freqs[1] = _get_peak_from_diff(is_peak, freqs, peak_freqs[0], peak_freqs[1])
    else:
        if use_edges and edge_start.item():
            peak_freqs[0].append(freqs[:1])    
            peak_freqs[1].append(mnt.as_tensor([[0]]))  
        if is_peak.sum()>0:
            peak_freqs[0].append((freqs[1:-1])[is_peak])
            peak_freqs[1].append(is_peak.argwhere()+1)
        if use_edges and edge_end.item():
            peak_freqs[0].append(freqs[-1:])    
            peak_freqs[1].append(mnt.as_tensor([[freqs.shape[0]-1]]))  
        peak_freqs[0] = mnt.cat(peak_freqs[0]).view(-1) 
        peak_freqs[1] = mnt.cat(peak_freqs[1]).view(-1)

    return peak_freqs