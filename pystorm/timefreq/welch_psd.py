from pystorm.utils import minitorch as mnt
from pystorm.utils.minitorch import ensure_numpy
from sys import stderr 
__all__ = ["welch_psd_source_space", "welch_psd_sensor_space", "welch_psd_time_resolved_source_space", "welch_psd_time_resolved_sensor_space"]

def welch_psd_source_space(
                            time_series, fs : int, ker = None,
                            window_length = 2, overlap = 0.5,
                            device = "cpu",
                            return_torch = False, return_on_CPU = True, move_window_to_GPU = True,
    ):

    
    """ This function computes the PSD of source-space signals using the welch method. It can be called on the source time series directly or on the sensor time series with the imaging kernel
    
        Args: 
            time_series: list/numpy array/torch tensor 
                The time series over which to compute the PSD
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            ker: list/numpy array/torch tensor 
                The imaging kernel (must be passed in only if the time_series is a sensor-space signal)
            window_length : float
                Length of the sliding window (in seconds)
            overlap : float
                Overlap percentage between windows (in 0-1 range)
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            move_window_to_device: bool
                Only useful if the device is not "cpu". Specifies whether to move only the windowed time_series to the device (slower runtime & less memory) or the whole time_series (faster runtime & more memory).
        Returns: 
            psd: numpy array (or torch tensor)               
                The PSD of the source signal
            freqs: numpy array (or torch tensor)                      
                The frequencies of the PSD.
    """

    if ker is None:
        return welch_psd_sensor_space(time_series, fs, window_length=window_length, overlap = overlap, device = device, return_torch = return_torch, return_on_CPU=return_on_CPU)
    nTime = time_series.shape[-1]
    Lwin = round(window_length*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    ker_device = mnt.ensure_torch(ker, type_float=True).to(device).type(mnt.__default_complex_dtype__)
    sensor_device = mnt.ensure_torch(time_series, type_float=True)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)
    if not move_window_to_GPU:
        sensor_device = sensor_device.to(device)
        time_sequence = time_sequence.to(device)

    fft_blocks = 0
    for i in range(Nwin):
        iTimes =  time_sequence + (i)*(Lwin-Loverlap)
        block = sensor_device[..., iTimes]
        if move_window_to_GPU:
            block = block.to(device)
        windowed_ts = (block) * hamming_window
        fft_res = mnt.rfft(windowed_ts,dim = -1) * scaling_term
        fft_res[..., 1:] /= 2**(1/2)
        fft_blocks += (ker_device @ fft_res).abs()**2/Nwin
    
    if return_torch:
        return mnt.ensure_torch(fft_blocks, move_to_CPU = return_on_CPU), mnt.ensure_torch(freqs, move_to_CPU = return_on_CPU)
    return ensure_numpy(fft_blocks), ensure_numpy(freqs)

def welch_psd_sensor_space(
                            time_series, fs : int,
                            window_length = 2, overlap = 0.5,
                            device = "cpu", return_torch = False, return_on_CPU = True, move_window_to_GPU = True,
    ):
    """ This function computes the PSD of sensor-space signals using the welch method.
    
        Args: 
            time_series: list/numpy array/torch tensor 
                The sensor time series over which to compute the PSD
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            window_length : float
                Length of the sliding window (in seconds)
            overlap : float
                Overlap percentage between windows (in 0-1 range)
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            move_window_to_device: bool
                Only useful if the device is not "cpu". Specifies whether to move only the windowed time_series to the device (slower runtime & less memory) or the whole time_series (faster runtime & more memory).
        Returns: 
            psd: numpy array (or torch tensor)               
                The PSD of the sensor signal
            freqs: numpy array (or torch tensor)                      
                The frequencies of the PSD.
    """

    nTime = time_series.shape[-1]
    Lwin = round(window_length*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    sensor_device = mnt.ensure_torch(time_series, type_float=True)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    
    time_sequence = mnt.arange(0,Lwin, dtype = int)
    if not move_window_to_GPU:
        sensor_device = sensor_device.to(device)
        time_sequence = time_sequence.to(device)

    fft_blocks = 0
    for i in range(Nwin):
        iTimes =  time_sequence + (i)*(Lwin-Loverlap)
        block = sensor_device[..., iTimes]
        if move_window_to_GPU:
            block = block.to(device)
        windowed_ts = (block) * hamming_window
        fft_res = mnt.rfft(windowed_ts,dim = -1) * scaling_term
        fft_res[..., 1:] /= 2**(1/2)
        fft_blocks += (fft_res).abs()**2/Nwin

    if return_torch:
        return mnt.ensure_torch(fft_blocks, move_to_CPU = return_on_CPU), mnt.ensure_torch(freqs, move_to_CPU = return_on_CPU)
    return ensure_numpy(fft_blocks), ensure_numpy(freqs)



def welch_psd_time_resolved_source_space(
                            time_series, fs : int, ker = None,
                            window_length = 2, overlap = 0.5,
                            device = "cpu",
                            return_torch = False, return_on_CPU = True, move_window_to_GPU = True,
    ):

    
    """ This function computes the time-resolved PSD of source-space signals using the welch method. It can be called on the source time series directly or on the sensor time series with the imaging kernel
    
        Args: 
            time_series: list/numpy array/torch tensor 
                The time series over which to compute the PSD
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            ker: list/numpy array/torch tensor 
                The imaging kernel (must be passed in only if the time_series is a sensor-space signal)
            window_length : float
                Length of the sliding window (in seconds)
            overlap : float
                Overlap percentage between windows (in 0-1 range)
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            move_window_to_device: bool
                Only useful if the device is not "cpu". Specifies whether to move only the windowed time_series to the device (slower runtime & less memory) or the whole time_series (faster runtime & more memory).
        Returns: 
            psd: numpy array (or torch tensor)               
                The PSD of the source signal
            freqs: numpy array (or torch tensor)                      
                The frequencies of the PSD.
            all_times: numpy array (or torch tensor)
                The list of start and end times for each window.
    """

    if ker is None:
        return welch_psd_time_resolved_sensor_space(time_series, fs, window_length=window_length, overlap = overlap, device = device, return_torch = return_torch, return_on_CPU=return_on_CPU)
    nTime = time_series.shape[-1]
    Lwin = round(window_length*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    ker_device = mnt.ensure_torch(ker, type_float=True).to(device).type(mnt.__default_complex_dtype__)
    sensor_device = mnt.ensure_torch(time_series, type_float=True)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)
    if not move_window_to_GPU:
        sensor_device = sensor_device.to(device)
        time_sequence = time_sequence.to(device)

        
    all_times = []
    fft_blocks = []
    for i in range(Nwin):
        iTimes =  time_sequence + (i)*(Lwin-Loverlap)
        all_times.append([iTimes.min().item(),iTimes.max().item()])
        block = sensor_device[..., iTimes]
        if move_window_to_GPU:
            block = block.to(device)
        windowed_ts = (block) * hamming_window
        fft_res = mnt.rfft(windowed_ts,dim = -1) * scaling_term
        fft_res[..., 1:] /= 2**(1/2)
        fft_blocks.append(((ker_device @ fft_res).abs()**2)[None,...])

    fft_blocks = mnt.cat(fft_blocks,dim=0)
    if return_torch:
        return mnt.ensure_torch(fft_blocks, move_to_CPU = return_on_CPU), mnt.ensure_torch(freqs, move_to_CPU = return_on_CPU), mnt.ensure_torch(all_times, move_to_CPU = return_on_CPU)
    return ensure_numpy(fft_blocks), ensure_numpy(freqs),ensure_numpy(all_times)

def welch_psd_time_resolved_sensor_space(
                            time_series, fs : int,
                            window_length = 2, overlap = 0.5,
                            device = "cpu", return_torch = False, return_on_CPU = True, move_window_to_GPU = True,
    ):
    """ This function computes the time-resolved PSD of sensor-space signals using the welch method.
    
        Args: 
            time_series: list/numpy array/torch tensor 
                The sensor time series over which to compute the PSD
            fs : int
                The signal's sampling rate.
        Keyword Args: 
            window_length : float
                Length of the sliding window (in seconds)
            overlap : float
                Overlap percentage between windows (in 0-1 range)
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            move_window_to_device: bool
                Only useful if the device is not "cpu". Specifies whether to move only the windowed time_series to the device (slower runtime & less memory) or the whole time_series (faster runtime & more memory).
        Returns: 
            psd: numpy array (or torch tensor)               
                The PSD of the sensor signal
            freqs: numpy array (or torch tensor)                      
                The frequencies of the PSD.
            all_times: numpy array (or torch tensor)
                The list of start and end times for each window.
    """

    nTime = time_series.shape[-1]
    Lwin = round(window_length*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))
    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    sensor_device = mnt.ensure_torch(time_series)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)
    if not move_window_to_GPU:
        sensor_device = sensor_device.to(device)
        time_sequence = time_sequence.to(device)
    all_times = []
    fft_blocks = []
    for i in range(Nwin):
        iTimes =  time_sequence + (i)*(Lwin-Loverlap)
        all_times.append([iTimes.min().item(),iTimes.max().item()])
        block = sensor_device[..., iTimes]
        if move_window_to_GPU:
            block = block.to(device)
        windowed_ts = (block) * hamming_window
        fft_res = mnt.rfft(windowed_ts,dim = -1) * scaling_term
        fft_res[..., 1:] /= 2**(1/2)
        fft_blocks.append(((fft_res).abs()**2)[None,...])

    fft_blocks = mnt.cat(fft_blocks,dim=0)
    if return_torch:
        return mnt.ensure_torch(fft_blocks, move_to_CPU = return_on_CPU), mnt.ensure_torch(freqs, move_to_CPU = return_on_CPU), mnt.ensure_torch(all_times, move_to_CPU = return_on_CPU)
    return ensure_numpy(fft_blocks), ensure_numpy(freqs),ensure_numpy(all_times)
