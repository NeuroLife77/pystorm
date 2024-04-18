from pystorm.utils import minitorch as mnt
from pystorm.utils.minitorch import ensure_numpy
from sys import stderr 
__all__ = ["welch_psd_source_space", "welch_psd_sensor_space"]

def welch_psd_source_space(
                            time_series, fs : int, ker = None,
                            window_length = 2, overlap = 0.5,
                            device = "cpu"
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
        Returns: 
            psd: numpy array                       
                The PSD of the source signal
            freqs: numpy array                       
                The frequencies of the PSD.
    """

    if ker is None:
        return welch_psd_sensor_space(time_series, fs, window_length=window_length, overlap = overlap, device = device)
    nTime = time_series.shape[-1]
    Lwin = round(window_length*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    ker_cuda = mnt.ensure_torch(ker, type_float=True).to(device).type(mnt.__default_complex_dtype__)
    sensor_cuda = mnt.ensure_torch(time_series, type_float=True)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)

    fft_blocks = 0
    for i in range(Nwin):
        iTimes =  time_sequence + (i - 1)*(Lwin-Loverlap)
        block = sensor_cuda[..., iTimes].to(device)
        windowed_ts = (block) * hamming_window
        fft_res = mnt.rfft(windowed_ts,dim = -1) * scaling_term
        fft_res[..., 1:] /= 2**(1/2)
        fft_blocks += (ker_cuda @ fft_res).abs()**2/Nwin
        
    return ensure_numpy(fft_blocks), ensure_numpy(freqs)

def welch_psd_sensor_space(
                            time_series, fs : int,
                            window_length = 2, overlap = 0.5,
                            device = "cpu"
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
        Returns: 
            psd: numpy array                       
                The PSD of the sensor signal
            freqs: numpy array                       
                The frequencies of the PSD.
    """

    nTime = time_series.shape[-1]
    Lwin = round(window_length*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    sensor_cuda = mnt.ensure_torch(time_series)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)

    fft_blocks = 0
    for i in range(Nwin):
        iTimes =  time_sequence + (i - 1)*(Lwin-Loverlap)
        block = sensor_cuda[..., iTimes].to(device)
        windowed_ts = (block) * hamming_window
        fft_res = mnt.rfft(windowed_ts,dim = -1) * scaling_term
        fft_res[..., 1:] /= 2**(1/2)
        fft_blocks += (fft_res).abs()**2/Nwin

    return ensure_numpy(fft_blocks), ensure_numpy(freqs)