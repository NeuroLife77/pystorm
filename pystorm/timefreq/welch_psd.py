from pystorm.utils import minitorch as mnt
from pystorm.utils.minitorch import ensure_numpy

def welch_psd_source_space(time_series, fs, ker = None,window_length=2000, overlap = 0.5, device = "cpu"):
    if ker is None:
        return welch_psd_sensor_space(time_series, fs, window_length=window_length, overlap = overlap, device = device)
    nTime = time_series.shape[-1]
    Lwin = round(window_length/1000*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    if time_series.dtype == mnt.float32:
        mnt.set_minitorch_default_dtype("float32")
        print(mnt.__default_complex_dtype__)
    ker_cuda = mnt.ensure_torch(ker, type_float=True).to(device).type(mnt.__default_complex_dtype__)
    sensor_cuda = mnt.ensure_torch(time_series, type_float=True).to(device)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)

    fft_blocks = 0
    for i in range(Nwin):
        iTimes =  time_sequence + (i - 1)*(Lwin-Loverlap)
        block = sensor_cuda[:,iTimes]
        source_ts = (block) * hamming_window
        fft_res = mnt.rfft(source_ts,dim = -1) * scaling_term
        fft_res[:,1:] /= 2**(1/2)
        fft_blocks += (ker_cuda @ fft_res).abs()**2/Nwin
    return ensure_numpy(fft_blocks), ensure_numpy(freqs)

def welch_psd_sensor_space(time_series, fs, window_length=2000, overlap = 0.5, device = "cpu"):
    nTime = time_series.shape[-1]
    Lwin = round(window_length/1000*fs)
    Loverlap = round(Lwin * overlap)
    Lwin = Lwin - Lwin%2
    Nwin = int((nTime - Loverlap)/(Lwin-Loverlap))

    freqs = mnt.rfftfreq(Lwin,1/fs).to(device)
    sensor_cuda = mnt.ensure_torch(time_series).to(device)
    hamming_window = (0.54 - 0.46 * mnt.cos(mnt.linspace(0,2*mnt.pi,Lwin))).to(device)

    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = (2/(win_noise_power_gain*fs)).sqrt()
    time_sequence = mnt.arange(0,Lwin, dtype = int)

    fft_blocks = 0
    for i in range(Nwin):
        iTimes =  time_sequence + (i - 1)*(Lwin-Loverlap)
        block = sensor_cuda[:,iTimes]
        source_ts = (block) * hamming_window
        fft_res = mnt.rfft(source_ts,dim = -1) * scaling_term
        fft_res[:,1:] /= 2**(1/2)
        fft_blocks += (fft_res).abs()**2/Nwin
    return ensure_numpy(fft_blocks), ensure_numpy(freqs)