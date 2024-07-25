from pystorm import mnt

__all__ = ["get_morlet_coefs"]
def _get_morlet(freq, sigma_tc,time_vals):
    wavelet = (1/(sigma_tc*(mnt.as_tensor(mnt.pi).sqrt())).sqrt()) * ( -(time_vals**2)/(2*sigma_tc**2)).exp() * (1j*2*mnt.pi*freq*time_vals).exp()
    return wavelet

def _convolve_same(
                            signal, win,
                            convolve_type = "auto", device="cpu",
                            return_on_CPU = False, return_torch=True
    ):


    signal = mnt.ensure_torch(signal).to(device)
    win = mnt.ensure_torch(win).to(device)
    
    convolution_output = mnt.convolve(signal,win, mode = "same")
    
    if return_torch:
        return mnt.ensure_torch(convolution_output, move_to_CPU = return_on_CPU)
    return mnt.ensure_numpy(convolution_output)

def get_morlet_coefs(
        signals, freqs, fs,
        central_freq = 1, full_width_half_max=3,precision=3,
        return_power = False,
        device = "cpu", 
        return_torch=False, return_on_CPU = True, 
):
    """ This function returns the complex coefficients (or power) of the convolution of signals with morlet wavelets.
    
        Args: 
            signals: list/numpy array/torch tensor 
                The signals to convolve the wavelets with.
            freqs: list/numpy array/torch tensor 
                The list of frequencies for the wavelets.
            fs: int
                The sampling rate (Hz).
        Keyword Args: 
            center_freq: float
                The central frequency of complex Morlet wavelet in Hz
            full_width_half_max: float
                The full width half max of complex Morlet wavelet in time.
            precision: float
                Multiples of the standard deviation to use when defining the wavelet kernels size.
            return_power: bool
                Specifies whether to return the power of the coefficients (coefs.abs()**2) or the complex coefficients.
            device: str
                Specifies the device in which to apply the filtering.
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
        Returns: 
            coef: numpy array (or torch tensor)                  
                The wavelet coefficients (or their power).
    """
    freqs = mnt.ensure_torch(freqs)
    signal = mnt.ensure_torch(signals+0j).to(device)
    scales  = freqs / central_freq
    sigma_tc = full_width_half_max/(8*(mnt.as_tensor(2).log())).sqrt()
    sigma_t = sigma_tc / scales
    coefs = []
    for s in range(scales.shape[0]):
        x_val = mnt.arange(-precision * sigma_t[s],precision * sigma_t[s],1/fs) * scales[s]
        wavelet = _get_morlet(central_freq,sigma_tc,x_val) * scales[s].sqrt()
        coef = _convolve_same(signal,wavelet.view(*[1 for _ in range(len(signal.shape[:-1]))],-1), device = device) * 1/fs
        coefs.append(coef[...,None,:])
    coefs = mnt.cat(coefs,dim = -2)
    if return_power:
        coefs = coefs.abs()**2
    if return_torch:
        return mnt.ensure_torch(coefs, move_to_CPU = return_on_CPU)
    return mnt.ensure_numpy(coefs)