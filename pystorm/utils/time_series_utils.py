from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from torch.linalg import svd as _svd

__all__ = ["align_signals_with_sign_flip"]

def align_signals_with_sign_flip(signal, return_torch = False, return_on_CPU = True, device = "cpu"):
    """
        This function aligns multiple signals together by flipping the signs of some of them based on the sign of entries in the U matrix derived using SVD. It is usually applied to a group of signals within a parcel (source space) after extracting them using the kernel (inverse model).
        
        Args: 
            signal: list/numpy array/torch tensor 
                The signal to filter
        Keyword Args: 
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            device: str
                Specifies the device in which to apply the filtering.
    """
    signal = ensure_torch(signal).to(device)
    U = _svd(signal, full_matrices=False)[0][:,0]
    flip = (1. - (2 * (U<0)))
    flip *= (1-(2*(flip.mean()<0)))
    flipped_signal = (signal * flip[:,None])
    if return_torch:
        return ensure_torch(flipped_signal, move_to_CPU = return_on_CPU)
    return ensure_numpy(flipped_signal)
