from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from torch.linalg import svd as _svd

__all__ = ["get_sign_flip_mask"]

def get_sign_flip_mask(surface_orientations, return_torch = False, return_on_CPU = True, device = "cpu"):
    """
        This function that returns a mask that specifies the sign flip to align signals (or parcel-specific subset of the kernel projection matrix) based on the geometry of the cortical surface. It is usually applied to a group of signals within a parcel (source space) after extracting them using the kernel (inverse model).
        
        Args: 
            surface_orientations: list/numpy array/torch tensor 
                The matrix of orientations for each source in the parcel (from the surface file as 'VertNormals')
        Keyword Args: 
            return_torch: bool
                Specifies if the output should be a torch tensor.
            return_on_CPU: bool
                Specifies is the output should be moved to CPU (useful only on torch backend and using GPU as device)
            device: str
                Specifies the device in which to perform SVD.
        Returns: 
            flip_mask: list/numpy array/ torch tensor
                Vector with either 1 or -1 entries that specifies the sign flip to align signals (or parcel-specific subset of the kernel projection matrix) based on the geometry of the cortical surface.
    """
    surface_orientations = ensure_torch(surface_orientations).to(device)
    U = _svd(surface_orientations, full_matrices=False)[0][:,0]
    flip_mask = (1. - (2 * (U<0)))
    flip_mask *= (1-(2*(flip_mask.mean()<0)))
    if return_torch:
        return ensure_torch(flip_mask, move_to_CPU = return_on_CPU)
    return ensure_numpy(flip_mask)
