from pystorm.utils.minitorch import ensure_numpy, ensure_torch
from torch.linalg import svd as _svd
import pystorm.utils.minitorch as mnt
from torch.linalg import eigh as _eigh
__all__ = ["get_sign_flip_mask", "get_scout_time_series", "get_data_cov"]

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


def get_data_cov(signal, device = "cpu"):
    signal_compute =  mnt.ensure_torch(signal).to(device)
    signal_mean_centered = signal_compute - signal_compute.mean(-1, keepdims=True)
    data_cov = signal_mean_centered @ signal_mean_centered.T 
    return data_cov


def get_scout_time_series(kernels, signal, collapse_function, device = "cpu", **kwargs):
    """
        This function that returns a mask that specifies the sign flip to align signals (or parcel-specific subset of the kernel projection matrix) based on the geometry of the cortical surface. It is usually applied to a group of signals within a parcel (source space) after extracting them using the kernel (inverse model).
        
        Args: 
            kernels: list/numpy array/torch tensor
                The kernel or list of kernels to project the sensor data onto the source space (or parcel-specific source space)
                    -If the collapse function is none the kernels should be of shape [nSources, nSensors].
                    -Otherwise it should be a list of kernels (or an array of objects containing kernels) of length k with each element in the list being a matrix of shape [nSource_in_parcel, nSensors]. Given the variable number of sources per parcel it cannot be a single tensor/array of numbers. It can be a list of matrices or it can be a tensor/array of objects which elements are the matrices.
            signal: list/numpy array/torch tensor 
                The sensor space signal
            collapse_function: str
                The function used to collapse the multiple signals from all sources within a parcel into a single signal
        Keyword Args: 
            device: str
                Specifies the device in which to perform SVD.
            kwargs: Expect to potentially receive the following additional arguments
                reference_PC: numpy array (or torch tensor) 
                    An array that aims to serve as reference to align the PCs across windows (or trials).
                data_cov: numpy array (or torch tensor) 
                    The sensor data covariance matrix (will compute it if not included)
                n_comp: int 
                    The number of principal components to use to reconstruct the parcel time series (Brainstorm only use n_comp=1).
        Returns: 
            parcellated_signal: list/numpy array/torch tensor 
                The parcellated source space signal 
            reference_PC: numpy array (or torch tensor) 
                Will return an array that aims to serve as reference to align the PCs across windows (or trials). Only happens if there is no 'reference_PC' key in the kwargs dict. 
                    -In an iterative process (looping over time windows or trials): Collect the reference_PC on the first call of the function and pass it as keyword argument over the next function calls.
    
    """

    
    signal_compute =  mnt.ensure_torch(signal).to(device)
    if collapse_function is None:
        parcellated_signal = mnt.ensure_torch(kernels).to(device) @ signal_compute
    elif collapse_function == "mean":
        parcellated_signal = mnt.zeros(len(kernels),signal_compute.shape[-1], device = device)
        for scout_index in range(parcellated_signal.shape[0]):
            parcellated_signal[scout_index] =  (mnt.ensure_torch(kernels[scout_index]).to(device) @ signal_compute).mean(0)
    elif collapse_function == "pca":
        if "reference_PC" in kwargs:
            reference_PC = mnt.ensure_torch(kwargs["reference_PC"]).to(device)
        else:
            try:
                reference_PC = mnt.ones(kernels.shape[0], device = device)
            except:
                reference_PC = mnt.ones(len(kernels), device = device)
        
        parcellated_signal = mnt.zeros(len(kernels),signal_compute.shape[-1], device = device)
        explained =  mnt.zeros(len(kernels))
        if "data_cov" in kwargs:
            data_cov = mnt.ensure_torch(kwargs["data_cov"]).to(device)
        else:  
            data_cov = get_data_cov(signal_compute, device = device)
        for scout_index in range(parcellated_signal.shape[0]):
            ker_compute = mnt.ensure_torch(kernels[scout_index]).to(device)
            scout_data_cov =  ker_compute @ data_cov @ ker_compute.T
            scout_data_cov = (scout_data_cov + scout_data_cov.T)/2
            eigvals, eigvects = _eigh(scout_data_cov)
            n_comp = 1
            
            if "n_comp" in kwargs:
                n_comp = kwargs["n_comp"]
            
            PCA_components=(eigvects[...,-n_comp:].T) /((ker_compute.shape[0]**(1/2)))
            explained[scout_index] = (eigvals[-n_comp:]/eigvals.sum()).sum()
            if "reference_PC" not in kwargs:
                reference_PC[scout_index] = 2*(PCA_components.sum(0).sum(0)> 1).int()-1
            PCA_projection = reference_PC[scout_index] *(PCA_components[...,None] * ker_compute).sum(0)
            parcellated_signal[scout_index] = (PCA_projection @ signal_compute).sum(0)
            
        if "reference_PC" not in kwargs:
            return parcellated_signal, reference_PC
    return parcellated_signal
