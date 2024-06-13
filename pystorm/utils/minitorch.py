"""
A module that serves as a lightweight pytorch++ (also has some numpy<->pytorch conversion utils, and some torchaudio functions).

It is intended to be a quality-of-life module that groups most pytorch-related functions that are used repeatedly within the package and to possibly minimize GPU RAM usage from imports when executing jobs with multiple processes running on the GPU in parallel.

The functions are 'copied' and not just imported to have their original docstrings available.
"""

from torch.fft import rfft as _rfft
from torch.fft import rfftfreq as _rfftfreq 
from torch.fft import fft as _fft
from torch.fft import ifft as _ifft
from torch.fft import fftfreq as _fftfreq 
from torch import cat as _cat
from torch import linspace as _linspace
from torch import pi as _pi
from torch import arange as _arange
from torch import zeros as _zeros
from torch import zeros_like as  _zeros_like
from torch import ones_like as  _ones_like
from torch import ones as _ones
from torch import as_tensor as _as_tensor
from torch import from_numpy as _from_numpy
from torch import complex64, complex128
from torch import cos as _cos
from torch import sin as _sin
from torch import eye as _eye
from torch import argwhere as _argwhere
from torch import logical_or as _logical_or
from torch import logical_and as _logical_and
from torchaudio.functional import convolve as _convolve
from torchaudio.functional import fftconvolve as _fftconvolve
from torch import float16,float32,float64
from torch import set_default_dtype as _set_default_dtype
from numpy import ndarray as _ndarray
from numpy import float32 as _np_float32
from numpy import float64 as _np_float64
from numpy import complex64 as _np_complex64
from numpy import complex128 as _np_complex128
from numpy import array as _np_array
from torch import Tensor as _Tensor
from torch.cuda import mem_get_info as _mem_get_info

def _check_available_memory():
    return _mem_get_info()[0]

pi = _pi
__all__=[
        "rfft","rfftfreq","fft","ifft","fftfreq","convolve","fftconvolve",
        "linspace","arange","argwhere",
        "cat","zeros","ones","zeros_like","ones_like","eye",
        "as_tensor","from_numpy","ensure_numpy","ensure_torch",
        "cos","pi", "logical_and", "logical_or",
        "float16","float32","float64","complex64","complex128","_np_float32","_np_float64",
        "set_minitorch_default_dtype","__default_dtype__","__default_complex_dtype__","_check_available_memory"
]
__default_dtype__ = float64
__default_complex_dtype__ = complex128
__default_np_dtype__ = _np_float64
__default_np_complex_dtype__ = _np_complex128
__default_dtype_str__ = "float64"
__default_complex_dtype_str__ = "complex128"

_set_default_dtype(__default_dtype__)
def set_minitorch_default_dtype(default_type: str = "float64"):
    """ This function allows for specifying a floating point precision (and its matching complex precision). Default is float64 and complex128.

        Args: 
            None
        Keyword Args: 
            default_type: str     
                String that specifies the default floating point type of the pytorch backend
        Returns: 
            None

    """

    global __default_dtype__, __default_complex_dtype__, __default_np_dtype__, __default_np_complex_dtype__, __default_dtype_str__, __default_complex_dtype_str__
    if default_type == "float16":
        __default_dtype__ = float16
        __default_dtype_str__ = "float16"
        print("Precision Type Warning [minitorch]: Most functions don't support this type (for now).")
    elif default_type == "float32":
        __default_dtype__ = float32
        __default_np_dtype__ = _np_float32
        __default_np_complex_dtype__ = _np_complex64
        __default_complex_dtype__ = complex64
        __default_dtype_str__ = "float32"
        __default_complex_dtype_str__ = "complex64"
    else:
        __default_dtype__ = float64
        __default_np_dtype__ = _np_float64
        __default_np_complex_dtype__ = _np_complex128
        __default_complex_dtype__ = complex128
        __default_dtype_str__ = "float64"
        __default_complex_dtype_str__ = "complex128"
    _set_default_dtype(__default_dtype__)
    print("Type Warning [set_minitorch_default_dtype()]: All previously defined tensors or arrays might have incompatible types with the new default, this could cause some functions to crash, especially those that depend on numba.")
    

def ensure_torch(x, type_float: bool = False, type_complex: bool = False, move_to_CPU = False):
    """ This function ensures that the variable is a torch tensor. It optionally also ensures that it is of the default type (as set by 'set_minitorch_default_dtype').
    
        Args: 
            x: list/numpy array/torch tensor/primitive type that can be contained in a torch tensor (e.g., float, int, bool)
                Input variable.
        Keyword Args: 
            type_float: bool          
                Specifies whether to set the type of the tensor.
            type_complex: bool        
                Specifies if the type of the tensor is meant to be complex
        Returns:
            x: Torch tensor
                The input ensured to be a torch tensor.

    """
    dtype_setting = __default_dtype__
    if type_complex:
        dtype_setting = __default_complex_dtype__
    if isinstance(x, _Tensor):
        if move_to_CPU:
            x = x.cpu()
        if not type_float:
            return x
        try:
            x = x.type(dtype_setting)
            return x
        except:
            pass
    try:
        x = _as_tensor(x)
        if type_float:
            x = x.type(dtype_setting)
        return x
    except:
        try:
            x = _from_numpy(x)
            if type_float:
                x = x.type(dtype_setting)
            return x
        except:
            pass
    return x


def ensure_numpy(x, type_float: bool = False, type_complex: bool = False):
    """ This function ensures that the variable is a numpy array.  It optionally also ensures that it is of the default type (as set by 'set_minitorch_default_dtype')
    
        Args: 
            x: list/numpy array/torch tensor/primitive type that can be contained in a numpy array (e.g., float, int, bool)
                Input variable.
        Keyword Args: 
            type_float: bool          
                Specifies whether to set the type of the tensor.
            type_complex: bool        
                Specifies if the type of the tensor is meant to be complex
        Returns: 
            x: numpy array (or original variable if casting failed)                 
                The input ensured to be a numpy array.

    """ 
    dtype_setting = __default_np_dtype__
    if type_complex:
        dtype_setting = __default_np_complex_dtype__
    if isinstance(x, _ndarray):
        if not type_float:
            return x
        try:
            x = x.astype(dtype_setting)
            return x
        except:
            pass
    try:
        x = x.detach()
    except:
        pass
    try:
        x = x.cpu()
    except:
        pass
    try:
        x = x.numpy()
    except:
        pass
    if type_float:
        try:
            x = x.astype(dtype_setting)
            return x
        except:
            pass
    try:
        x = _np_array(x)
        if type_float:
            x = x.astype(dtype_setting)
        return x
    except:
        pass
    return x

def _ensure_torch(x, type_float: bool = False, type_complex: bool = False):
    """ This function ensures that the variable is a torch tensor. It optionally also ensures that it is of the default type (as set by 'set_minitorch_default_dtype'). Differs from 'ensure_torch' by returning False if it could not cast it as a tensor instead of returning the input.
    
        Args: 
            x: list/numpy array/torch tensor/primitive type that can be contained in a torch tensor (e.g., float, int, bool)
                Input variable.
        Keyword Args: 
            type_float: bool          
                Specifies whether to set the type of the tensor.
            type_complex: bool        
                Specifies if the type of the tensor is meant to be complex
        Returns:
            casting_success: bool
                Boolean that specifies if casting succeeded or not. 
            x: Torch tensor
                The input ensured to be a torch tensor.

    """
    dtype_setting = __default_dtype__
    if type_complex:
        dtype_setting = __default_complex_dtype__
    if isinstance(x, _Tensor):
        if not type_float:
            return True, x
        try:
            x = x.type(dtype_setting)
            return True, x
        except:
            pass
    try:
        x = _as_tensor(x)
        if type_float:
            x = x.type(dtype_setting)
        return True, x
    except:
        try:
            x = _from_numpy(x)
            if type_float:
                x = x.type(dtype_setting)
            return True, x
        except:
            pass
    return False, x

def eye(*args, **kwargs):
    """
    eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): the number of rows
        m (int, optional): the number of columns with default being :attr:`n`

    Keyword arguments:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.

    Returns:
        Tensor: A 2-D tensor with ones on the diagonal and zeros elsewhere
        """
    return _eye(*args,**kwargs)

def zeros(*args, **kwargs):
    """
    zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Returns a tensor filled with the scalar value `0`, with the shape defined
    by the variable argument :attr:`size`.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
            """
    return _zeros(*args,**kwargs)

def ones(*args, **kwargs):
    """
    ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Returns a tensor filled with the scalar value `1`, with the shape defined
    by the variable argument :attr:`size`.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword arguments:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        """
    return _ones(*args,**kwargs)

def cat(*args, **kwargs):
    """
    cat(tensors, dim=0, *, out=None) -> Tensor

    Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating
    dimension) or be empty.

    :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
    and :func:`torch.chunk`.

    :func:`torch.cat` can be best understood via examples.

    .. seealso::

        :func:`torch.stack` concatenates the given sequence along a new dimension.

    Args:
        tensors (sequence of Tensors): any python sequence of tensors of the same type.
            Non-empty tensors provided must have the same shape, except in the
            cat dimension.
        dim (int, optional): the dimension over which the tensors are concatenated

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _cat(*args,**kwargs)

def arange(*args, **kwargs):
    """
    arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`
    with values from the interval ``[start, end)`` taken with common difference
    :attr:`step` beginning from `start`.

    Note that non-integer :attr:`step` is subject to floating point rounding errors when
    comparing against :attr:`end`; to avoid inconsistency, we advise subtracting a small epsilon from :attr:`end`
    in such cases.

    .. math::
        \text{out}_{{i+1}} = \text{out}_{i} + \text{step}

    Args:
        start (Number): the starting value for the set of points. Default: ``0``.
        end (Number): the ending value for the set of points
        step (Number): the gap between each pair of adjacent points. Default: ``1``.

    Keyword args:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). If `dtype` is not given, infer the data type from the other input
            arguments. If any of `start`, `end`, or `stop` are floating-point, the
            `dtype` is inferred to be the default dtype, see
            :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
            be `torch.int64`.
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.

    """
    return _arange(*args,**kwargs)

def linspace(*args, **kwargs):
    """
    linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
    spaced from :attr:`start` to :attr:`end`, inclusive. That is, the value are:

    .. math::
        (\text{start},
        \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \ldots,
        \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \text{end})


    From PyTorch 1.11 linspace requires the steps argument. Use steps=100 to restore the previous behavior.

    Args:
        start (float or Tensor): the starting value for the set of points. If `Tensor`, it must be 0-dimensional
        end (float or Tensor): the ending value for the set of points. If `Tensor`, it must be 0-dimensional
        steps (int): size of the constructed tensor

    Keyword arguments:
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the data type to perform the computation in.
            Default: if None, uses the global default dtype (see torch.get_default_dtype())
            when both :attr:`start` and :attr:`end` are real,
            and corresponding complex dtype when either is complex.
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
    """
    return _linspace(*args,**kwargs)

def ones_like(*args, **kwargs):
    """
    ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

    Returns a tensor filled with the scalar value `1`, with the same size as
    :attr:`input`. ``torch.ones_like(input)`` is equivalent to
    ``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

    .. warning::
        As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
        the old ``torch.ones_like(input, out=output)`` is equivalent to
        ``torch.ones(input.size(), out=output)``.

    Args:
        input (Tensor): the size of :attr:`input` will determine size of the output tensor.

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: if ``None``, defaults to the dtype of :attr:`input`.
        layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
            Default: if ``None``, defaults to the layout of :attr:`input`.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, defaults to the device of :attr:`input`.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.preserve_format``.

    """
    return _ones_like(*args,**kwargs)

def zeros_like(*args, **kwargs):
    """
    zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

    Returns a tensor filled with the scalar value `0`, with the same size as
    :attr:`input`. ``torch.zeros_like(input)`` is equivalent to
    ``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

    .. warning::
        As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
        the old ``torch.zeros_like(input, out=output)`` is equivalent to
        ``torch.zeros(input.size(), out=output)``.

    Args:
        input (Tensor): the size of :attr:`input` will determine size of the output tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: if ``None``, defaults to the dtype of :attr:`input`.
        layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
            Default: if ``None``, defaults to the layout of :attr:`input`.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, defaults to the device of :attr:`input`.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.preserve_format``.

    """
    return _zeros_like(*args,**kwargs)

def as_tensor(*args,**kwargs):
    """
    as_tensor(data, dtype=None, device=None) -> Tensor

    Converts :attr:`data` into a tensor, sharing data and preserving autograd
    history if possible.

    If :attr:`data` is already a tensor with the requested dtype and device
    then :attr:`data` itself is returned, but if :attr:`data` is a
    tensor with a different dtype or device then it's copied as if using
    `data.to(dtype=dtype, device=device)`.

    If :attr:`data` is a NumPy array (an ndarray) with the same dtype and device then a
    tensor is constructed using :func:`torch.from_numpy`.

    .. seealso::

        :func:`torch.tensor` never shares its data and creates a new "leaf tensor" (see :doc:`/notes/autograd`).


    Args:
        data (array_like): Initial data for the tensor. Can be a list, tuple,
            NumPy ``ndarray``, scalar, and other types.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, infers data type from :attr:`data`.
        device (:class:`torch.device`, optional): the device of the constructed tensor. If None and data is a tensor
            then the device of data is used. If None and data is not a tensor then
            the result tensor is constructed on the current device.

    """
    return _as_tensor(*args,**kwargs)


def from_numpy(*args,**kwargs):
    """
        
    from_numpy(ndarray) -> Tensor

    Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

    The returned tensor and :attr:`ndarray` share the same memory. Modifications to
    the tensor will be reflected in the :attr:`ndarray` and vice versa. The returned
    tensor is not resizable.

    It currently accepts :attr:`ndarray` with dtypes of ``numpy.float64``,
    ``numpy.float32``, ``numpy.float16``, ``numpy.complex64``, ``numpy.complex128``,
    ``numpy.int64``, ``numpy.int32``, ``numpy.int16``, ``numpy.int8``, ``numpy.uint8``,
    and ``bool``.

    .. warning::
        Writing to a tensor created from a read-only NumPy array is not supported and will result in undefined behavior.

    """
    return _from_numpy(*args,**kwargs)

def rfft(*args,**kwargs):
    """
    rfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

    Computes the one dimensional Fourier transform of real-valued :attr:`input`.

    The FFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])`` so
    the output contains only the positive frequencies below the Nyquist frequency.
    To compute the full output, use :func:`~torch.fft.fft`

    Note:
        Supports torch.half on CUDA with GPU Architecture SM53 or greater.
        However it only supports powers of 2 signal length in every transformed dimension.

    Args:
        input (Tensor): the real input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the real FFT.
        dim (int, optional): The dimension along which to take the one dimensional real FFT.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`~torch.fft.rfft`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Calling the backward transform (:func:`~torch.fft.irfft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`~torch.fft.irfft`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _rfft(*args,**kwargs)
    
def rfftfreq(*args,**kwargs):
    """
    rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Computes the sample frequencies for :func:`~torch.fft.rfft` with a signal of size :attr:`n`.

    Note:
        :func:`~torch.fft.rfft` returns Hermitian one-sided output, so only the
        positive frequency terms are returned. For a real FFT of length :attr:`n`
        and with inputs spaced in length unit :attr:`d`, the frequencies are::

            f = torch.arange((n + 1) // 2) / (d * n)

    Note:
        For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
        either negative or positive. Unlike :func:`~torch.fft.fftfreq`,
        :func:`~torch.fft.rfftfreq` always returns it as positive.

    Args:
        n (int): the real FFT length
        d (float, optional): The sampling length scale.
            The spacing between individual samples of the FFT input.
            The default assumes unit spacing, dividing that result by the actual
            spacing gives the result in physical frequency units.

    Keyword Args:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
    """
    return _rfftfreq(*args,**kwargs)

def cos(*args,**kwargs):
    """
        
    cos(input, *, out=None) -> Tensor

    Returns a new tensor with the cosine  of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _cos(*args,**kwargs)

def sin(*args,**kwargs):
    """
        
    sin(input, *, out=None) -> Tensor

    Returns a new tensor with the sine  of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sin(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _sin(*args,**kwargs)


def fft(*args,**kwargs):
    """  
    fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

    Computes the one dimensional discrete Fourier transform of :attr:`input`.

    Note:
        The Fourier domain representation of any real signal satisfies the
        Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
        the positive and negative frequency terms even though, for real inputs, the
        negative frequencies are redundant. :func:`~torch.fft.rfft` returns the
        more compact one-sided representation where only the positive frequencies
        are returned.

    Note:
        Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
        However it only supports powers of 2 signal length in every transformed dimension.

    Args:
        input (Tensor): the input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the FFT.
        dim (int, optional): The dimension along which to take the one dimensional FFT.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`~torch.fft.fft`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Calling the backward transform (:func:`~torch.fft.ifft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`~torch.fft.ifft`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _fft(*args,**kwargs)

def ifft(*args,**kwargs):
    """
    ifft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

    Computes the one dimensional inverse discrete Fourier transform of :attr:`input`.

    Note:
        Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
        However it only supports powers of 2 signal length in every transformed dimension.

    Args:
        input (Tensor): the input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the IFFT.
        dim (int, optional): The dimension along which to take the one dimensional IFFT.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`~torch.fft.ifft`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

            Calling the forward transform (:func:`~torch.fft.fft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`~torch.fft.ifft`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _ifft(*args,**kwargs)

def fftfreq(*args,**kwargs):
    """
    fftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

    Computes the discrete Fourier Transform sample frequencies for a signal of size :attr:`n`.

    Note:
        By convention, :func:`~torch.fft.fft` returns positive frequency terms
        first, followed by the negative frequencies in reverse order, so that
        ``f[-i]`` for all :math:`0 < i \leq n/2`` in Python gives the negative
        frequency terms. For an FFT of length :attr:`n` and with inputs spaced in
        length unit :attr:`d`, the frequencies are::

            f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

    Note:
        For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
        either negative or positive. :func:`~torch.fft.fftfreq` follows NumPy's
        convention of taking it to be negative.

    Args:
        n (int): the FFT length
        d (float, optional): The sampling length scale.
            The spacing between individual samples of the FFT input.
            The default assumes unit spacing, dividing that result by the actual
            spacing gives the result in physical frequency units.

    Keyword Args:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.

    """
    return _fftfreq(*args,**kwargs)


def convolve(*args, **kwargs):
    r"""
    Convolves inputs along their last dimension using the direct method.
    Note that, in contrast to :meth:`torch.nn.functional.conv1d`, which actually applies the valid cross-correlation
    operator, this function applies the true `convolution`_ operator.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        x (torch.Tensor): First convolution operand, with shape `(..., N)`.
        y (torch.Tensor): Second convolution operand, with shape `(..., M)`
            (leading dimensions must be broadcast-able with those of ``x``).
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
        the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    return _convolve(*args,**kwargs)

    
def fftconvolve(*args, **kwargs):
    r"""
    Convolves inputs along their last dimension using FFT. For inputs with large last dimensions, this function
    is generally much faster than :meth:`convolve`.
    Note that, in contrast to :meth:`torch.nn.functional.conv1d`, which actually applies the valid cross-correlation
    operator, this function applies the true `convolution`_ operator.
    Also note that this function can only output float tensors (int tensor inputs will be cast to float).

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        x (torch.Tensor): First convolution operand, with shape `(..., N)`.
        y (torch.Tensor): Second convolution operand, with shape `(..., M)`
            (leading dimensions must be broadcast-able with those of ``x``).
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
        the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    return _fftconvolve(*args,**kwargs)


def argwhere(*args, **kwargs):
    """
    argwhere(input) -> Tensor

    Returns a tensor containing the indices of all non-zero elements of
    :attr:`input`.  Each row in the result contains the indices of a non-zero
    element in :attr:`input`. The result is sorted lexicographically, with
    the last index changing the fastest (C-style).

    If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
    :attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
    non-zero elements in the :attr:`input` tensor.

    .. note::
        This function is similar to NumPy's `argwhere`.

        When :attr:`input` is on CUDA, this function causes host-device synchronization.

    Args:
        {input}

    """
    return _argwhere(*args,**kwargs)

def logical_and(*args,**kwargs):
    """
    logical_and(input, other, *, out=None) -> Tensor

    Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are
    treated as ``True``.

    Args:
        input (Tensor): the input tensor.
        other (Tensor): the tensor to compute AND with

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _logical_and(*args,**kwargs)

def logical_or(*args,**kwargs):
    """
    logical_or(input, other, *, out=None) -> Tensor

    Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
    treated as ``True``.

    Args:
        input (Tensor): the input tensor.
        other (Tensor): the tensor to compute OR with

    Keyword args:
        out (Tensor, optional): the output tensor.
    """
    return _logical_or(*args,**kwargs)