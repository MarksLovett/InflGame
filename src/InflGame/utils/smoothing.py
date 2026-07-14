"""
.. module:: smoothing
   :synopsis: Provides surface smoothing utilities for 2D gridded data in influencer games visualizations.

Smoothing Module
================

This module provides simple 2D smoothing functions for gridded surfaces used in influencer games visualizations.
It includes Gaussian and median filtering capabilities with automatic fallback to NumPy-based implementations
when SciPy is not available.

The module is designed to work with the `InflGame` package and supports smoothing of heat maps, contour plots,
and other 2D surface visualizations to improve visual quality and reduce noise.

Dependencies:
-------------
- numpy (required)
- scipy.ndimage (optional, recommended for better performance)

Usage:
------
The smoothing functions accept and return NumPy arrays of shape :math:`(H, W)` where :math:`H` is height
and :math:`W` is width. When SciPy is available, optimized filters are used; otherwise, NumPy-based
fallback implementations are employed.

Examples
--------

.. code-block:: python
    
    from InflGame.utils.smoothing import gaussian_smooth, median_smooth
    import numpy as np

    # Create a noisy 2D surface
    surface = np.random.rand(100, 100)
    
    # Apply Gaussian smoothing
    smoothed = gaussian_smooth(surface, sigma=2.0)
    
    # Apply median filtering
    filtered = median_smooth(surface, kernel_size=5)
    
    # Check if SciPy is available
    from InflGame.utils.smoothing import try_import_scipy
    has_scipy = try_import_scipy()
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from scipy.ndimage import gaussian_filter, median_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def gaussian_smooth(surface: np.ndarray, sigma: float = 1.0, truncate: float = 4.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 2D surface.
    
    Performs a Gaussian blur operation on a 2D grid using either SciPy's optimized
    implementation or a NumPy-based separable convolution fallback. The Gaussian kernel
    is defined by its standard deviation (sigma) and is truncated beyond a specified
    number of standard deviations.

    Parameters
    ----------
    surface : np.ndarray
        2D numpy array of shape :math:`(H, W)` representing the surface to smooth.
    sigma : float, optional
        Standard deviation for the Gaussian kernel, measured in grid units, by default 1.0.
        Larger values produce more smoothing.
    truncate : float, optional
        Truncate the Gaussian filter at this many standard deviations, by default 4.0.
        The filter radius will be ``int(truncate * sigma + 0.5)``.

    Returns
    -------
    np.ndarray
        Smoothed 2D numpy array of the same shape as input.
        
    Raises
    ------
    ValueError
        If surface is not a 2D array.
        
    Notes
    -----
    When SciPy is available, this function uses :func:`scipy.ndimage.gaussian_filter` for
    optimal performance. Otherwise, it falls back to a separable 1D Gaussian kernel convolution
    using NumPy, which is slower but produces similar results.
    
    The Gaussian kernel is defined as:
    
    .. math::
        G(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{x^2}{2\\sigma^2}\\right)
    
    Examples
    --------
    >>> import numpy as np
    >>> surface = np.random.rand(50, 50)
    >>> smoothed = gaussian_smooth(surface, sigma=2.0)
    >>> smoothed.shape
    (50, 50)
    """
    if not isinstance(surface, np.ndarray):
        surface = np.asarray(surface)
    if surface.ndim != 2:
        raise ValueError("gaussian_smooth expects a 2D array")

    if _HAS_SCIPY:
        return gaussian_filter(surface, sigma=sigma, truncate=truncate)

    # NumPy fallback: separable Gaussian via 1D kernel convolution
    # Create 1D kernel
    radius = int(truncate * sigma + 0.5)
    if radius <= 0:
        return surface.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Convolve along rows then columns using np.apply_along_axis + np.convolve
    temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=surface)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=temp)
    return out


def median_smooth(surface: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filtering to a 2D surface.
    
    Performs median filtering on a 2D grid using either SciPy's optimized implementation
    or a NumPy-based sliding window fallback. Median filtering is particularly effective
    at removing salt-and-pepper noise while preserving edges.

    Parameters
    ----------
    surface : np.ndarray
        2D numpy array of shape :math:`(H, W)` representing the surface to filter.
    kernel_size : int, optional
        Size of the square filtering window, by default 3. Must be an odd positive integer.
        Larger kernel sizes provide more aggressive noise reduction but may blur fine details.

    Returns
    -------
    np.ndarray
        Filtered 2D numpy array of the same shape as input.
        
    Raises
    ------
    ValueError
        If surface is not a 2D array.
    ValueError
        If kernel_size is not an odd positive integer.
        
    Notes
    -----
    When SciPy is available, this function uses :func:`scipy.ndimage.median_filter` for
    optimal performance. The NumPy fallback uses edge-padded sliding windows and is
    significantly slower for large arrays but produces identical results.
    
    The median operation for each pixel is computed over a :math:`k \\times k` window
    centered on that pixel, where :math:`k` = kernel_size.
    
    Examples
    --------
    >>> import numpy as np
    >>> surface = np.random.rand(50, 50)
    >>> # Add some noise
    >>> surface[10, 10] = 100
    >>> surface[20, 20] = -100
    >>> # Remove noise with median filter
    >>> filtered = median_smooth(surface, kernel_size=5)
    >>> filtered.shape
    (50, 50)
    """
    if not isinstance(surface, np.ndarray):
        surface = np.asarray(surface)
    if surface.ndim != 2:
        raise ValueError("median_smooth expects a 2D array")

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be an odd positive integer")

    if _HAS_SCIPY:
        return median_filter(surface, size=kernel_size)

    # NumPy fallback: sliding-window median (slow but simple)
    pad = kernel_size // 2
    padded = np.pad(surface, pad, mode='edge')
    H, W = surface.shape
    out = np.empty_like(surface)
    for i in range(H):
        for j in range(W):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            out[i, j] = np.median(window)
    return out


def try_import_scipy() -> bool:
    """
    Check if SciPy is available in the current environment.
    
    This function returns the availability status of SciPy's ndimage module, which is used
    for optimized filtering operations. When SciPy is not available, the smoothing functions
    automatically fall back to NumPy-based implementations.

    Returns
    -------
    bool
        True if SciPy is available and can be imported, False otherwise.
        
    Examples
    --------
    >>> has_scipy = try_import_scipy()
    >>> if has_scipy:
    ...     print("Using optimized SciPy filters")
    ... else:
    ...     print("Using NumPy fallback implementations")
    """
    return _HAS_SCIPY
