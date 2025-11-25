"""Surface smoothing utilities.

Provides simple 2D smoothing functions for gridded surfaces. Uses SciPy if
available for higher-quality filters; falls back to NumPy + convolution if not.

Functions:
- gaussian_smooth(surface, sigma=1.0, truncate=4.0): Gaussian blur
- median_smooth(surface, kernel_size=3): Median filter (numpy fallback only)

The functions accept and return numpy arrays of shape (H, W).
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
    """Apply a Gaussian smoothing to a 2D surface.

    Parameters
    - surface: 2D numpy array (H, W)
    - sigma: standard deviation for Gaussian kernel (in grid units)
    - truncate: truncate filter at this many standard deviations

    Returns
    - smoothed 2D numpy array of same shape as input
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
    """Apply a median filter to a 2D surface.

    Parameters
    - surface: 2D numpy array (H, W)
    - kernel_size: size of the square window (must be odd)

    Returns
    - filtered 2D numpy array
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
    """Return True if SciPy is available in this environment."""
    return _HAS_SCIPY
