"""
Preprocessing Module for Road Surface Layer Analyzer
Handles noise reduction, contrast enhancement, and color space conversion.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal

from .config import PREPROCESSING_CONFIG


def apply_noise_filter(
    image: np.ndarray,
    method: Literal["gaussian", "median", "bilateral"] = "median",
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply noise reduction filter to image.
    
    Args:
        image: Input image (BGR or grayscale)
        method: Filter type - "gaussian", "median", or "bilateral"
        kernel_size: Size of filter kernel (must be odd)
        
    Returns:
        Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    
    elif method == "bilateral":
        # Bilateral filter preserves edges while smoothing
        d = kernel_size
        sigma_color = 75
        sigma_space = 75
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    else:
        raise ValueError(f"Unknown filter method: {method}")


def enhance_contrast(
    image: np.ndarray,
    method: Literal["histogram_eq", "clahe", "gamma"] = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    gamma: float = 1.0
) -> np.ndarray:
    """
    Enhance image contrast using various methods.
    
    Args:
        image: Input image (BGR or grayscale)
        method: Enhancement method - "histogram_eq", "clahe", or "gamma"
        clip_limit: CLAHE clip limit (for CLAHE method)
        tile_grid_size: CLAHE tile grid size (for CLAHE method)
        gamma: Gamma value for gamma correction (for gamma method)
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to grayscale if needed for some operations
    if len(image.shape) == 3:
        is_color = True
        # For color images, convert to LAB and enhance L channel
        if method in ["histogram_eq", "clahe"]:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            is_color = True
    else:
        is_color = False
        l_channel = image.copy()
    
    if method == "histogram_eq":
        enhanced_l = cv2.equalizeHist(l_channel)
        
        if is_color:
            lab[:, :, 0] = enhanced_l
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced_l
    
    elif method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        enhanced_l = clahe.apply(l_channel)
        
        if is_color:
            lab[:, :, 0] = enhanced_l
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced_l
    
    elif method == "gamma":
        # Gamma correction
        # gamma < 1 brightens image, gamma > 1 darkens image
        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


def convert_color_space(
    image: np.ndarray,
    target_space: Literal["gray", "hsv", "lab", "rgb"] = "gray"
) -> np.ndarray:
    """
    Convert image to different color space.
    
    Args:
        image: Input BGR image
        target_space: Target color space - "gray", "hsv", "lab", or "rgb"
        
    Returns:
        Converted image
    """
    if target_space == "gray":
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    elif target_space == "hsv":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    elif target_space == "lab":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    elif target_space == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    else:
        raise ValueError(f"Unknown color space: {target_space}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size or by scale factor.
    
    Args:
        image: Input image
        target_size: Target (width, height) tuple
        scale: Scale factor (alternative to target_size)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    elif scale is not None:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    else:
        return image


def normalize_image(
    image: np.ndarray,
    method: Literal["minmax", "zscore"] = "minmax"
) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image
        method: Normalization method - "minmax" (0-1) or "zscore"
        
    Returns:
        Normalized image as float32
    """
    image_float = image.astype(np.float32)
    
    if method == "minmax":
        min_val = image_float.min()
        max_val = image_float.max()
        if max_val - min_val > 0:
            return (image_float - min_val) / (max_val - min_val)
        return image_float
    
    elif method == "zscore":
        mean = image_float.mean()
        std = image_float.std()
        if std > 0:
            return (image_float - mean) / std
        return image_float - mean
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_image(
    image: np.ndarray,
    denoise: Optional[str] = "median",
    denoise_kernel: int = 3,
    enhance: Optional[str] = "clahe",
    enhance_clip_limit: float = 2.0,
    color_space: str = "bgr",
    resize_to: Optional[Tuple[int, int]] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Complete preprocessing pipeline for road surface images.
    
    Args:
        image: Input BGR image
        denoise: Noise filter method (None to skip)
        denoise_kernel: Kernel size for noise filter
        enhance: Contrast enhancement method (None to skip)
        enhance_clip_limit: CLAHE clip limit
        color_space: Output color space
        resize_to: Target size (width, height) or None
        normalize: Whether to normalize to 0-1 range
        
    Returns:
        Preprocessed image
    """
    result = image.copy()
    
    # Step 1: Resize if needed
    if resize_to is not None:
        result = resize_image(result, target_size=resize_to)
    
    # Step 2: Denoise
    if denoise is not None:
        result = apply_noise_filter(result, method=denoise, kernel_size=denoise_kernel)
    
    # Step 3: Enhance contrast
    if enhance is not None:
        result = enhance_contrast(result, method=enhance, clip_limit=enhance_clip_limit)
    
    # Step 4: Convert color space
    if color_space != "bgr":
        result = convert_color_space(result, target_space=color_space)
    
    # Step 5: Normalize
    if normalize:
        result = normalize_image(result, method="minmax")
    
    return result


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    info = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": float(image.min()),
        "max": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std())
    }
    
    if len(image.shape) == 3:
        info["channels"] = image.shape[2]
        info["height"] = image.shape[0]
        info["width"] = image.shape[1]
    else:
        info["channels"] = 1
        info["height"] = image.shape[0]
        info["width"] = image.shape[1]
    
    return info
