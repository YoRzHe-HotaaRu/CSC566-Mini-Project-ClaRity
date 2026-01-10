"""
Morphological Operations Module for Road Surface Layer Analyzer
Handles post-processing operations for segmentation refinement.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops

from .config import MORPHOLOGY_CONFIG


def get_structuring_element(
    shape: Literal["rect", "ellipse", "cross"] = "ellipse",
    size: int = 5
) -> np.ndarray:
    """
    Create morphological structuring element.
    
    Args:
        shape: Shape type ("rect", "ellipse", "cross")
        size: Size of kernel
        
    Returns:
        Structuring element
    """
    if shape == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        raise ValueError(f"Unknown shape: {shape}")


def erode(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "ellipse",
    iterations: int = 1
) -> np.ndarray:
    """
    Apply erosion operation.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        iterations: Number of erosion iterations
        
    Returns:
        Eroded image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "ellipse",
    iterations: int = 1
) -> np.ndarray:
    """
    Apply dilation operation.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        iterations: Number of dilation iterations
        
    Returns:
        Dilated image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)


def opening(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "ellipse"
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).
    Removes small bright regions (noise).
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        
    Returns:
        Opened image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "ellipse"
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).
    Fills small dark regions (holes).
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        
    Returns:
        Closed image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def gradient(
    image: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "ellipse"
) -> np.ndarray:
    """
    Apply morphological gradient (dilation - erosion).
    Highlights edges.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        
    Returns:
        Gradient image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def top_hat(
    image: np.ndarray,
    kernel_size: int = 9,
    kernel_shape: str = "ellipse"
) -> np.ndarray:
    """
    Apply top-hat transform (image - opening).
    Extracts bright regions smaller than structuring element.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        
    Returns:
        Top-hat transformed image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def black_hat(
    image: np.ndarray,
    kernel_size: int = 9,
    kernel_shape: str = "ellipse"
) -> np.ndarray:
    """
    Apply black-hat transform (closing - image).
    Extracts dark regions smaller than structuring element.
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        
    Returns:
        Black-hat transformed image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def fill_holes(image: np.ndarray, max_hole_size: int = 500) -> np.ndarray:
    """
    Fill holes in binary image.
    
    Args:
        image: Input binary image
        max_hole_size: Maximum size of holes to fill
        
    Returns:
        Image with holes filled
    """
    # Convert to boolean
    binary = image > 0
    
    # Remove small holes
    filled = remove_small_holes(binary, area_threshold=max_hole_size)
    
    return (filled * 255).astype(np.uint8)


def remove_small_regions(
    image: np.ndarray,
    min_size: int = 100
) -> np.ndarray:
    """
    Remove small connected regions from binary image.
    
    Args:
        image: Input binary image
        min_size: Minimum region size to keep
        
    Returns:
        Image with small regions removed
    """
    # Convert to boolean
    binary = image > 0
    
    # Remove small objects
    cleaned = remove_small_objects(binary, min_size=min_size)
    
    return (cleaned * 255).astype(np.uint8)


def clean_segmentation(
    labels: np.ndarray,
    min_region_size: int = 100,
    fill_holes: bool = True,
    max_hole_size: int = 500
) -> np.ndarray:
    """
    Clean segmentation labels by removing small regions and filling holes.
    
    Args:
        labels: Segmentation label image
        min_region_size: Minimum region size to keep
        fill_holes: Whether to fill holes
        max_hole_size: Maximum hole size to fill
        
    Returns:
        Cleaned label image
    """
    cleaned = labels.copy()
    unique_labels = np.unique(labels)
    
    for lbl in unique_labels:
        if lbl == 0:
            continue
        
        # Create binary mask for this label
        mask = (labels == lbl).astype(np.uint8) * 255
        
        # Remove small regions
        mask = remove_small_regions(mask, min_region_size)
        
        # Fill holes if requested
        if fill_holes:
            mask = fill_holes(mask, max_hole_size)
        
        # Update cleaned labels
        cleaned[labels == lbl] = 0
        cleaned[mask > 0] = lbl
    
    return cleaned


def connected_components(
    image: np.ndarray,
    connectivity: int = 8
) -> Tuple[np.ndarray, int]:
    """
    Find connected components in binary image.
    
    Args:
        image: Input binary image
        connectivity: 4 or 8 connectivity
        
    Returns:
        Tuple of (labeled image, number of components)
    """
    n_labels, labels = cv2.connectedComponents(image, connectivity=connectivity)
    return labels, n_labels


def get_region_properties(labels: np.ndarray) -> list:
    """
    Get properties of labeled regions.
    
    Args:
        labels: Labeled image
        
    Returns:
        List of region properties dictionaries
    """
    props = regionprops(labels)
    
    results = []
    for prop in props:
        results.append({
            "label": prop.label,
            "area": prop.area,
            "centroid": prop.centroid,
            "bbox": prop.bbox,
            "perimeter": prop.perimeter,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "extent": prop.extent
        })
    
    return results


def apply_morphology_pipeline(
    image: np.ndarray,
    operations: list = None,
    kernel_size: int = 5,
    kernel_shape: str = "ellipse"
) -> np.ndarray:
    """
    Apply a sequence of morphological operations.
    
    Args:
        image: Input image
        operations: List of operation names (e.g., ["opening", "closing", "dilate"])
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        
    Returns:
        Processed image
    """
    if operations is None:
        operations = ["opening", "closing"]
    
    result = image.copy()
    
    operation_map = {
        "erode": erode,
        "dilate": dilate,
        "opening": opening,
        "closing": closing,
        "gradient": gradient,
        "top_hat": top_hat,
        "black_hat": black_hat
    }
    
    for op in operations:
        if op in operation_map:
            result = operation_map[op](result, kernel_size, kernel_shape)
        else:
            raise ValueError(f"Unknown operation: {op}")
    
    return result


def refine_boundaries(
    labels: np.ndarray,
    iterations: int = 2,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Refine segment boundaries using morphological operations.
    
    Args:
        labels: Segmentation labels
        iterations: Number of refinement iterations
        kernel_size: Size of morphological kernel
        
    Returns:
        Refined labels
    """
    refined = labels.copy()
    unique_labels = np.unique(labels)
    
    for _ in range(iterations):
        for lbl in unique_labels:
            if lbl == 0:
                continue
            
            # Create mask for this label
            mask = (refined == lbl).astype(np.uint8) * 255
            
            # Apply opening then closing
            mask = opening(mask, kernel_size, "ellipse")
            mask = closing(mask, kernel_size, "ellipse")
            
            # Update refined labels
            refined[labels == lbl] = 0
            refined[mask > 0] = lbl
    
    return refined
