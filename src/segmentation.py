"""
Segmentation Module for Road Surface Layer Analyzer
Implements K-Means, Watershed, Superpixels, and thresholding methods.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal, List
from sklearn.cluster import KMeans
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.filters import sobel
from skimage.measure import label, regionprops

from .config import SEGMENTATION_CONFIG


# =============================================================================
# K-Means Clustering Segmentation
# =============================================================================

def kmeans_segment(
    image: np.ndarray,
    n_clusters: int = 5,
    max_iter: int = 300,
    n_init: int = 10,
    use_spatial: bool = False,
    spatial_weight: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment image using K-Means clustering.
    
    Args:
        image: Input image (BGR or grayscale)
        n_clusters: Number of clusters (default 5 for road layers)
        max_iter: Maximum iterations
        n_init: Number of initializations
        use_spatial: Include spatial coordinates in features
        spatial_weight: Weight for spatial features (0-1)
        
    Returns:
        Tuple of (labels, cluster_centers)
    """
    # Get image dimensions
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
        image = image.reshape(h, w, 1)
    
    # Reshape to pixel features
    pixels = image.reshape(-1, c).astype(np.float32)
    
    # Add spatial coordinates if requested
    if use_spatial:
        # Create normalized spatial coordinates
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        xx = (xx.reshape(-1, 1) / w).astype(np.float32)
        yy = (yy.reshape(-1, 1) / h).astype(np.float32)
        
        # Weight spatial features
        spatial = np.hstack([xx, yy]) * spatial_weight * 255
        pixels = np.hstack([pixels, spatial])
    
    # Apply K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=42
    )
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image shape
    labels = labels.reshape(h, w)
    
    return labels, kmeans.cluster_centers_


def kmeans_with_texture(
    image: np.ndarray,
    texture_features: np.ndarray,
    n_clusters: int = 5,
    texture_weight: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment using K-Means with both color and texture features.
    
    Args:
        image: Input BGR image
        texture_features: Texture feature map (same spatial size)
        n_clusters: Number of clusters
        texture_weight: Weight for texture features
        
    Returns:
        Tuple of (labels, cluster_centers)
    """
    h, w = image.shape[:2]
    
    # Flatten color features
    if len(image.shape) == 3:
        color_features = image.reshape(-1, 3).astype(np.float32)
    else:
        color_features = image.reshape(-1, 1).astype(np.float32)
    
    # Flatten and scale texture features
    if len(texture_features.shape) == 3:
        texture_flat = texture_features.reshape(-1, texture_features.shape[-1])
    else:
        texture_flat = texture_features.reshape(-1, 1)
    
    # Resize texture features if needed
    if texture_flat.shape[0] != color_features.shape[0]:
        # Interpolate texture features to match image size
        texture_resized = cv2.resize(
            texture_features, 
            (w, h), 
            interpolation=cv2.INTER_LINEAR
        )
        if len(texture_resized.shape) == 3:
            texture_flat = texture_resized.reshape(-1, texture_resized.shape[-1])
        else:
            texture_flat = texture_resized.reshape(-1, 1)
    
    texture_flat = texture_flat.astype(np.float32) * texture_weight
    
    # Combine features
    combined = np.hstack([color_features, texture_flat])
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(combined)
    
    return labels.reshape(h, w), kmeans.cluster_centers_


# =============================================================================
# Superpixel Segmentation (SLIC)
# =============================================================================

def slic_segment(
     image: np.ndarray,
     n_segments: int = 200,
     compactness: float = 10.0,
     sigma: float = 1.0,
     start_label: int = 1
) -> np.ndarray:
    """
    Segment image using SLIC superpixels.
    
    Args:
        image: Input image (RGB)
        n_segments: Approximate number of superpixels
        compactness: Balances color proximity and space proximity
        sigma: Width of Gaussian smoothing kernel
        start_label: Starting label value
        
    Returns:
        Superpixel labels
    """
    # Convert to RGB if BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=start_label,
        channel_axis=-1 if len(image.shape) == 3 else None
    )
    
    return segments


def superpixel_to_region(
    image: np.ndarray,
    segments: np.ndarray
) -> np.ndarray:
    """
    Create region-averaged image from superpixels.
    
    Args:
        image: Input image
        segments: Superpixel labels
        
    Returns:
        Region-averaged image
    """
    output = np.zeros_like(image, dtype=np.float32)
    
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                output[:, :, c][mask] = image[:, :, c][mask].mean()
        else:
            output[mask] = image[mask].mean()
    
    return output.astype(image.dtype)


# =============================================================================
# Watershed Segmentation
# =============================================================================

def watershed_segment(
    image: np.ndarray,
    markers: Optional[np.ndarray] = None,
    gradient_threshold: float = 0.1
) -> np.ndarray:
    """
    Segment image using Watershed algorithm.
    
    Args:
        image: Input grayscale image
        markers: Optional marker array for seeds
        gradient_threshold: Threshold for gradient-based markers
        
    Returns:
        Watershed labels
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute gradient
    gradient = sobel(gray.astype(np.float64))
    
    # Create markers if not provided
    if markers is None:
        # Threshold gradient to find seeds
        markers = np.zeros_like(gray, dtype=np.int32)
        markers[gradient < gradient_threshold] = 1
        markers = label(markers)
    
    # Apply watershed
    labels = watershed(gradient, markers)
    
    return labels


# =============================================================================
# Thresholding Methods
# =============================================================================

def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's automatic thresholding.
    
    Args:
        image: Grayscale image
        
    Returns:
        Tuple of (binary image, threshold value)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    threshold, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary, threshold


def adaptive_threshold(
    image: np.ndarray,
    method: Literal["mean", "gaussian"] = "gaussian",
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """
    Apply adaptive thresholding.
    
    Args:
        image: Grayscale image
        method: "mean" or "gaussian" adaptive method
        block_size: Size of local neighborhood (must be odd)
        c: Constant subtracted from mean
        
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if block_size % 2 == 0:
        block_size += 1
    
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    
    binary = cv2.adaptiveThreshold(
        image, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c
    )
    
    return binary


def multi_level_threshold(
    image: np.ndarray,
    n_levels: int = 5
) -> np.ndarray:
    """
    Apply multi-level thresholding for multiple regions.
    
    Args:
        image: Grayscale image
        n_levels: Number of threshold levels (creates n_levels+1 regions)
        
    Returns:
        Labeled image with levels 0 to n_levels
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute threshold values using histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    
    # Find threshold values that divide histogram into equal parts
    cumsum = np.cumsum(hist)
    total = cumsum[-1]
    thresholds = []
    
    for i in range(1, n_levels):
        target = total * i / n_levels
        threshold_idx = np.searchsorted(cumsum, target)
        thresholds.append(threshold_idx)
    
    # Apply thresholds
    labels = np.zeros_like(image, dtype=np.uint8)
    thresholds = [0] + thresholds + [256]
    
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i + 1])
        labels[mask] = i
    
    return labels


# =============================================================================
# Edge Detection
# =============================================================================

def detect_edges(
    image: np.ndarray,
    method: Literal["sobel", "canny", "prewitt", "laplacian"] = "canny",
    threshold1: float = 50,
    threshold2: float = 150
) -> np.ndarray:
    """
    Detect edges in image.
    
    Args:
        image: Input image
        method: Edge detection method
        threshold1: Lower threshold (for Canny)
        threshold2: Upper threshold (for Canny)
        
    Returns:
        Edge image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == "canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
    
    elif method == "sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    
    elif method == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(gray.astype(np.float64), -1, kernel_x)
        prewitt_y = cv2.filter2D(gray.astype(np.float64), -1, kernel_y)
        edges = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    
    elif method == "laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(edges)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    return edges


# =============================================================================
# Utility Functions
# =============================================================================

def create_segmentation_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create overlay visualization of segmentation.
    
    Args:
        image: Original image
        labels: Segmentation labels
        alpha: Overlay transparency
        colors: List of BGR colors for each label
        
    Returns:
        Overlay image
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Generate colors if not provided
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    if colors is None:
        # Generate distinct colors
        colors = []
        for i in range(n_labels):
            hue = int(180 * i / n_labels)
            color = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
            colors.append(tuple(map(int, color[0, 0])))
    
    # Create colored label image
    overlay = np.zeros_like(image)
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = colors[i % len(colors)]
        overlay[mask] = color
    
    # Blend with original
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def get_segment_properties(labels: np.ndarray) -> List[dict]:
    """
    Get properties of each segment.
    
    Args:
        labels: Segmentation labels
        
    Returns:
        List of dictionaries with segment properties
    """
    props = regionprops(labels)
    
    results = []
    for prop in props:
        results.append({
            "label": prop.label,
            "area": prop.area,
            "centroid": prop.centroid,
            "bbox": prop.bbox,
            "perimeter": prop.perimeter if hasattr(prop, 'perimeter') else 0,
            "eccentricity": prop.eccentricity if hasattr(prop, 'eccentricity') else 0
        })
    
    return results
