"""
Descriptors Module for Road Surface Layer Analyzer
Implements boundary and region descriptors for shape analysis.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from skimage.measure import regionprops, label


# =============================================================================
# Boundary Representations
# =============================================================================

def extract_boundary(
    binary_image: np.ndarray,
    method: str = "contour"
) -> List[np.ndarray]:
    """
    Extract boundaries from binary image.
    
    Args:
        binary_image: Binary image
        method: Extraction method ("contour" or "gradient")
        
    Returns:
        List of boundary point arrays
    """
    if method == "contour":
        contours, _ = cv2.findContours(
            binary_image.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        return [c.squeeze() for c in contours if len(c) > 2]
    
    elif method == "gradient":
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.dilate(binary_image, kernel) - cv2.erode(binary_image, kernel)
        coords = np.where(gradient > 0)
        return [np.column_stack((coords[1], coords[0]))]
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_chain_code(
    boundary: np.ndarray,
    connectivity: int = 8
) -> np.ndarray:
    """
    Compute Freeman chain code for boundary.
    
    Args:
        boundary: Boundary points as (N, 2) array
        connectivity: 4 or 8 connectivity
        
    Returns:
        Chain code array
    """
    if len(boundary) < 2:
        return np.array([])
    
    # Direction lookup for 8-connectivity
    # Direction: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    if connectivity == 8:
        directions = {
            (1, 0): 0,   # East
            (1, -1): 1,  # Northeast
            (0, -1): 2,  # North
            (-1, -1): 3, # Northwest
            (-1, 0): 4,  # West
            (-1, 1): 5,  # Southwest
            (0, 1): 6,   # South
            (1, 1): 7    # Southeast
        }
    else:  # 4-connectivity
        directions = {
            (1, 0): 0,   # East
            (0, -1): 1,  # North
            (-1, 0): 2,  # West
            (0, 1): 3    # South
        }
    
    chain_code = []
    
    for i in range(len(boundary) - 1):
        dx = np.sign(boundary[i + 1][0] - boundary[i][0])
        dy = np.sign(boundary[i + 1][1] - boundary[i][1])
        
        direction = (int(dx), int(dy))
        
        if direction in directions:
            chain_code.append(directions[direction])
    
    return np.array(chain_code)


def normalize_chain_code(chain_code: np.ndarray) -> np.ndarray:
    """
    Normalize chain code for rotation invariance.
    Uses first difference method.
    
    Args:
        chain_code: Original chain code
        
    Returns:
        Normalized (first difference) chain code
    """
    if len(chain_code) < 2:
        return chain_code
    
    # Compute first difference
    diff = np.diff(chain_code)
    # Handle wrap-around (8 directions)
    diff = (diff + 8) % 8
    
    # Find minimum integer for starting point invariance
    diff_str = ''.join(map(str, diff))
    min_rotation = min([diff_str[i:] + diff_str[:i] for i in range(len(diff_str))])
    
    return np.array([int(c) for c in min_rotation])


def chain_code_histogram(
    chain_code: np.ndarray,
    connectivity: int = 8
) -> np.ndarray:
    """
    Compute histogram of chain code directions.
    
    Args:
        chain_code: Chain code array
        connectivity: 4 or 8 connectivity
        
    Returns:
        Normalized histogram
    """
    n_bins = connectivity
    hist, _ = np.histogram(chain_code, bins=n_bins, range=(0, n_bins), density=True)
    return hist


# =============================================================================
# Fourier Descriptors
# =============================================================================

def compute_fourier_descriptors(
    boundary: np.ndarray,
    n_descriptors: int = None
) -> np.ndarray:
    """
    Compute Fourier descriptors of boundary.
    
    Args:
        boundary: Boundary points as (N, 2) array
        n_descriptors: Number of descriptors to return (None = all)
        
    Returns:
        Fourier descriptor array
    """
    if len(boundary) < 4:
        return np.array([])
    
    # Convert boundary to complex representation
    complex_boundary = boundary[:, 0] + 1j * boundary[:, 1]
    
    # Compute FFT
    descriptors = np.fft.fft(complex_boundary)
    
    # Take magnitude for rotation invariance
    descriptors = np.abs(descriptors)
    
    # Normalize by first descriptor (scale invariance)
    if descriptors[0] != 0:
        descriptors = descriptors / descriptors[0]
    
    if n_descriptors is not None:
        descriptors = descriptors[:n_descriptors]
    
    return descriptors


def reconstruct_from_fourier(
    descriptors: np.ndarray,
    n_points: int,
    n_harmonics: int = None
) -> np.ndarray:
    """
    Reconstruct boundary from Fourier descriptors.
    
    Args:
        descriptors: Fourier descriptors
        n_points: Number of points in reconstruction
        n_harmonics: Number of harmonics to use
        
    Returns:
        Reconstructed boundary points
    """
    if n_harmonics is None:
        n_harmonics = len(descriptors)
    
    # Limit harmonics
    truncated = np.zeros_like(descriptors)
    truncated[:min(n_harmonics, len(descriptors))] = descriptors[:min(n_harmonics, len(descriptors))]
    
    # Inverse FFT
    reconstructed = np.fft.ifft(truncated)
    
    # Convert to (N, 2) array
    boundary = np.column_stack((reconstructed.real, reconstructed.imag))
    
    # Resample to desired number of points
    if len(boundary) != n_points:
        indices = np.linspace(0, len(boundary) - 1, n_points).astype(int)
        boundary = boundary[indices]
    
    return boundary


# =============================================================================
# Region Descriptors
# =============================================================================

def compute_region_descriptors(
    labeled_image: np.ndarray,
    intensity_image: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Compute descriptors for labeled regions.
    
    Args:
        labeled_image: Labeled region image
        intensity_image: Optional intensity image for additional features
        
    Returns:
        List of region descriptor dictionaries
    """
    props = regionprops(labeled_image, intensity_image=intensity_image)
    
    results = []
    for prop in props:
        region_dict = {
            # Basic measurements
            "label": prop.label,
            "area": prop.area,
            "perimeter": prop.perimeter,
            "centroid": prop.centroid,
            "bbox": prop.bbox,
            
            # Shape descriptors
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "extent": prop.extent,
            "orientation": prop.orientation,
            
            # Computed descriptors
            "compactness": compute_compactness(prop.area, prop.perimeter),
            "circularity": compute_circularity(prop.area, prop.perimeter),
            "aspect_ratio": compute_aspect_ratio(prop.bbox),
            
            # Moments
            "moments_hu": prop.moments_hu.tolist() if hasattr(prop, 'moments_hu') else None
        }
        
        # Intensity features if available
        if intensity_image is not None:
            region_dict["mean_intensity"] = prop.mean_intensity
            region_dict["min_intensity"] = prop.min_intensity
            region_dict["max_intensity"] = prop.max_intensity
        
        results.append(region_dict)
    
    return results


def compute_compactness(area: float, perimeter: float) -> float:
    """
    Compute compactness (4 * pi * area / perimeter^2).
    Circle = 1, more complex shapes > 1.
    
    Args:
        area: Region area
        perimeter: Region perimeter
        
    Returns:
        Compactness value
    """
    if perimeter == 0:
        return 0
    return (perimeter ** 2) / (4 * np.pi * area)


def compute_circularity(area: float, perimeter: float) -> float:
    """
    Compute circularity (inverse of compactness, normalized).
    Circle = 1, non-circular < 1.
    
    Args:
        area: Region area
        perimeter: Region perimeter
        
    Returns:
        Circularity value (0-1)
    """
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def compute_aspect_ratio(bbox: Tuple) -> float:
    """
    Compute aspect ratio from bounding box.
    
    Args:
        bbox: Bounding box (min_row, min_col, max_row, max_col)
        
    Returns:
        Aspect ratio (width / height)
    """
    min_row, min_col, max_row, max_col = bbox
    height = max_row - min_row
    width = max_col - min_col
    
    if height == 0:
        return 0
    return width / height


def compute_euler_number(binary_image: np.ndarray) -> int:
    """
    Compute Euler number (C - H where C = components, H = holes).
    
    Args:
        binary_image: Binary image
        
    Returns:
        Euler number
    """
    # Label connected components
    labeled = label(binary_image)
    n_components = labeled.max()
    
    # Count holes (background regions inside objects)
    labeled_bg = label(1 - binary_image)
    # Exclude the main background (label 1)
    n_holes = labeled_bg.max() - 1
    
    return n_components - n_holes


# =============================================================================
# Combined Descriptor Extraction
# =============================================================================

def extract_all_descriptors(
    binary_image: np.ndarray,
    intensity_image: Optional[np.ndarray] = None
) -> Dict:
    """
    Extract all descriptors from binary image.
    
    Args:
        binary_image: Binary image
        intensity_image: Optional grayscale image
        
    Returns:
        Dictionary with all descriptors
    """
    # Label regions
    labeled = label(binary_image)
    
    # Get boundaries
    boundaries = extract_boundary(binary_image)
    
    descriptors = {
        "n_regions": labeled.max(),
        "euler_number": compute_euler_number(binary_image),
        "regions": [],
        "boundaries": []
    }
    
    # Region descriptors
    descriptors["regions"] = compute_region_descriptors(labeled, intensity_image)
    
    # Boundary descriptors for each boundary
    for i, boundary in enumerate(boundaries):
        if len(boundary) > 3:
            chain = compute_chain_code(boundary)
            fourier = compute_fourier_descriptors(boundary, n_descriptors=20)
            
            boundary_desc = {
                "index": i,
                "n_points": len(boundary),
                "chain_code_length": len(chain),
                "chain_code_histogram": chain_code_histogram(chain).tolist(),
                "fourier_descriptors": fourier.tolist()[:10]  # First 10
            }
            descriptors["boundaries"].append(boundary_desc)
    
    return descriptors


def compare_shapes(
    shape1_descriptors: Dict,
    shape2_descriptors: Dict
) -> float:
    """
    Compare two shapes using their descriptors.
    
    Args:
        shape1_descriptors: Descriptors of first shape
        shape2_descriptors: Descriptors of second shape
        
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    # Compare Fourier descriptors if available
    if "fourier_descriptors" in shape1_descriptors and "fourier_descriptors" in shape2_descriptors:
        fd1 = np.array(shape1_descriptors["fourier_descriptors"])
        fd2 = np.array(shape2_descriptors["fourier_descriptors"])
        
        # Pad to same length
        max_len = max(len(fd1), len(fd2))
        fd1_padded = np.zeros(max_len)
        fd2_padded = np.zeros(max_len)
        fd1_padded[:len(fd1)] = fd1
        fd2_padded[:len(fd2)] = fd2
        
        # Euclidean distance
        distance = np.linalg.norm(fd1_padded - fd2_padded)
        
        # Convert to similarity
        similarity = 1 / (1 + distance)
        return similarity
    
    return 0.0
