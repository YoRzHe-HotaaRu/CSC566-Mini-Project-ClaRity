"""
Texture Feature Extraction Module for Road Surface Layer Analyzer
Implements GLCM, LBP, Gabor filters, and statistical texture measures.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel

from .config import GLCM_CONFIG, LBP_CONFIG, GABOR_CONFIG


# =============================================================================
# GLCM (Gray-Level Co-occurrence Matrix) Features
# =============================================================================

def compute_glcm(
    image: np.ndarray,
    distances: List[int] = None,
    angles: List[int] = None,
    levels: int = 256,
    symmetric: bool = True,
    normed: bool = True
) -> np.ndarray:
    """
    Compute Gray-Level Co-occurrence Matrix.
    
    Args:
        image: Grayscale image (uint8)
        distances: List of pixel pair distances
        angles: List of angles in degrees
        levels: Number of gray levels
        symmetric: If True, matrix is symmetric
        normed: If True, normalize the matrix
        
    Returns:
        GLCM matrix
    """
    if distances is None:
        distances = GLCM_CONFIG["distances"]
    if angles is None:
        angles = GLCM_CONFIG["angles"]
    
    # Convert angles from degrees to radians
    angles_rad = [np.deg2rad(a) for a in angles]
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    
    # Reduce levels if needed for efficiency
    if levels < 256:
        image = (image // (256 // levels)).astype(np.uint8)
    
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles_rad,
        levels=levels,
        symmetric=symmetric,
        normed=normed
    )
    
    return glcm


def extract_glcm_features(
    image: np.ndarray,
    distances: List[int] = None,
    angles: List[int] = None,
    levels: int = 256
) -> Dict[str, float]:
    """
    Extract GLCM texture features from image.
    
    Args:
        image: Grayscale image
        distances: List of pixel pair distances
        angles: List of angles in degrees
        levels: Number of gray levels
        
    Returns:
        Dictionary with GLCM features (contrast, energy, homogeneity, correlation, entropy)
    """
    # Compute GLCM
    glcm = compute_glcm(image, distances, angles, levels)
    
    # Extract properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    asm = graycoprops(glcm, 'ASM').mean()
    
    # Compute entropy manually
    glcm_norm = glcm / (glcm.sum() + 1e-10)
    entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))
    
    return {
        "contrast": float(contrast),
        "dissimilarity": float(dissimilarity),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
        "correlation": float(correlation),
        "asm": float(asm),
        "entropy": float(entropy)
    }


def extract_glcm_features_windowed(
    image: np.ndarray,
    window_size: int = 32,
    stride: int = 16,
    distances: List[int] = None,
    angles: List[int] = None
) -> np.ndarray:
    """
    Extract GLCM features using sliding window approach.
    
    Args:
        image: Grayscale image
        window_size: Size of sliding window
        stride: Step size for window movement
        distances: GLCM distances
        angles: GLCM angles
        
    Returns:
        Feature map with GLCM features at each window position
    """
    h, w = image.shape[:2]
    n_windows_h = (h - window_size) // stride + 1
    n_windows_w = (w - window_size) // stride + 1
    
    # 7 GLCM features
    feature_map = np.zeros((n_windows_h, n_windows_w, 7), dtype=np.float32)
    
    for i in range(n_windows_h):
        for j in range(n_windows_w):
            y = i * stride
            x = j * stride
            window = image[y:y+window_size, x:x+window_size]
            
            features = extract_glcm_features(window, distances, angles, levels=64)
            feature_map[i, j] = [
                features["contrast"],
                features["dissimilarity"],
                features["homogeneity"],
                features["energy"],
                features["correlation"],
                features["asm"],
                features["entropy"]
            ]
    
    return feature_map


# =============================================================================
# LBP (Local Binary Pattern) Features
# =============================================================================

def compute_lbp(
    image: np.ndarray,
    radius: int = None,
    n_points: int = None,
    method: str = None
) -> np.ndarray:
    """
    Compute Local Binary Pattern image.
    
    Args:
        image: Grayscale image
        radius: Radius of circle for sampling
        n_points: Number of sampling points
        method: LBP method ('default', 'ror', 'uniform', 'nri_uniform', 'var')
        
    Returns:
        LBP image
    """
    if radius is None:
        radius = LBP_CONFIG["radius"]
    if n_points is None:
        n_points = LBP_CONFIG["n_points"]
    if method is None:
        method = LBP_CONFIG["method"]
    
    # Ensure image is float
    if image.dtype == np.uint8:
        image = image.astype(np.float64)
    
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    
    return lbp


def extract_lbp_histogram(
    image: np.ndarray,
    radius: int = None,
    n_points: int = None,
    method: str = "uniform",
    n_bins: int = None
) -> np.ndarray:
    """
    Extract LBP histogram features.
    
    Args:
        image: Grayscale image
        radius: LBP radius
        n_points: Number of sampling points
        method: LBP method
        n_bins: Number of histogram bins
        
    Returns:
        Normalized LBP histogram
    """
    if radius is None:
        radius = LBP_CONFIG["radius"]
    if n_points is None:
        n_points = LBP_CONFIG["n_points"]
    
    # Number of bins for uniform LBP
    if n_bins is None:
        if method == "uniform":
            n_bins = n_points + 2
        else:
            n_bins = 2 ** n_points
    
    lbp = compute_lbp(image, radius, n_points, method)
    
    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_lbp_features(
    image: np.ndarray,
    radius: int = None,
    n_points: int = None
) -> Dict[str, float]:
    """
    Extract LBP-based texture features.
    
    Args:
        image: Grayscale image
        radius: LBP radius
        n_points: Number of sampling points
        
    Returns:
        Dictionary with LBP features
    """
    hist = extract_lbp_histogram(image, radius, n_points, method="uniform")
    
    # Compute statistics from histogram
    uniformity = np.sum(hist ** 2)  # Energy/uniformity
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return {
        "histogram": hist,
        "uniformity": float(uniformity),
        "entropy": float(entropy),
        "mean": float(np.mean(hist)),
        "std": float(np.std(hist))
    }


# =============================================================================
# Gabor Filter Features
# =============================================================================

def create_gabor_kernels(
    frequencies: List[float] = None,
    orientations: List[float] = None,
    sigma: float = None
) -> List[np.ndarray]:
    """
    Create bank of Gabor filter kernels.
    
    Args:
        frequencies: List of spatial frequencies
        orientations: List of orientations in degrees
        sigma: Standard deviation of Gaussian envelope
        
    Returns:
        List of Gabor kernels
    """
    if frequencies is None:
        frequencies = GABOR_CONFIG["frequencies"]
    if orientations is None:
        orientations = GABOR_CONFIG["orientations"]
    if sigma is None:
        sigma = GABOR_CONFIG["sigma"]
    
    kernels = []
    for freq in frequencies:
        for theta in orientations:
            theta_rad = np.deg2rad(theta)
            kernel = gabor_kernel(freq, theta=theta_rad, sigma_x=sigma, sigma_y=sigma)
            kernels.append(kernel)
    
    return kernels


def apply_gabor_filter(
    image: np.ndarray,
    kernel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gabor filter to image.
    
    Args:
        image: Grayscale image
        kernel: Gabor kernel
        
    Returns:
        Tuple of (magnitude, phase) response
    """
    # Apply filter (real and imaginary parts)
    filtered_real = ndimage.convolve(image.astype(np.float64), np.real(kernel), mode='wrap')
    filtered_imag = ndimage.convolve(image.astype(np.float64), np.imag(kernel), mode='wrap')
    
    magnitude = np.sqrt(filtered_real ** 2 + filtered_imag ** 2)
    phase = np.arctan2(filtered_imag, filtered_real)
    
    return magnitude, phase


def extract_gabor_features(
    image: np.ndarray,
    frequencies: List[float] = None,
    orientations: List[float] = None
) -> Dict[str, np.ndarray]:
    """
    Extract Gabor filter features from image.
    
    Args:
        image: Grayscale image
        frequencies: List of spatial frequencies
        orientations: List of orientations
        
    Returns:
        Dictionary with Gabor features
    """
    kernels = create_gabor_kernels(frequencies, orientations)
    
    means = []
    stds = []
    energies = []
    
    for kernel in kernels:
        magnitude, _ = apply_gabor_filter(image, kernel)
        means.append(magnitude.mean())
        stds.append(magnitude.std())
        energies.append(np.sum(magnitude ** 2))
    
    return {
        "means": np.array(means),
        "stds": np.array(stds),
        "energies": np.array(energies),
        "mean_of_means": float(np.mean(means)),
        "mean_of_stds": float(np.mean(stds)),
        "total_energy": float(np.sum(energies))
    }


# =============================================================================
# Statistical Texture Features
# =============================================================================

def extract_statistical_features(image: np.ndarray) -> Dict[str, float]:
    """
    Extract basic statistical texture features.
    
    Args:
        image: Grayscale image
        
    Returns:
        Dictionary with statistical features
    """
    # Flatten image
    pixels = image.ravel().astype(np.float64)
    
    # Basic statistics
    mean = np.mean(pixels)
    std = np.std(pixels)
    variance = np.var(pixels)
    
    # Normalized variance (smoothness)
    smoothness = 1 - 1 / (1 + variance)
    
    # Third and fourth moments
    if std > 0:
        skewness = np.mean(((pixels - mean) / std) ** 3)
        kurtosis = np.mean(((pixels - mean) / std) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0
    
    # Entropy
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255), density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return {
        "mean": float(mean),
        "std": float(std),
        "variance": float(variance),
        "smoothness": float(smoothness),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "entropy": float(entropy)
    }


# =============================================================================
# Combined Feature Extraction
# =============================================================================

def extract_all_texture_features(
    image: np.ndarray,
    use_glcm: bool = True,
    use_lbp: bool = True,
    use_gabor: bool = False,
    use_statistical: bool = True
) -> Dict[str, any]:
    """
    Extract all texture features from image.
    
    Args:
        image: Grayscale image
        use_glcm: Whether to extract GLCM features
        use_lbp: Whether to extract LBP features
        use_gabor: Whether to extract Gabor features
        use_statistical: Whether to extract statistical features
        
    Returns:
        Dictionary with all extracted features
    """
    features = {}
    
    if use_glcm:
        features["glcm"] = extract_glcm_features(image)
    
    if use_lbp:
        lbp_features = extract_lbp_features(image)
        # Remove histogram from dict for JSON serialization
        features["lbp"] = {k: v for k, v in lbp_features.items() if k != "histogram"}
        features["lbp_histogram"] = lbp_features["histogram"].tolist()
    
    if use_gabor:
        gabor_features = extract_gabor_features(image)
        features["gabor"] = {
            "mean_of_means": gabor_features["mean_of_means"],
            "mean_of_stds": gabor_features["mean_of_stds"],
            "total_energy": gabor_features["total_energy"]
        }
    
    if use_statistical:
        features["statistical"] = extract_statistical_features(image)
    
    return features


def features_to_vector(features: Dict) -> np.ndarray:
    """
    Convert feature dictionary to flat feature vector.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        1D numpy array of feature values
    """
    vector = []
    
    if "glcm" in features:
        glcm = features["glcm"]
        vector.extend([
            glcm["contrast"],
            glcm["dissimilarity"],
            glcm["homogeneity"],
            glcm["energy"],
            glcm["correlation"],
            glcm["asm"],
            glcm["entropy"]
        ])
    
    if "lbp" in features:
        lbp = features["lbp"]
        vector.extend([
            lbp["uniformity"],
            lbp["entropy"],
            lbp["mean"],
            lbp["std"]
        ])
    
    if "gabor" in features:
        gabor = features["gabor"]
        vector.extend([
            gabor["mean_of_means"],
            gabor["mean_of_stds"],
            gabor["total_energy"]
        ])
    
    if "statistical" in features:
        stats = features["statistical"]
        vector.extend([
            stats["mean"],
            stats["std"],
            stats["variance"],
            stats["smoothness"],
            stats["skewness"],
            stats["kurtosis"],
            stats["entropy"]
        ])
    
    return np.array(vector)
