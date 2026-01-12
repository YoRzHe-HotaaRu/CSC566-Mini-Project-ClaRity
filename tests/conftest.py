"""
Pytest Configuration and Fixtures
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    # Create a synthetic texture image
    np.random.seed(42)
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return image


@pytest.fixture
def sample_color_image():
    """Create a sample color (BGR) image for testing."""
    np.random.seed(42)
    # Create structured regions instead of pure random noise
    # This helps SLIC superpixels find meaningful segments
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Region 1: Red-ish area (top-left)
    image[0:128, 0:128, 2] = 150  # Red channel
    image[0:128, 0:128, 1] = 50
    image[0:128, 0:128, 0] = 50
    
    # Region 2: Green-ish area (top-right)
    image[0:128, 128:256, 2] = 50
    image[0:128, 128:256, 1] = 150  # Green channel
    image[0:128, 128:256, 0] = 50
    
    # Region 3: Blue-ish area (bottom-left)
    image[128:256, 0:128, 2] = 50
    image[128:256, 0:128, 1] = 50
    image[128:256, 0:128, 0] = 150  # Blue channel
    
    # Region 4: Mixed/white area (bottom-right)
    image[128:256, 128:256] = 180
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, (256, 256, 3)).clip(-20, 20).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


@pytest.fixture
def sample_textured_image():
    """Create an image with distinct textures for testing segmentation."""
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Region 1: Smooth (low texture)
    image[0:128, 0:128] = 50
    
    # Region 2: Medium texture
    np.random.seed(42)
    image[0:128, 128:256] = np.random.normal(128, 20, (128, 128)).clip(0, 255).astype(np.uint8)
    
    # Region 3: High texture (rough)
    image[128:256, 0:128] = np.random.normal(200, 50, (128, 128)).clip(0, 255).astype(np.uint8)
    
    # Region 4: Gradient
    gradient = np.tile(np.linspace(0, 255, 128), (128, 1)).T.astype(np.uint8)
    image[128:256, 128:256] = gradient
    
    return image


@pytest.fixture
def sample_binary_image():
    """Create a sample binary image for testing."""
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Create some shapes
    cv2.circle(image, (128, 128), 50, 255, -1)
    cv2.rectangle(image, (50, 50), (100, 100), 255, -1)
    
    return image


@pytest.fixture
def sample_road_features():
    """Create sample texture features dictionary."""
    return {
        "glcm": {
            "contrast": 0.5,
            "dissimilarity": 0.3,
            "homogeneity": 0.8,
            "energy": 0.4,
            "correlation": 0.7,
            "asm": 0.16,
            "entropy": 2.5
        },
        "lbp": {
            "uniformity": 0.15,
            "entropy": 3.2,
            "mean": 0.04,
            "std": 0.02
        },
        "statistical": {
            "mean": 128.0,
            "std": 45.0,
            "variance": 2025.0,
            "smoothness": 0.95,
            "skewness": 0.1,
            "kurtosis": -0.5,
            "entropy": 7.5
        }
    }


@pytest.fixture
def temp_image_path(tmp_path, sample_color_image):
    """Create a temporary image file for testing."""
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), sample_color_image)
    return image_path


@pytest.fixture
def labels_5_class():
    """Create sample 5-class segmentation labels."""
    labels = np.zeros((256, 256), dtype=np.uint8)
    
    # Divide into 5 vertical strips
    for i in range(5):
        start = i * 51
        end = min((i + 1) * 51, 256)
        labels[:, start:end] = i + 1
    
    return labels
