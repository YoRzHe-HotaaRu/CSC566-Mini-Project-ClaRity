"""
Configuration and Constants for Road Surface Layer Analyzer
CSC566 Image Processing Mini Project
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SUBGRADE_DIR = DATA_DIR / "subgrade"
SUBBASE_DIR = DATA_DIR / "subbase"
BASE_COURSE_DIR = DATA_DIR / "base_course"
BINDER_COURSE_DIR = DATA_DIR / "binder_course"
SURFACE_COURSE_DIR = DATA_DIR / "surface_course"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# ROAD LAYER DEFINITIONS
# =============================================================================

ROAD_LAYERS = {
    1: {
        "name": "Subgrade",
        "full_name": "Compacted Subgrade",
        "material": "In-site soil/backfill",
        "color": (139, 90, 43),  # Brown (BGR)
        "hex_color": "#8B5A2B"
    },
    2: {
        "name": "Subbase Course",
        "full_name": "Subbase Course",
        "material": "Crushed aggregate (coarse)",
        "color": (128, 128, 128),  # Gray (BGR)
        "hex_color": "#808080"
    },
    3: {
        "name": "Base Course",
        "full_name": "Base Course",
        "material": "Crushed aggregate (finer)",
        "color": (169, 169, 169),  # Dark Gray (BGR)
        "hex_color": "#A9A9A9"
    },
    4: {
        "name": "Binder Course",
        "full_name": "Binder Course",
        "material": "Premix asphalt",
        "color": (50, 50, 50),  # Dark (BGR)
        "hex_color": "#323232"
    },
    5: {
        "name": "Surface Course",
        "full_name": "Wearing Course",
        "material": "Premix asphalt (smooth)",
        "color": (30, 30, 30),  # Very Dark (BGR)
        "hex_color": "#1E1E1E"
    }
}

# Layer color map for visualization (RGB for matplotlib)
LAYER_COLORS_RGB = {
    1: (139/255, 90/255, 43/255),   # Brown - Subgrade
    2: (128/255, 128/255, 128/255), # Gray - Subbase
    3: (169/255, 169/255, 169/255), # Light Gray - Base
    4: (80/255, 80/255, 80/255),    # Dark Gray - Binder
    5: (40/255, 40/255, 40/255)     # Very Dark - Surface
}

# =============================================================================
# ZENMUX API CONFIGURATION (VLM - GLM-4.6V)
# =============================================================================

ZENMUX_API_KEY = os.getenv("ZENMUX_API_KEY", "")
ZENMUX_BASE_URL = os.getenv("ZENMUX_BASE_URL", "https://zenmux.ai/api/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "z-ai/glm-4.6v")

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================

PREPROCESSING_CONFIG = {
    "noise_filters": ["gaussian", "median", "bilateral"],
    "default_noise_filter": "median",
    "default_kernel_size": 3,
    "contrast_methods": ["histogram_eq", "clahe", "gamma"],
    "default_contrast_method": "clahe",
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "gamma_default": 1.0
}

# =============================================================================
# TEXTURE FEATURE PARAMETERS
# =============================================================================

GLCM_CONFIG = {
    "distances": [1, 2, 3],
    "angles": [0, 45, 90, 135],  # Degrees
    "levels": 256,
    "symmetric": True,
    "normed": True
}

LBP_CONFIG = {
    "radius": 3,
    "n_points": 24,
    "method": "uniform"
}

GABOR_CONFIG = {
    "frequencies": [0.1, 0.2, 0.3, 0.4],
    "orientations": [0, 45, 90, 135],  # Degrees
    "sigma": 1.0
}

# =============================================================================
# SEGMENTATION PARAMETERS
# =============================================================================

SEGMENTATION_CONFIG = {
    "methods": ["kmeans", "watershed", "slic", "felzenszwalb"],
    "default_method": "kmeans",
    "kmeans_n_clusters": 5,  # 5 road layers
    "kmeans_max_iter": 300,
    "kmeans_n_init": 10,
    "slic_n_segments": 200,
    "slic_compactness": 10,
    "watershed_markers": None
}

# =============================================================================
# MORPHOLOGY PARAMETERS
# =============================================================================

MORPHOLOGY_CONFIG = {
    "kernel_size": 5,
    "kernel_shape": "ellipse",  # "rect", "ellipse", "cross"
    "iterations": 1,
    "min_area": 100,  # Minimum region area to keep
    "fill_holes": True
}

# =============================================================================
# DEEP LEARNING PARAMETERS
# =============================================================================

DEEP_LEARNING_CONFIG = {
    "model_name": "DeepLabV3Plus",
    "encoder_name": "resnet101",
    "encoder_weights": "imagenet",
    "in_channels": 3,
    "classes": 5,  # 5 road layers
    "activation": None,
    "device": "cuda",  # "cuda" or "cpu"
    "batch_size": 1,
    "image_size": (512, 512)
}

# =============================================================================
# VLM ANALYSIS PARAMETERS
# =============================================================================

VLM_CONFIG = {
    "temperature": 0.3,
    "max_tokens": 1000,
    "timeout": 30
}

# Road analysis prompt for VLM
ROAD_ANALYSIS_PROMPT = """
Analyze this aerial satellite image of a road construction site.
Identify which of the 5 road construction layers is shown:

1. Subgrade (in-site soil/backfill) - Earth/soil, brown tones, irregular texture
2. Subbase Course (crushed aggregate) - Coarse stones, gray, rough texture
3. Base Course (crushed aggregate) - Finer aggregate, more uniform gray
4. Binder Course (premix asphalt) - Dark surface with visible aggregate
5. Surface Course (premix asphalt) - Smooth, uniform dark/black surface

Provide your analysis in the following format:
- Layer Number: [1-5]
- Layer Name: [name]
- Confidence: [0-100%]
- Material Observed: [description]
- Texture Characteristics: [description]
- Additional Notes: [any relevant observations]
"""

# =============================================================================
# GUI CONFIGURATION
# =============================================================================

GUI_CONFIG = {
    "window_title": "Road Surface Layer Analyzer",
    "window_size": (1400, 900),
    "min_size": (1000, 700),
    "theme": "dark"
}

# =============================================================================
# TESTING CONFIGURATION
# =============================================================================

TEST_CONFIG = {
    "coverage_threshold": 80,
    "benchmark_iterations": 100
}
