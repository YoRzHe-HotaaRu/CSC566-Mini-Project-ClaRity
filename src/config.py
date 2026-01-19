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
        "color": (43, 90, 139),  # Brown (BGR)
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
        "color": (169, 169, 169),  # Light Gray (BGR)
        "hex_color": "#A9A9A9"
    },
    4: {
        "name": "Binder Course",
        "full_name": "Binder Course",
        "material": "Premix asphalt",
        "color": (70, 70, 70),  # Dark Gray (BGR) - asphalt
        "hex_color": "#464646"
    },
    5: {
        "name": "Surface Course",
        "full_name": "Wearing Course",
        "material": "Premix asphalt (smooth)",
        "color": (50, 50, 50),  # Very Dark Gray (BGR) - asphalt
        "hex_color": "#323232"
    }
}

# Layer color map for visualization (RGB for matplotlib)
LAYER_COLORS_RGB = {
    1: (139/255, 90/255, 43/255),   # Brown - Subgrade
    2: (128/255, 128/255, 128/255), # Gray - Subbase
    3: (169/255, 169/255, 169/255), # Light Gray - Base
    4: (70/255, 70/255, 70/255),    # Dark Gray - Binder (asphalt)
    5: (50/255, 50/255, 50/255)     # Very Dark Gray - Surface (asphalt)
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
# YOLOv11 INSTANCE SEGMENTATION PARAMETERS
# =============================================================================

YOLO_CONFIG = {
    "model_path": "YOLOv11/YOLOv11_11_weight.pt",
    "device": "cuda",  # "cuda" or "cpu"
    "confidence": 0.5,
    "iou_threshold": 0.45,
    "target_fps": 30,
    # Class name to layer number mapping
    "class_mapping": {
        "layer_subgrade": 1,
        "layer_subbase": 2,
        "layer_basecourse": 3,
        "layer_bindercourse": 4,
        "layer_surfacecourse": 5
    }
}

# =============================================================================
# VLM ANALYSIS PARAMETERS
# =============================================================================

VLM_CONFIG = {
    "temperature": 0.3,
    "max_tokens": 1000,
    "timeout": 30
}

# Road analysis prompt for VLM (IMPROVED - Universal prompt)
ROAD_ANALYSIS_PROMPT = """
Analyze this road construction layer image and identify which layer is shown.

THE 5 ROAD LAYERS:

Layer 1 - SUBGRADE
- What: Natural soil/earth layer
- Color: Brown, tan, earthy tones
- Texture: Irregular, rough, soil-like
- Key features: Organic matter, soil clumps, plant material possible

Layer 2 - SUBBASE COURSE  
- What: Coarse crushed aggregate base
- Color: Gray to light gray
- Texture: Very rough, loose stones visible
- Key features: Large stones (2-4cm), high texture variation, voids

Layer 3 - BASE COURSE
- What: Fine crushed aggregate layer
- Color: Uniform gray
- Texture: Moderately rough but uniform
- Key features: Smaller stones (0.5-2cm), compacted surface

Layer 4 - BINDER COURSE
- What: Coarse asphalt mix
- Color: Dark gray/black with visible stones
- Texture: Asphalt with aggregate texture
- Key features: Black binder material, aggregate visible on surface

Layer 5 - SURFACE/WEARING COURSE
- What: Finished asphalt surface
- Color: Very dark black or dark gray  
- Texture: Smooth to slightly textured
- Key features: No visible aggregate, uniform appearance, polished look

ANALYSIS CHECKLIST:
1. What is the DOMINANT COLOR? (brown → soil/gray → stones/black → asphalt)
2. What is the TEXTURE? (soil-like/rough stones/medium stones/asphalt/smooth)
3. Can you see INDIVIDUAL STONES? (yes → subbase/base, no → asphalt/soil)
4. Is it ASPHALT or SOIL/AGGREGATE? (black & binder → asphalt, brown → soil, gray stones → aggregate)

Provide answer in this EXACT format:
LAYER: [1-5]
NAME: [exact name from list above]
CONFIDENCE: [0-100%]
REASONING: [2-3 sentences explaining your choice based on color, texture, and material observed]
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
