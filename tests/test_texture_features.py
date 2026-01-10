"""
Unit Tests for Texture Features Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np

from src.texture_features import (
    compute_glcm,
    extract_glcm_features,
    compute_lbp,
    extract_lbp_histogram,
    extract_lbp_features,
    create_gabor_kernels,
    apply_gabor_filter,
    extract_gabor_features,
    extract_statistical_features,
    extract_all_texture_features,
    features_to_vector
)


class TestGLCMFeatures:
    """Tests for GLCM feature extraction."""
    
    def test_compute_glcm(self, sample_grayscale_image):
        """Test GLCM computation."""
        glcm = compute_glcm(sample_grayscale_image, distances=[1], angles=[0])
        
        assert glcm is not None
        assert glcm.ndim == 4
    
    def test_extract_glcm_features(self, sample_grayscale_image):
        """Test GLCM feature extraction."""
        features = extract_glcm_features(sample_grayscale_image)
        
        assert "contrast" in features
        assert "energy" in features
        assert "homogeneity" in features
        assert "correlation" in features
        assert "entropy" in features
        
        # All features should be finite numbers
        for name, value in features.items():
            assert np.isfinite(value), f"{name} is not finite"
    
    def test_glcm_contrast_values(self, sample_textured_image):
        """Test GLCM contrast distinguishes textures."""
        # Smooth region
        smooth_region = sample_textured_image[0:128, 0:128]
        smooth_features = extract_glcm_features(smooth_region)
        
        # Rough region
        rough_region = sample_textured_image[128:256, 0:128]
        rough_features = extract_glcm_features(rough_region)
        
        # Rough region should have higher contrast
        assert rough_features["contrast"] > smooth_features["contrast"]
    
    def test_glcm_with_different_parameters(self, sample_grayscale_image):
        """Test GLCM with various parameters."""
        features1 = extract_glcm_features(sample_grayscale_image, distances=[1])
        features2 = extract_glcm_features(sample_grayscale_image, distances=[1, 2, 3])
        
        # Both should produce valid features
        assert features1["contrast"] >= 0
        assert features2["contrast"] >= 0


class TestLBPFeatures:
    """Tests for LBP feature extraction."""
    
    def test_compute_lbp(self, sample_grayscale_image):
        """Test LBP computation."""
        lbp = compute_lbp(sample_grayscale_image, radius=1, n_points=8)
        
        assert lbp.shape == sample_grayscale_image.shape
    
    def test_lbp_histogram(self, sample_grayscale_image):
        """Test LBP histogram extraction."""
        hist = extract_lbp_histogram(sample_grayscale_image, radius=3, n_points=24)
        
        assert hist.ndim == 1
        assert np.isclose(hist.sum(), 1, atol=0.01)  # Normalized
    
    def test_lbp_features(self, sample_grayscale_image):
        """Test LBP feature extraction."""
        features = extract_lbp_features(sample_grayscale_image)
        
        assert "histogram" in features
        assert "uniformity" in features
        assert "entropy" in features
        
        assert features["uniformity"] >= 0
        assert features["entropy"] >= 0
    
    def test_lbp_different_radii(self, sample_grayscale_image):
        """Test LBP with different radii."""
        lbp1 = compute_lbp(sample_grayscale_image, radius=1, n_points=8)
        lbp2 = compute_lbp(sample_grayscale_image, radius=3, n_points=24)
        
        assert lbp1.shape == lbp2.shape


class TestGaborFeatures:
    """Tests for Gabor filter features."""
    
    def test_create_gabor_kernels(self):
        """Test Gabor kernel creation."""
        kernels = create_gabor_kernels(
            frequencies=[0.1, 0.2],
            orientations=[0, 45]
        )
        
        assert len(kernels) == 4  # 2 frequencies x 2 orientations
    
    def test_apply_gabor_filter(self, sample_grayscale_image):
        """Test Gabor filter application."""
        kernels = create_gabor_kernels(frequencies=[0.1], orientations=[0])
        magnitude, phase = apply_gabor_filter(sample_grayscale_image, kernels[0])
        
        assert magnitude.shape == sample_grayscale_image.shape
        assert phase.shape == sample_grayscale_image.shape
    
    def test_gabor_features(self, sample_grayscale_image):
        """Test Gabor feature extraction."""
        features = extract_gabor_features(sample_grayscale_image)
        
        assert "means" in features
        assert "stds" in features
        assert "energies" in features
        assert "mean_of_means" in features
        
        assert np.isfinite(features["mean_of_means"])


class TestStatisticalFeatures:
    """Tests for statistical texture features."""
    
    def test_statistical_features(self, sample_grayscale_image):
        """Test statistical feature extraction."""
        features = extract_statistical_features(sample_grayscale_image)
        
        assert "mean" in features
        assert "std" in features
        assert "variance" in features
        assert "smoothness" in features
        assert "skewness" in features
        assert "kurtosis" in features
        assert "entropy" in features
    
    def test_smoothness_range(self, sample_grayscale_image):
        """Test smoothness is in valid range."""
        features = extract_statistical_features(sample_grayscale_image)
        
        # Smoothness should be between 0 and 1
        assert 0 <= features["smoothness"] <= 1
    
    def test_entropy_positive(self, sample_grayscale_image):
        """Test entropy is positive."""
        features = extract_statistical_features(sample_grayscale_image)
        
        assert features["entropy"] >= 0


class TestCombinedFeatures:
    """Tests for combined feature extraction."""
    
    def test_extract_all_features(self, sample_grayscale_image):
        """Test extracting all texture features."""
        features = extract_all_texture_features(
            sample_grayscale_image,
            use_glcm=True,
            use_lbp=True,
            use_gabor=True,
            use_statistical=True
        )
        
        assert "glcm" in features
        assert "lbp" in features
        assert "gabor" in features
        assert "statistical" in features
    
    def test_selective_features(self, sample_grayscale_image):
        """Test extracting selective features."""
        features = extract_all_texture_features(
            sample_grayscale_image,
            use_glcm=True,
            use_lbp=False,
            use_gabor=False,
            use_statistical=True
        )
        
        assert "glcm" in features
        assert "statistical" in features
        assert "lbp" not in features
        assert "gabor" not in features
    
    def test_features_to_vector(self, sample_road_features):
        """Test converting features to vector."""
        vector = features_to_vector(sample_road_features)
        
        assert isinstance(vector, np.ndarray)
        assert vector.ndim == 1
        assert len(vector) > 0
    
    def test_feature_vector_length(self, sample_grayscale_image):
        """Test feature vector has consistent length."""
        features = extract_all_texture_features(
            sample_grayscale_image,
            use_glcm=True,
            use_lbp=True,
            use_gabor=False,
            use_statistical=True
        )
        
        vector = features_to_vector(features)
        
        # GLCM: 7, LBP: 4, Statistical: 7 = 18 features
        assert len(vector) == 18
