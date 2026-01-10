"""
Unit Tests for Preprocessing Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
import cv2

from src.preprocessing import (
    apply_noise_filter,
    enhance_contrast,
    convert_color_space,
    resize_image,
    normalize_image,
    preprocess_image,
    get_image_info
)


class TestNoiseFilter:
    """Tests for noise filtering functions."""
    
    def test_median_filter(self, sample_grayscale_image):
        """Test median filter reduces noise."""
        # Add salt and pepper noise
        noisy = sample_grayscale_image.copy()
        mask = np.random.random(noisy.shape) < 0.1
        noisy[mask] = 255
        
        filtered = apply_noise_filter(noisy, method="median", kernel_size=3)
        
        assert filtered.shape == noisy.shape
        assert filtered.dtype == noisy.dtype
    
    def test_gaussian_filter(self, sample_grayscale_image):
        """Test Gaussian filter smooths image."""
        filtered = apply_noise_filter(sample_grayscale_image, method="gaussian", kernel_size=5)
        
        assert filtered.shape == sample_grayscale_image.shape
        # Gaussian should reduce variance
        assert filtered.std() <= sample_grayscale_image.std()
    
    def test_bilateral_filter(self, sample_color_image):
        """Test bilateral filter preserves edges."""
        filtered = apply_noise_filter(sample_color_image, method="bilateral", kernel_size=9)
        
        assert filtered.shape == sample_color_image.shape
    
    def test_invalid_filter_method(self, sample_grayscale_image):
        """Test invalid filter method raises error."""
        with pytest.raises(ValueError):
            apply_noise_filter(sample_grayscale_image, method="invalid")
    
    def test_even_kernel_size(self, sample_grayscale_image):
        """Test even kernel size is corrected to odd."""
        # Should not raise error, kernel size corrected internally
        filtered = apply_noise_filter(sample_grayscale_image, method="median", kernel_size=4)
        assert filtered is not None


class TestContrastEnhancement:
    """Tests for contrast enhancement functions."""
    
    def test_histogram_equalization(self, sample_grayscale_image):
        """Test histogram equalization."""
        enhanced = enhance_contrast(sample_grayscale_image, method="histogram_eq")
        
        assert enhanced.shape == sample_grayscale_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_clahe(self, sample_grayscale_image):
        """Test CLAHE enhancement."""
        enhanced = enhance_contrast(
            sample_grayscale_image,
            method="clahe",
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )
        
        assert enhanced.shape == sample_grayscale_image.shape
    
    def test_clahe_color(self, sample_color_image):
        """Test CLAHE on color image (enhances L channel)."""
        enhanced = enhance_contrast(sample_color_image, method="clahe")
        
        assert enhanced.shape == sample_color_image.shape
    
    def test_gamma_correction(self, sample_grayscale_image):
        """Test gamma correction."""
        # Gamma < 1 brightens
        brightened = enhance_contrast(sample_grayscale_image, method="gamma", gamma=0.5)
        assert brightened.mean() >= sample_grayscale_image.mean()
        
        # Gamma > 1 darkens
        darkened = enhance_contrast(sample_grayscale_image, method="gamma", gamma=2.0)
        assert darkened.mean() <= sample_grayscale_image.mean()
    
    def test_invalid_enhancement_method(self, sample_grayscale_image):
        """Test invalid enhancement method raises error."""
        with pytest.raises(ValueError):
            enhance_contrast(sample_grayscale_image, method="invalid")


class TestColorSpaceConversion:
    """Tests for color space conversion."""
    
    def test_bgr_to_gray(self, sample_color_image):
        """Test BGR to grayscale conversion."""
        gray = convert_color_space(sample_color_image, target_space="gray")
        
        assert len(gray.shape) == 2
        assert gray.shape == sample_color_image.shape[:2]
    
    def test_bgr_to_hsv(self, sample_color_image):
        """Test BGR to HSV conversion."""
        hsv = convert_color_space(sample_color_image, target_space="hsv")
        
        assert hsv.shape == sample_color_image.shape
    
    def test_bgr_to_lab(self, sample_color_image):
        """Test BGR to LAB conversion."""
        lab = convert_color_space(sample_color_image, target_space="lab")
        
        assert lab.shape == sample_color_image.shape
    
    def test_bgr_to_rgb(self, sample_color_image):
        """Test BGR to RGB conversion."""
        rgb = convert_color_space(sample_color_image, target_space="rgb")
        
        assert rgb.shape == sample_color_image.shape
    
    def test_invalid_color_space(self, sample_color_image):
        """Test invalid color space raises error."""
        with pytest.raises(ValueError):
            convert_color_space(sample_color_image, target_space="invalid")


class TestResizeImage:
    """Tests for image resizing."""
    
    def test_resize_to_target_size(self, sample_grayscale_image):
        """Test resize to specific dimensions."""
        resized = resize_image(sample_grayscale_image, target_size=(128, 128))
        
        assert resized.shape == (128, 128)
    
    def test_resize_by_scale(self, sample_color_image):
        """Test resize by scale factor."""
        resized = resize_image(sample_color_image, scale=0.5)
        
        assert resized.shape[0] == sample_color_image.shape[0] // 2
        assert resized.shape[1] == sample_color_image.shape[1] // 2
    
    def test_no_resize(self, sample_grayscale_image):
        """Test no resize when no parameters given."""
        result = resize_image(sample_grayscale_image)
        
        assert result.shape == sample_grayscale_image.shape


class TestNormalization:
    """Tests for image normalization."""
    
    def test_minmax_normalization(self, sample_grayscale_image):
        """Test min-max normalization to 0-1 range."""
        normalized = normalize_image(sample_grayscale_image, method="minmax")
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_zscore_normalization(self, sample_grayscale_image):
        """Test z-score normalization."""
        normalized = normalize_image(sample_grayscale_image, method="zscore")
        
        assert normalized.dtype == np.float32
        assert np.abs(normalized.mean()) < 0.001  # Mean should be ~0
    
    def test_invalid_normalization_method(self, sample_grayscale_image):
        """Test invalid normalization method raises error."""
        with pytest.raises(ValueError):
            normalize_image(sample_grayscale_image, method="invalid")


class TestPreprocessPipeline:
    """Tests for complete preprocessing pipeline."""
    
    def test_full_pipeline(self, sample_color_image):
        """Test full preprocessing pipeline."""
        result = preprocess_image(
            sample_color_image,
            denoise="median",
            denoise_kernel=3,
            enhance="clahe",
            color_space="gray"
        )
        
        assert len(result.shape) == 2  # Should be grayscale
    
    def test_pipeline_no_denoise(self, sample_color_image):
        """Test pipeline without denoising."""
        result = preprocess_image(
            sample_color_image,
            denoise=None,
            enhance="clahe"
        )
        
        assert result.shape == sample_color_image.shape
    
    def test_pipeline_with_resize(self, sample_color_image):
        """Test pipeline with resize."""
        result = preprocess_image(
            sample_color_image,
            resize_to=(128, 128),
            denoise="gaussian"
        )
        
        assert result.shape[:2] == (128, 128)
    
    def test_pipeline_with_normalization(self, sample_grayscale_image):
        """Test pipeline with normalization."""
        result = preprocess_image(
            sample_grayscale_image,
            normalize=True
        )
        
        assert result.dtype == np.float32


class TestGetImageInfo:
    """Tests for image info function."""
    
    def test_grayscale_info(self, sample_grayscale_image):
        """Test info for grayscale image."""
        info = get_image_info(sample_grayscale_image)
        
        assert "shape" in info
        assert info["channels"] == 1
        assert info["height"] == 256
        assert info["width"] == 256
    
    def test_color_info(self, sample_color_image):
        """Test info for color image."""
        info = get_image_info(sample_color_image)
        
        assert info["channels"] == 3
        assert "mean" in info
        assert "std" in info
