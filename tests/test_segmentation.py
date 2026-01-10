"""
Unit Tests for Segmentation Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np

from src.segmentation import (
    kmeans_segment,
    kmeans_with_texture,
    slic_segment,
    superpixel_to_region,
    watershed_segment,
    otsu_threshold,
    adaptive_threshold,
    multi_level_threshold,
    detect_edges,
    create_segmentation_overlay,
    get_segment_properties
)


class TestKMeansSegmentation:
    """Tests for K-Means segmentation."""
    
    def test_kmeans_basic(self, sample_color_image):
        """Test basic K-Means segmentation."""
        labels, centers = kmeans_segment(sample_color_image, n_clusters=5)
        
        assert labels.shape == sample_color_image.shape[:2]
        assert len(np.unique(labels)) <= 5
    
    def test_kmeans_grayscale(self, sample_grayscale_image):
        """Test K-Means on grayscale image."""
        labels, centers = kmeans_segment(sample_grayscale_image, n_clusters=3)
        
        assert labels.shape == sample_grayscale_image.shape
    
    def test_kmeans_with_spatial(self, sample_color_image):
        """Test K-Means with spatial coordinates."""
        labels, centers = kmeans_segment(
            sample_color_image,
            n_clusters=5,
            use_spatial=True,
            spatial_weight=0.5
        )
        
        assert labels.shape == sample_color_image.shape[:2]
    
    def test_kmeans_cluster_count(self, sample_textured_image):
        """Test K-Means produces correct number of clusters."""
        n = 4
        labels, _ = kmeans_segment(sample_textured_image, n_clusters=n)
        
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= n


class TestSuperpixelSegmentation:
    """Tests for SLIC superpixel segmentation."""
    
    def test_slic_basic(self, sample_color_image):
        """Test basic SLIC segmentation."""
        segments = slic_segment(sample_color_image, n_segments=100)
        
        assert segments.shape == sample_color_image.shape[:2]
        assert len(np.unique(segments)) > 1
    
    def test_slic_segment_count(self, sample_color_image):
        """Test SLIC produces approximately correct segment count."""
        n_segments = 50
        segments = slic_segment(sample_color_image, n_segments=n_segments)
        
        # SLIC produces approximately n_segments
        actual_segments = len(np.unique(segments))
        assert actual_segments > n_segments * 0.5
        assert actual_segments < n_segments * 2
    
    def test_superpixel_to_region(self, sample_color_image):
        """Test converting superpixels to region-averaged image."""
        segments = slic_segment(sample_color_image, n_segments=50)
        region_image = superpixel_to_region(sample_color_image, segments)
        
        assert region_image.shape == sample_color_image.shape


class TestWatershedSegmentation:
    """Tests for Watershed segmentation."""
    
    def test_watershed_basic(self, sample_grayscale_image):
        """Test basic Watershed segmentation."""
        labels = watershed_segment(sample_grayscale_image)
        
        assert labels.shape == sample_grayscale_image.shape
    
    def test_watershed_color(self, sample_color_image):
        """Test Watershed on color image."""
        labels = watershed_segment(sample_color_image)
        
        assert labels.shape == sample_color_image.shape[:2]


class TestThresholding:
    """Tests for thresholding methods."""
    
    def test_otsu_threshold(self, sample_grayscale_image):
        """Test Otsu's automatic thresholding."""
        binary, threshold = otsu_threshold(sample_grayscale_image)
        
        assert binary.shape == sample_grayscale_image.shape
        assert set(np.unique(binary)).issubset({0, 255})
        assert 0 < threshold < 255
    
    def test_adaptive_threshold_gaussian(self, sample_grayscale_image):
        """Test Gaussian adaptive thresholding."""
        binary = adaptive_threshold(sample_grayscale_image, method="gaussian")
        
        assert binary.shape == sample_grayscale_image.shape
        assert set(np.unique(binary)).issubset({0, 255})
    
    def test_adaptive_threshold_mean(self, sample_grayscale_image):
        """Test mean adaptive thresholding."""
        binary = adaptive_threshold(sample_grayscale_image, method="mean")
        
        assert binary.shape == sample_grayscale_image.shape
    
    def test_multi_level_threshold(self, sample_grayscale_image):
        """Test multi-level thresholding."""
        labels = multi_level_threshold(sample_grayscale_image, n_levels=5)
        
        assert labels.shape == sample_grayscale_image.shape
        assert len(np.unique(labels)) <= 5


class TestEdgeDetection:
    """Tests for edge detection."""
    
    def test_canny_edges(self, sample_grayscale_image):
        """Test Canny edge detection."""
        edges = detect_edges(sample_grayscale_image, method="canny")
        
        assert edges.shape == sample_grayscale_image.shape
        assert set(np.unique(edges)).issubset({0, 255})
    
    def test_sobel_edges(self, sample_grayscale_image):
        """Test Sobel edge detection."""
        edges = detect_edges(sample_grayscale_image, method="sobel")
        
        assert edges.shape == sample_grayscale_image.shape
    
    def test_prewitt_edges(self, sample_grayscale_image):
        """Test Prewitt edge detection."""
        edges = detect_edges(sample_grayscale_image, method="prewitt")
        
        assert edges.shape == sample_grayscale_image.shape
    
    def test_laplacian_edges(self, sample_grayscale_image):
        """Test Laplacian edge detection."""
        edges = detect_edges(sample_grayscale_image, method="laplacian")
        
        assert edges.shape == sample_grayscale_image.shape
    
    def test_invalid_edge_method(self, sample_grayscale_image):
        """Test invalid edge detection method raises error."""
        with pytest.raises(ValueError):
            detect_edges(sample_grayscale_image, method="invalid")


class TestVisualization:
    """Tests for segmentation visualization."""
    
    def test_create_overlay(self, sample_color_image, labels_5_class):
        """Test creating segmentation overlay."""
        overlay = create_segmentation_overlay(sample_color_image, labels_5_class)
        
        assert overlay.shape == sample_color_image.shape
    
    def test_overlay_grayscale(self, sample_grayscale_image, labels_5_class):
        """Test overlay on grayscale image."""
        overlay = create_segmentation_overlay(sample_grayscale_image, labels_5_class)
        
        # Should be converted to color for overlay
        assert len(overlay.shape) == 3


class TestSegmentProperties:
    """Tests for segment property extraction."""
    
    def test_get_properties(self, labels_5_class):
        """Test getting segment properties."""
        props = get_segment_properties(labels_5_class)
        
        assert isinstance(props, list)
        assert len(props) > 0
        
        for prop in props:
            assert "label" in prop
            assert "area" in prop
            assert "centroid" in prop
