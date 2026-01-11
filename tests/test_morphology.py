"""
Unit Tests for Morphology Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
import cv2

from src.morphology import (
    get_structuring_element,
    erode,
    dilate,
    opening,
    closing,
    gradient,
    top_hat,
    black_hat,
    fill_holes,
    remove_small_regions,
    connected_components,
    get_region_properties,
    apply_morphology_pipeline,
    refine_boundaries
)


class TestStructuringElement:
    """Tests for structuring element creation."""
    
    def test_rect_element(self):
        """Test rectangular structuring element."""
        kernel = get_structuring_element(shape="rect", size=5)
        
        assert kernel.shape == (5, 5)
        assert kernel.sum() == 25  # All ones
    
    def test_ellipse_element(self):
        """Test elliptical structuring element."""
        kernel = get_structuring_element(shape="ellipse", size=5)
        
        assert kernel.shape == (5, 5)
        assert kernel.sum() < 25  # Not all pixels
    
    def test_cross_element(self):
        """Test cross structuring element."""
        kernel = get_structuring_element(shape="cross", size=5)
        
        assert kernel.shape == (5, 5)
    
    def test_invalid_shape(self):
        """Test invalid shape raises error."""
        with pytest.raises(ValueError):
            get_structuring_element(shape="invalid", size=5)


class TestBasicMorphology:
    """Tests for basic morphological operations."""
    
    def test_erode(self, sample_binary_image):
        """Test erosion operation."""
        eroded = erode(sample_binary_image, kernel_size=3)
        
        assert eroded.shape == sample_binary_image.shape
        # Erosion should reduce white area
        assert eroded.sum() <= sample_binary_image.sum()
    
    def test_dilate(self, sample_binary_image):
        """Test dilation operation."""
        dilated = dilate(sample_binary_image, kernel_size=3)
        
        assert dilated.shape == sample_binary_image.shape
        # Dilation should increase white area
        assert dilated.sum() >= sample_binary_image.sum()
    
    def test_opening(self, sample_binary_image):
        """Test opening operation (erosion + dilation)."""
        opened = opening(sample_binary_image, kernel_size=3)
        
        assert opened.shape == sample_binary_image.shape
    
    def test_closing(self, sample_binary_image):
        """Test closing operation (dilation + erosion)."""
        closed = closing(sample_binary_image, kernel_size=3)
        
        assert closed.shape == sample_binary_image.shape
    
    def test_gradient(self, sample_binary_image):
        """Test morphological gradient."""
        grad = gradient(sample_binary_image, kernel_size=3)
        
        assert grad.shape == sample_binary_image.shape
    
    def test_top_hat(self, sample_grayscale_image):
        """Test top-hat transform."""
        result = top_hat(sample_grayscale_image, kernel_size=9)
        
        assert result.shape == sample_grayscale_image.shape
    
    def test_black_hat(self, sample_grayscale_image):
        """Test black-hat transform."""
        result = black_hat(sample_grayscale_image, kernel_size=9)
        
        assert result.shape == sample_grayscale_image.shape
    
    def test_iterations(self, sample_binary_image):
        """Test multiple iterations."""
        eroded_1 = erode(sample_binary_image, iterations=1)
        eroded_3 = erode(sample_binary_image, iterations=3)
        
        # More iterations = more erosion
        assert eroded_3.sum() <= eroded_1.sum()


class TestHoleFilling:
    """Tests for hole filling operations."""
    
    def test_fill_holes_basic(self):
        """Test basic hole filling."""
        # Create image with hole
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 40, 255, -1)
        cv2.circle(image, (50, 50), 20, 0, -1)  # Create hole
        
        filled = fill_holes(image, max_hole_size=2000)
        
        # Hole should be filled
        assert filled.sum() >= image.sum()
    
    def test_fill_holes_preserves_shape(self, sample_binary_image):
        """Test that fill_holes preserves image shape."""
        filled = fill_holes(sample_binary_image)
        
        assert filled.shape == sample_binary_image.shape
        assert filled.dtype == np.uint8


class TestSmallRegionRemoval:
    """Tests for small region removal."""
    
    def test_remove_small_regions(self):
        """Test removing small regions."""
        # Create image with small and large regions
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 255, -1)  # Large
        cv2.circle(image, (10, 10), 3, 255, -1)   # Small
        
        cleaned = remove_small_regions(image, min_size=100)
        
        # Small region should be removed
        assert cleaned.sum() < image.sum()
        # Large region should remain
        assert cleaned[50, 50] == 255
    
    def test_remove_small_regions_preserves_large(self, sample_binary_image):
        """Test that large regions are preserved."""
        cleaned = remove_small_regions(sample_binary_image, min_size=10)
        
        # Should still have content
        assert cleaned.sum() > 0


class TestConnectedComponents:
    """Tests for connected component analysis."""
    
    def test_connected_components(self, sample_binary_image):
        """Test connected component labeling."""
        labels, n_labels = connected_components(sample_binary_image)
        
        assert labels.shape == sample_binary_image.shape
        assert n_labels >= 1  # At least background
    
    def test_connected_components_count(self):
        """Test correct component count."""
        # Create image with 3 separate regions
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (20, 20), 10, 255, -1)
        cv2.circle(image, (50, 50), 10, 255, -1)
        cv2.circle(image, (80, 80), 10, 255, -1)
        
        labels, n_labels = connected_components(image)
        
        # 3 regions + 1 background = 4 labels
        assert n_labels == 4


class TestRegionProperties:
    """Tests for region property extraction."""
    
    def test_get_region_properties(self, labels_5_class):
        """Test region property extraction."""
        props = get_region_properties(labels_5_class)
        
        assert isinstance(props, list)
        assert len(props) >= 1
        
        for prop in props:
            assert "label" in prop
            assert "area" in prop
            assert "perimeter" in prop
            assert "centroid" in prop


class TestMorphologyPipeline:
    """Tests for morphology pipeline."""
    
    def test_apply_pipeline(self, sample_binary_image):
        """Test applying morphology pipeline."""
        result = apply_morphology_pipeline(
            sample_binary_image,
            operations=["opening", "closing"],
            kernel_size=3
        )
        
        assert result.shape == sample_binary_image.shape
    
    def test_pipeline_default(self, sample_binary_image):
        """Test pipeline with default operations."""
        result = apply_morphology_pipeline(sample_binary_image)
        
        assert result.shape == sample_binary_image.shape
    
    def test_pipeline_invalid_operation(self, sample_binary_image):
        """Test pipeline with invalid operation."""
        with pytest.raises(ValueError):
            apply_morphology_pipeline(
                sample_binary_image,
                operations=["invalid_op"]
            )


class TestBoundaryRefinement:
    """Tests for boundary refinement."""
    
    def test_refine_boundaries(self, labels_5_class):
        """Test segment boundary refinement."""
        refined = refine_boundaries(labels_5_class, iterations=1)
        
        assert refined.shape == labels_5_class.shape
    
    def test_refine_boundaries_multiple_iterations(self, labels_5_class):
        """Test multiple refinement iterations."""
        refined = refine_boundaries(labels_5_class, iterations=3)
        
        assert refined.shape == labels_5_class.shape
