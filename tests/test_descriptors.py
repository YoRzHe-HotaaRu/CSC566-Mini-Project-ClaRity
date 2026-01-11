"""
Unit Tests for Descriptors Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
import cv2

from src.descriptors import (
    extract_boundary,
    compute_chain_code,
    normalize_chain_code,
    chain_code_histogram,
    compute_fourier_descriptors,
    reconstruct_from_fourier,
    compute_region_descriptors,
    compute_compactness,
    compute_circularity,
    compute_aspect_ratio,
    compute_euler_number,
    extract_all_descriptors,
    compare_shapes
)


class TestBoundaryExtraction:
    """Tests for boundary extraction."""
    
    def test_extract_boundary_contour(self, sample_binary_image):
        """Test boundary extraction using contour method."""
        boundaries = extract_boundary(sample_binary_image, method="contour")
        
        assert isinstance(boundaries, list)
        assert len(boundaries) >= 1
        
        # Each boundary should be array of points
        for boundary in boundaries:
            assert boundary.ndim == 2
            assert boundary.shape[1] == 2  # x, y coordinates
    
    def test_extract_boundary_gradient(self, sample_binary_image):
        """Test boundary extraction using gradient method."""
        boundaries = extract_boundary(sample_binary_image, method="gradient")
        
        assert isinstance(boundaries, list)
        assert len(boundaries) >= 1


class TestChainCode:
    """Tests for chain code computation."""
    
    @pytest.fixture
    def simple_boundary(self):
        """Create a simple rectangular boundary."""
        # Simple 4-point square
        return np.array([
            [0, 0], [1, 0], [2, 0], [2, 1], [2, 2],
            [1, 2], [0, 2], [0, 1], [0, 0]
        ])
    
    def test_compute_chain_code_8(self, simple_boundary):
        """Test 8-connectivity chain code."""
        chain = compute_chain_code(simple_boundary, connectivity=8)
        
        assert isinstance(chain, np.ndarray)
        assert len(chain) == len(simple_boundary) - 1
        
        # All values should be 0-7
        assert all(0 <= c < 8 for c in chain)
    
    def test_compute_chain_code_4(self, simple_boundary):
        """Test 4-connectivity chain code."""
        chain = compute_chain_code(simple_boundary, connectivity=4)
        
        assert isinstance(chain, np.ndarray)
    
    def test_chain_code_empty_boundary(self):
        """Test chain code with too few points."""
        boundary = np.array([[0, 0]])
        chain = compute_chain_code(boundary)
        
        assert len(chain) == 0
    
    def test_normalize_chain_code(self):
        """Test chain code normalization."""
        chain = np.array([0, 2, 4, 6, 0, 2])
        normalized = normalize_chain_code(chain)
        
        assert isinstance(normalized, np.ndarray)
    
    def test_chain_code_histogram(self):
        """Test chain code histogram."""
        chain = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0])
        hist = chain_code_histogram(chain, connectivity=8)
        
        assert len(hist) == 8
        assert np.isclose(hist.sum(), 1, atol=0.01)  # Normalized


class TestFourierDescriptors:
    """Tests for Fourier descriptors."""
    
    @pytest.fixture
    def circle_boundary(self):
        """Create a circular boundary."""
        theta = np.linspace(0, 2*np.pi, 100)
        x = 50 + 30 * np.cos(theta)
        y = 50 + 30 * np.sin(theta)
        return np.column_stack([x, y])
    
    def test_compute_fourier_descriptors(self, circle_boundary):
        """Test Fourier descriptor computation."""
        descriptors = compute_fourier_descriptors(circle_boundary)
        
        assert isinstance(descriptors, np.ndarray)
        assert len(descriptors) == len(circle_boundary)
    
    def test_fourier_descriptors_limited(self, circle_boundary):
        """Test limited number of Fourier descriptors."""
        n = 10
        descriptors = compute_fourier_descriptors(circle_boundary, n_descriptors=n)
        
        assert len(descriptors) == n
    
    def test_fourier_descriptors_empty(self):
        """Test Fourier descriptors with too few points."""
        boundary = np.array([[0, 0], [1, 1]])
        descriptors = compute_fourier_descriptors(boundary)
        
        assert len(descriptors) == 0
    
    def test_reconstruct_from_fourier(self, circle_boundary):
        """Test boundary reconstruction from Fourier descriptors."""
        descriptors = compute_fourier_descriptors(circle_boundary)
        reconstructed = reconstruct_from_fourier(descriptors, n_points=50, n_harmonics=20)
        
        assert reconstructed.shape == (50, 2)


class TestRegionDescriptors:
    """Tests for region descriptors."""
    
    def test_compute_region_descriptors(self, labels_5_class):
        """Test region descriptor computation."""
        props = compute_region_descriptors(labels_5_class)
        
        assert isinstance(props, list)
        assert len(props) >= 1
        
        for prop in props:
            assert "label" in prop
            assert "area" in prop
            assert "perimeter" in prop
            assert "eccentricity" in prop
            assert "solidity" in prop
    
    def test_region_descriptors_with_intensity(self, labels_5_class, sample_grayscale_image):
        """Test region descriptors with intensity image."""
        # Resize intensity image to match labels if needed
        intensity = cv2.resize(sample_grayscale_image, labels_5_class.shape[::-1])
        
        props = compute_region_descriptors(labels_5_class, intensity_image=intensity)
        
        for prop in props:
            assert "mean_intensity" in prop


class TestShapeMetrics:
    """Tests for shape metrics."""
    
    def test_compute_compactness(self):
        """Test compactness computation."""
        # Circle has compactness close to 1
        area = np.pi * 50**2
        perimeter = 2 * np.pi * 50
        
        compactness = compute_compactness(area, perimeter)
        
        assert compactness > 0
        assert np.isclose(compactness, 1.0, atol=0.1)
    
    def test_compute_circularity(self):
        """Test circularity computation."""
        # Circle has circularity of 1
        area = np.pi * 50**2
        perimeter = 2 * np.pi * 50
        
        circularity = compute_circularity(area, perimeter)
        
        assert 0 <= circularity <= 1
        assert np.isclose(circularity, 1.0, atol=0.1)
    
    def test_compute_aspect_ratio(self):
        """Test aspect ratio computation."""
        bbox = (0, 0, 100, 200)  # min_row, min_col, max_row, max_col
        
        aspect = compute_aspect_ratio(bbox)
        
        assert aspect == 2.0  # width / height = 200 / 100
    
    def test_compute_euler_number(self, sample_binary_image):
        """Test Euler number computation."""
        euler = compute_euler_number(sample_binary_image)
        
        assert isinstance(euler, (int, np.integer))


class TestCombinedDescriptors:
    """Tests for combined descriptor extraction."""
    
    def test_extract_all_descriptors(self, sample_binary_image):
        """Test extracting all descriptors."""
        descriptors = extract_all_descriptors(sample_binary_image)
        
        assert "n_regions" in descriptors
        assert "euler_number" in descriptors
        assert "regions" in descriptors
        assert "boundaries" in descriptors
    
    def test_extract_all_with_intensity(self, sample_binary_image, sample_grayscale_image):
        """Test extracting all descriptors with intensity."""
        descriptors = extract_all_descriptors(
            sample_binary_image,
            intensity_image=sample_grayscale_image
        )
        
        assert "regions" in descriptors


class TestShapeComparison:
    """Tests for shape comparison."""
    
    def test_compare_shapes(self):
        """Test shape comparison using Fourier descriptors."""
        desc1 = {"fourier_descriptors": [1.0, 0.5, 0.25, 0.1]}
        desc2 = {"fourier_descriptors": [1.0, 0.5, 0.25, 0.1]}
        
        similarity = compare_shapes(desc1, desc2)
        
        # Identical shapes should have high similarity
        assert similarity > 0.9
    
    def test_compare_different_shapes(self):
        """Test comparing different shapes."""
        desc1 = {"fourier_descriptors": [1.0, 0.5, 0.25, 0.1]}
        desc2 = {"fourier_descriptors": [1.0, 0.1, 0.05, 0.01]}
        
        similarity = compare_shapes(desc1, desc2)
        
        # Different shapes should have lower similarity
        assert 0 < similarity < 1
    
    def test_compare_missing_descriptors(self):
        """Test comparison with missing descriptors."""
        desc1 = {}
        desc2 = {}
        
        similarity = compare_shapes(desc1, desc2)
        
        assert similarity == 0.0
