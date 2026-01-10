"""
Integration Tests for Road Surface Layer Analyzer
Tests complete pipeline from image to classification result.

CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.preprocessing import preprocess_image
from src.texture_features import extract_all_texture_features, features_to_vector
from src.segmentation import kmeans_segment, slic_segment
from src.morphology import opening, closing, fill_holes
from src.classification import RoadLayerClassifier, get_layer_by_texture_heuristic
from src.visualization import create_colored_segmentation


class TestFullPipeline:
    """Integration tests for complete processing pipeline."""
    
    def test_classical_pipeline(self, sample_color_image):
        """Test complete classical analysis pipeline."""
        # Step 1: Preprocess
        preprocessed = preprocess_image(
            sample_color_image,
            denoise="median",
            enhance="clahe",
            color_space="gray"
        )
        assert preprocessed is not None
        
        # Step 2: Extract texture features
        features = extract_all_texture_features(preprocessed, use_glcm=True, use_lbp=True)
        assert "glcm" in features
        assert "lbp" in features
        
        # Step 3: Segment
        labels, centers = kmeans_segment(sample_color_image, n_clusters=5)
        assert labels.shape == sample_color_image.shape[:2]
        
        # Step 4: Classify using heuristic
        result = get_layer_by_texture_heuristic(features)
        assert 1 <= result["layer_number"] <= 5
        
        # Step 5: Create visualization
        colored = create_colored_segmentation(labels + 1)
        assert colored.shape == sample_color_image.shape
    
    def test_pipeline_with_superpixels(self, sample_color_image):
        """Test pipeline with superpixel segmentation."""
        # Preprocess
        preprocessed = preprocess_image(sample_color_image, denoise="gaussian")
        
        # Superpixel segmentation
        segments = slic_segment(sample_color_image, n_segments=100)
        
        # Should have multiple segments
        assert len(np.unique(segments)) > 50
    
    def test_pipeline_with_morphology(self, sample_textured_image):
        """Test pipeline includes morphological operations."""
        # Segment
        labels, _ = kmeans_segment(sample_textured_image, n_clusters=3)
        
        # Create binary mask for one label
        binary = (labels == 0).astype(np.uint8) * 255
        
        # Apply morphological operations
        opened = opening(binary, kernel_size=5)
        closed = closing(opened, kernel_size=5)
        
        assert closed.shape == binary.shape


class TestFeatureClassificationPipeline:
    """Tests for feature extraction to classification pipeline."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Create and train a classifier."""
        np.random.seed(42)
        X = np.random.randn(100, 18).astype(np.float32)
        y = np.random.randint(1, 6, 100)
        
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        return classifier
    
    def test_features_to_classification(self, sample_grayscale_image, trained_classifier):
        """Test from feature extraction to classification."""
        # Extract features
        features = extract_all_texture_features(
            sample_grayscale_image,
            use_glcm=True,
            use_lbp=True,
            use_gabor=False,
            use_statistical=True
        )
        
        # Convert to vector
        vector = features_to_vector(features)
        assert len(vector) == 18
        
        # Classify
        result = trained_classifier.predict_single(vector)
        
        assert "layer_number" in result
        assert "confidence" in result
        assert 1 <= result["layer_number"] <= 5
    
    def test_multiple_regions_classification(self, sample_textured_image, trained_classifier):
        """Test classifying multiple regions in one image."""
        # Segment into regions
        labels, _ = kmeans_segment(sample_textured_image, n_clusters=4)
        
        results = []
        for label_id in range(4):
            # Extract region
            mask = labels == label_id
            region = sample_textured_image.copy()
            region[~mask] = 0
            
            # Only process non-empty regions
            if mask.sum() > 100:
                features = extract_all_texture_features(region)
                vector = features_to_vector(features)
                result = trained_classifier.predict_single(vector)
                results.append(result)
        
        assert len(results) > 0


class TestEndToEndWorkflow:
    """Tests simulating real-world usage."""
    
    def test_batch_processing(self, sample_color_image):
        """Test processing multiple images in batch."""
        images = [sample_color_image.copy() for _ in range(3)]
        
        results = []
        for img in images:
            preprocessed = preprocess_image(img, denoise="median", color_space="gray")
            features = extract_all_texture_features(preprocessed)
            result = get_layer_by_texture_heuristic(features)
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert "layer_name" in result
    
    def test_different_preprocessing_options(self, sample_color_image):
        """Test pipeline works with different preprocessing configurations."""
        configs = [
            {"denoise": "median", "enhance": "clahe"},
            {"denoise": "gaussian", "enhance": "histogram_eq"},
            {"denoise": "bilateral", "enhance": "gamma"}
        ]
        
        for config in configs:
            preprocessed = preprocess_image(sample_color_image, **config, color_space="gray")
            features = extract_all_texture_features(preprocessed, use_glcm=True)
            
            assert "glcm" in features
    
    def test_error_handling(self):
        """Test pipeline handles edge cases gracefully."""
        # Very small image
        small_image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        
        preprocessed = preprocess_image(small_image, denoise="median")
        features = extract_all_texture_features(preprocessed)
        
        # Should still produce features
        assert features is not None


class TestVisualizationIntegration:
    """Tests for visualization integration."""
    
    def test_colored_output(self, sample_color_image):
        """Test creating colored segmentation output."""
        labels, _ = kmeans_segment(sample_color_image, n_clusters=5)
        
        # Convert to 1-indexed
        labels_1_indexed = labels + 1
        
        colored = create_colored_segmentation(labels_1_indexed)
        
        assert colored.shape == sample_color_image.shape
        assert colored.dtype == np.uint8
    
    def test_result_overlay(self, sample_color_image):
        """Test creating result overlay."""
        labels, _ = kmeans_segment(sample_color_image, n_clusters=5)
        labels_1_indexed = labels + 1
        
        colored = create_colored_segmentation(labels_1_indexed)
        
        # Create overlay
        overlay = cv2.addWeighted(sample_color_image, 0.6, colored, 0.4, 0)
        
        assert overlay.shape == sample_color_image.shape
