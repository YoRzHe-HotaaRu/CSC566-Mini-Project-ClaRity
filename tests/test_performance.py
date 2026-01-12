"""
Performance Benchmark Tests for Road Surface Layer Analyzer
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
import time
from functools import wraps


def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


class TestPreprocessingPerformance:
    """Performance tests for preprocessing module."""
    
    def test_noise_filter_performance(self, sample_color_image):
        """Benchmark noise filtering speed."""
        from src.preprocessing import apply_noise_filter
        
        # Warm up
        apply_noise_filter(sample_color_image, method="median")
        
        # Benchmark
        iterations = 10
        start = time.perf_counter()
        
        for _ in range(iterations):
            apply_noise_filter(sample_color_image, method="median", kernel_size=5)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # Should complete in reasonable time (< 100ms per image)
        assert avg_time < 0.1, f"Median filter too slow: {avg_time:.3f}s"
    
    def test_clahe_performance(self, sample_color_image):
        """Benchmark CLAHE enhancement speed."""
        from src.preprocessing import enhance_contrast
        
        # Warm up
        enhance_contrast(sample_color_image, method="clahe")
        
        # Benchmark
        iterations = 10
        start = time.perf_counter()
        
        for _ in range(iterations):
            enhance_contrast(sample_color_image, method="clahe")
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        assert avg_time < 0.1, f"CLAHE too slow: {avg_time:.3f}s"
    
    def test_preprocessing_pipeline_performance(self, sample_color_image):
        """Benchmark full preprocessing pipeline."""
        from src.preprocessing import preprocess_image
        
        iterations = 5
        start = time.perf_counter()
        
        for _ in range(iterations):
            preprocess_image(
                sample_color_image,
                denoise="median",
                enhance="clahe",
                color_space="gray"
            )
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        assert avg_time < 0.2, f"Preprocessing pipeline too slow: {avg_time:.3f}s"


class TestTextureFeaturePerformance:
    """Performance tests for texture feature extraction."""
    
    def test_glcm_performance(self, sample_grayscale_image):
        """Benchmark GLCM feature extraction."""
        from src.texture_features import extract_glcm_features
        
        # Warm up
        extract_glcm_features(sample_grayscale_image)
        
        # Benchmark
        iterations = 5
        start = time.perf_counter()
        
        for _ in range(iterations):
            extract_glcm_features(sample_grayscale_image)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # GLCM can be slower, allow up to 500ms
        assert avg_time < 0.5, f"GLCM too slow: {avg_time:.3f}s"
    
    def test_lbp_performance(self, sample_grayscale_image):
        """Benchmark LBP feature extraction."""
        from src.texture_features import extract_lbp_features
        
        # Warm up
        extract_lbp_features(sample_grayscale_image)
        
        # Benchmark
        iterations = 10
        start = time.perf_counter()
        
        for _ in range(iterations):
            extract_lbp_features(sample_grayscale_image)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        assert avg_time < 0.2, f"LBP too slow: {avg_time:.3f}s"
    
    def test_all_features_performance(self, sample_grayscale_image):
        """Benchmark extracting all texture features."""
        from src.texture_features import extract_all_texture_features
        
        iterations = 3
        start = time.perf_counter()
        
        for _ in range(iterations):
            extract_all_texture_features(
                sample_grayscale_image,
                use_glcm=True,
                use_lbp=True,
                use_gabor=False,  # Gabor is slow
                use_statistical=True
            )
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # Allow up to 1 second for all features
        assert avg_time < 1.0, f"All features too slow: {avg_time:.3f}s"


class TestSegmentationPerformance:
    """Performance tests for segmentation."""
    
    def test_kmeans_performance(self, sample_color_image):
        """Benchmark K-Means segmentation."""
        from src.segmentation import kmeans_segment
        
        # Warm up
        kmeans_segment(sample_color_image, n_clusters=5)
        
        # Benchmark
        iterations = 3
        start = time.perf_counter()
        
        for _ in range(iterations):
            kmeans_segment(sample_color_image, n_clusters=5)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # K-Means should be reasonable
        assert avg_time < 2.0, f"K-Means too slow: {avg_time:.3f}s"
    
    def test_slic_performance(self, sample_color_image):
        """Benchmark SLIC superpixel segmentation."""
        from src.segmentation import slic_segment
        
        # Warm up
        slic_segment(sample_color_image, n_segments=100)
        
        # Benchmark
        iterations = 3
        start = time.perf_counter()
        
        for _ in range(iterations):
            slic_segment(sample_color_image, n_segments=100)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        assert avg_time < 1.0, f"SLIC too slow: {avg_time:.3f}s"


class TestClassificationPerformance:
    """Performance tests for classification."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier for benchmarking."""
        from src.classification import RoadLayerClassifier
        
        np.random.seed(42)
        X = np.random.randn(200, 18).astype(np.float32)
        y = np.random.randint(1, 6, 200)
        
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        return classifier
    
    def test_prediction_performance(self, trained_classifier):
        """Benchmark prediction speed."""
        np.random.seed(42)
        X_test = np.random.randn(100, 18).astype(np.float32)
        
        # Warm up
        trained_classifier.predict(X_test[:10])
        
        # Benchmark
        iterations = 10
        start = time.perf_counter()
        
        for _ in range(iterations):
            trained_classifier.predict(X_test)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # Prediction should be reasonably fast (adjusted for different machines)
        assert avg_time < 0.15, f"Prediction too slow: {avg_time:.3f}s"
    
    def test_single_prediction_performance(self, trained_classifier):
        """Benchmark single sample prediction."""
        np.random.seed(42)
        sample = np.random.randn(18).astype(np.float32)
        
        # Benchmark
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            trained_classifier.predict_single(sample)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # Single prediction should be < 100ms (adjusted for different machines)
        assert avg_time < 0.10, f"Single prediction too slow: {avg_time * 1000:.2f}ms"


class TestFullPipelinePerformance:
    """Performance tests for complete pipeline."""
    
    def test_full_classical_pipeline(self, sample_color_image):
        """Benchmark complete classical analysis pipeline."""
        from src.preprocessing import preprocess_image
        from src.texture_features import extract_all_texture_features
        from src.segmentation import kmeans_segment
        from src.classification import get_layer_by_texture_heuristic
        
        start = time.perf_counter()
        
        # Step 1: Preprocess
        preprocessed = preprocess_image(
            sample_color_image,
            denoise="median",
            enhance="clahe",
            color_space="gray"
        )
        
        # Step 2: Extract features
        features = extract_all_texture_features(
            preprocessed,
            use_glcm=True,
            use_lbp=True,
            use_gabor=False,
            use_statistical=True
        )
        
        # Step 3: Segment
        labels, _ = kmeans_segment(sample_color_image, n_clusters=5)
        
        # Step 4: Classify
        result = get_layer_by_texture_heuristic(features)
        
        elapsed = time.perf_counter() - start
        
        # Full pipeline should complete in < 5 seconds
        assert elapsed < 5.0, f"Full pipeline too slow: {elapsed:.2f}s"
        
        # Verify result is valid
        assert 1 <= result["layer_number"] <= 5


class TestMemoryUsage:
    """Tests for memory efficiency."""
    
    def test_large_image_handling(self):
        """Test handling of larger images."""
        from src.preprocessing import preprocess_image
        from src.segmentation import kmeans_segment
        
        # Create larger image (1024x1024)
        large_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        
        # Should complete without memory error
        preprocessed = preprocess_image(large_image, denoise="median", color_space="gray")
        labels, _ = kmeans_segment(large_image, n_clusters=5)
        
        assert preprocessed.shape == (1024, 1024)
        assert labels.shape == (1024, 1024)
    
    def test_batch_processing_memory(self, sample_color_image):
        """Test memory usage during batch processing."""
        from src.preprocessing import preprocess_image
        from src.texture_features import extract_all_texture_features
        
        # Process multiple images
        results = []
        for i in range(10):
            preprocessed = preprocess_image(sample_color_image, color_space="gray")
            features = extract_all_texture_features(preprocessed, use_glcm=True)
            results.append(features)
        
        assert len(results) == 10
