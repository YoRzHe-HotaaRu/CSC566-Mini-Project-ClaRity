"""
Unit Tests for Deep Learning Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestDeepLabSegmenter:
    """Tests for DeepLabv3+ segmentation (requires PyTorch)."""
    
    def test_check_cuda_available(self):
        """Test CUDA availability check."""
        from src.deep_learning import check_cuda_available
        
        result = check_cuda_available()
        
        assert "available" in result
        assert isinstance(result["available"], bool)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_segmenter_init_cpu(self):
        """Test DeepLabSegmenter initialization on CPU."""
        try:
            from src.deep_learning import DeepLabSegmenter
            
            segmenter = DeepLabSegmenter(use_cuda=False)
            
            assert segmenter.device.type == "cpu"
            assert segmenter.model is not None
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_preprocess(self, sample_color_image):
        """Test image preprocessing for model input."""
        try:
            from src.deep_learning import DeepLabSegmenter
            
            segmenter = DeepLabSegmenter(use_cuda=False)
            tensor = segmenter.preprocess(sample_color_image)
            
            assert tensor.dim() == 4  # Batch, Channel, H, W
            assert tensor.shape[0] == 1  # Batch size
            assert tensor.shape[1] == 3  # RGB channels
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_segment(self, sample_color_image):
        """Test segmentation inference."""
        try:
            from src.deep_learning import DeepLabSegmenter
            
            segmenter = DeepLabSegmenter(use_cuda=False)
            labels = segmenter.segment(sample_color_image)
            
            assert labels.shape == sample_color_image.shape[:2]
            assert labels.dtype == np.uint8
            
            # Labels should be 1-5 for road layers
            unique = np.unique(labels)
            assert all(1 <= l <= 5 for l in unique)
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_segment_with_proba(self, sample_color_image):
        """Test segmentation with probability output."""
        try:
            from src.deep_learning import DeepLabSegmenter
            
            segmenter = DeepLabSegmenter(use_cuda=False)
            labels, proba = segmenter.segment(sample_color_image, return_proba=True)
            
            assert labels.shape == sample_color_image.shape[:2]
            assert proba.shape[0] == 5  # 5 classes
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_create_colored_output(self, labels_5_class):
        """Test creating colored segmentation output."""
        try:
            from src.deep_learning import DeepLabSegmenter
            
            segmenter = DeepLabSegmenter(use_cuda=False)
            colored = segmenter.create_colored_output(labels_5_class)
            
            assert colored.shape == (*labels_5_class.shape, 3)
            assert colored.dtype == np.uint8
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestRoadLayerDataset:
    """Tests for PyTorch Dataset."""
    
    def test_dataset_creation(self, temp_image_path):
        """Test dataset creation."""
        try:
            from src.deep_learning import RoadLayerDataset
            
            dataset = RoadLayerDataset(
                image_paths=[temp_image_path],
                mask_paths=None,
                target_size=(256, 256)
            )
            
            assert len(dataset) == 1
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")
    
    def test_dataset_getitem(self, temp_image_path):
        """Test dataset item retrieval."""
        try:
            from src.deep_learning import RoadLayerDataset
            
            dataset = RoadLayerDataset(
                image_paths=[temp_image_path],
                mask_paths=None,
                target_size=(256, 256)
            )
            
            image = dataset[0]
            
            assert image.shape == (3, 256, 256)  # C, H, W
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")


class TestDeepLearningFallback:
    """Tests for when PyTorch is not available."""
    
    def test_import_without_torch(self):
        """Test graceful handling when PyTorch unavailable."""
        # This just verifies the module structure allows graceful degradation
        try:
            from src import deep_learning
            assert hasattr(deep_learning, 'TORCH_AVAILABLE')
        except ImportError:
            # Expected if dependencies missing
            pass
