"""
Unit Tests for Classification Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.classification import (
    RoadLayerClassifier,
    classify_from_texture_features,
    get_layer_by_texture_heuristic
)
from src.config import ROAD_LAYERS


class TestRoadLayerClassifier:
    """Tests for RoadLayerClassifier class."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 18
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(1, 6, n_samples)  # Classes 1-5
        
        return X, y
    
    def test_classifier_init_random_forest(self):
        """Test Random Forest classifier initialization."""
        classifier = RoadLayerClassifier(classifier_type="random_forest")
        
        assert classifier.classifier_type == "random_forest"
        assert classifier.is_trained == False
    
    def test_classifier_init_svm(self):
        """Test SVM classifier initialization."""
        classifier = RoadLayerClassifier(classifier_type="svm")
        
        assert classifier.classifier_type == "svm"
    
    def test_classifier_init_invalid(self):
        """Test invalid classifier type raises error."""
        with pytest.raises(ValueError):
            RoadLayerClassifier(classifier_type="invalid")
    
    def test_train_classifier(self, sample_training_data):
        """Test classifier training."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier()
        
        metrics = classifier.train(X, y, validation_split=0.2)
        
        assert classifier.is_trained == True
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_predict_untrained(self, sample_training_data):
        """Test prediction with untrained classifier raises error."""
        X, _ = sample_training_data
        classifier = RoadLayerClassifier()
        
        with pytest.raises(RuntimeError):
            classifier.predict(X)
    
    def test_predict_single(self, sample_training_data):
        """Test single sample prediction."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        
        result = classifier.predict_single(X[0])
        
        assert "layer_number" in result
        assert "layer_name" in result
        assert "confidence" in result
        
        assert 1 <= result["layer_number"] <= 5
        assert 0 <= result["confidence"] <= 1
    
    def test_predict_batch(self, sample_training_data):
        """Test batch prediction."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        
        predictions, probabilities = classifier.predict(X[:10], return_proba=True)
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 5)
    
    def test_evaluate(self, sample_training_data):
        """Test classifier evaluation."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        
        metrics = classifier.evaluate(X[:20], y[:20])
        
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class" in metrics
    
    def test_cross_validate(self, sample_training_data):
        """Test cross-validation."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier()
        
        cv_results = classifier.cross_validate(X, y, cv=3)
        
        assert "cv_scores" in cv_results
        assert "mean_accuracy" in cv_results
        assert len(cv_results["cv_scores"]) == 3
    
    def test_save_load_model(self, sample_training_data):
        """Test saving and loading model."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            classifier.save(filepath)
            assert Path(filepath).exists()
            
            # Load in new classifier
            new_classifier = RoadLayerClassifier()
            new_classifier.load(filepath)
            
            assert new_classifier.is_trained == True
            
            # Predictions should match
            pred1 = classifier.predict(X[:5], return_proba=False)
            pred2 = new_classifier.predict(X[:5], return_proba=False)
            
            np.testing.assert_array_equal(pred1, pred2)
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_feature_importance(self, sample_training_data):
        """Test feature importance for Random Forest."""
        X, y = sample_training_data
        classifier = RoadLayerClassifier(classifier_type="random_forest")
        classifier.train(X, y)
        
        importance = classifier.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == X.shape[1]


class TestHeuristicClassification:
    """Tests for heuristic classification function."""
    
    def test_heuristic_smooth_surface(self):
        """Test heuristic detects smooth surface layer."""
        features = {
            "glcm": {
                "contrast": 0.1,
                "homogeneity": 0.95,
                "energy": 0.8
            },
            "statistical": {
                "mean": 50.0,
                "smoothness": 0.9
            }
        }
        
        result = get_layer_by_texture_heuristic(features)
        
        assert result["layer_number"] == 5  # Surface course
        assert result["method"] == "heuristic"
    
    def test_heuristic_rough_subgrade(self):
        """Test heuristic detects rough subgrade."""
        features = {
            "glcm": {
                "contrast": 100.0,
                "homogeneity": 0.3,
                "energy": 0.1
            },
            "statistical": {
                "mean": 150.0,
                "smoothness": 0.4
            }
        }
        
        result = get_layer_by_texture_heuristic(features)
        
        # High contrast, low homogeneity -> lower layers
        assert result["layer_number"] in [1, 2]
    
    def test_heuristic_returns_required_fields(self, sample_road_features):
        """Test heuristic returns all required fields."""
        result = get_layer_by_texture_heuristic(sample_road_features)
        
        assert "layer_number" in result
        assert "layer_name" in result
        assert "full_name" in result
        assert "material" in result
        assert "confidence" in result
        assert "method" in result
    
    def test_heuristic_confidence_range(self, sample_road_features):
        """Test heuristic confidence is in valid range."""
        result = get_layer_by_texture_heuristic(sample_road_features)
        
        assert 0 <= result["confidence"] <= 1


class TestClassifyFromFeatures:
    """Tests for classify_from_texture_features function."""
    
    def test_classify_integration(self, sample_road_features):
        """Test classification from feature dictionary."""
        # Create and train a classifier
        np.random.seed(42)
        X = np.random.randn(50, 18).astype(np.float32)
        y = np.random.randint(1, 6, 50)
        
        classifier = RoadLayerClassifier()
        classifier.train(X, y)
        
        result = classify_from_texture_features(sample_road_features, classifier)
        
        assert "layer_number" in result
        assert 1 <= result["layer_number"] <= 5
