"""
Classification Module for Road Surface Layer Analyzer
Classifies road surface layers based on extracted texture features.

CSC566 Image Processing Mini Project
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import ROAD_LAYERS, PROJECT_ROOT


class RoadLayerClassifier:
    """
    Classifier for identifying road construction layers based on texture features.
    
    Classes:
        1. Subgrade (In-site soil/backfill)
        2. Subbase Course (Crushed aggregate - coarse)
        3. Base Course (Crushed aggregate - fine)
        4. Binder Course (Premix asphalt)
        5. Surface Course (Premix asphalt - smooth)
    """
    
    def __init__(
        self,
        classifier_type: str = "random_forest",
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            classifier_type: Type of classifier ("random_forest" or "svm")
            n_estimators: Number of trees for Random Forest
            random_state: Random seed for reproducibility
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if classifier_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
        elif classifier_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.classes = ROAD_LAYERS
        self.class_names = [self.classes[i]["name"] for i in sorted(self.classes.keys())]
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) with values 1-5
            validation_split: Fraction for validation set
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average='weighted'),
            "recall": recall_score(y_val, y_pred, average='weighted'),
            "f1": f1_score(y_val, y_pred, average='weighted')
        }
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict road layer class for features.
        
        Args:
            X: Feature matrix (n_samples, n_features) or (n_features,) for single sample
            return_proba: Whether to return class probabilities
            
        Returns:
            Predictions, and optionally probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        if return_proba:
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        
        return predictions
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """
        Predict road layer for a single feature vector.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with prediction details
        """
        predictions, probabilities = self.predict(features, return_proba=True)
        
        predicted_class = int(predictions[0])
        confidence = float(probabilities[0].max())
        
        layer_info = self.classes.get(predicted_class, {})
        
        return {
            "layer_number": predicted_class,
            "layer_name": layer_info.get("name", "Unknown"),
            "full_name": layer_info.get("full_name", "Unknown"),
            "material": layer_info.get("material", "Unknown"),
            "confidence": confidence,
            "all_probabilities": {
                self.classes[i+1]["name"]: float(probabilities[0][i])
                for i in range(len(probabilities[0]))
            }
        }
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Evaluate classifier on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X, return_proba=False)
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted'),
            "recall": recall_score(y, predictions, average='weighted'),
            "f1": f1_score(y, predictions, average='weighted'),
            "confusion_matrix": confusion_matrix(y, predictions).tolist(),
            "classification_report": classification_report(
                y, predictions, 
                target_names=self.class_names,
                output_dict=True
            )
        }
        
        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.class_names):
            class_label = i + 1
            mask = y == class_label
            if mask.any():
                per_class[name] = {
                    "accuracy": accuracy_score(y[mask], predictions[mask]),
                    "support": int(mask.sum())
                }
        metrics["per_class"] = per_class
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            "cv_scores": scores.tolist(),
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std())
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        save_data = {
            "model": self.model,
            "scaler": self.scaler,
            "classifier_type": self.classifier_type,
            "classes": self.classes,
            "class_names": self.class_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data["model"]
        self.scaler = save_data["scaler"]
        self.classifier_type = save_data["classifier_type"]
        self.classes = save_data["classes"]
        self.class_names = save_data["class_names"]
        self.is_trained = True
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (for Random Forest).
        
        Returns:
            Dictionary mapping feature index to importance, or None
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        
        return None


def classify_from_texture_features(
    features: Dict,
    classifier: RoadLayerClassifier
) -> Dict:
    """
    Classify road layer from extracted texture features dictionary.
    
    Args:
        features: Dictionary with texture features (from texture_features.py)
        classifier: Trained classifier
        
    Returns:
        Classification result
    """
    from .texture_features import features_to_vector
    
    feature_vector = features_to_vector(features)
    return classifier.predict_single(feature_vector)


def get_layer_by_texture_heuristic(features: Dict) -> Dict:
    """
    Classify road layer using heuristic rules based on texture features.
    Useful when classifier is not trained.
    
    Args:
        features: Dictionary with texture features
        
    Returns:
        Classification result with heuristics
    """
    # Extract key features
    glcm = features.get("glcm", {})
    statistical = features.get("statistical", {})
    
    contrast = glcm.get("contrast", 0)
    homogeneity = glcm.get("homogeneity", 0)
    energy = glcm.get("energy", 0)
    mean_intensity = statistical.get("mean", 128)
    smoothness = statistical.get("smoothness", 0)
    
    # Heuristic classification based on texture properties
    # Surface Course: smooth, uniform, dark
    if homogeneity > 0.85 and smoothness > 0.8 and mean_intensity < 80:
        layer = 5
        confidence = min(0.9, homogeneity)
    # Binder Course: semi-smooth, dark
    elif homogeneity > 0.7 and mean_intensity < 100:
        layer = 4
        confidence = min(0.8, homogeneity)
    # Base Course: moderate texture, gray
    elif 0.5 < homogeneity < 0.75 and 100 < mean_intensity < 150:
        layer = 3
        confidence = 0.7
    # Subbase Course: rough, varied
    elif contrast > 50 and homogeneity < 0.6:
        layer = 2
        confidence = 0.65
    # Subgrade: earth tones, irregular
    else:
        layer = 1
        confidence = 0.6
    
    layer_info = ROAD_LAYERS.get(layer, {})
    
    return {
        "layer_number": layer,
        "layer_name": layer_info.get("name", "Unknown"),
        "full_name": layer_info.get("full_name", "Unknown"),
        "material": layer_info.get("material", "Unknown"),
        "confidence": confidence,
        "method": "heuristic"
    }
