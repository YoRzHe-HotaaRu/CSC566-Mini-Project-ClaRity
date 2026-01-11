"""
Unit Tests for VLM Analyzer Module
CSC566 Image Processing Mini Project
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json


class TestVLMAnalyzer:
    """Tests for Vision Language Model analyzer."""
    
    def test_init_with_api_key(self):
        """Test VLM Analyzer initialization with API key."""
        from src.vlm_analyzer import VLMAnalyzer
        
        analyzer = VLMAnalyzer(api_key="test_key")
        
        assert analyzer.api_key == "test_key"
        assert analyzer.model == "z-ai/glm-4.6v"
    
    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        from src.vlm_analyzer import VLMAnalyzer
        
        # Temporarily clear the environment variable
        with patch.dict('os.environ', {'ZENMUX_API_KEY': ''}, clear=False):
            with patch('src.vlm_analyzer.ZENMUX_API_KEY', ''):
                with pytest.raises(ValueError):
                    VLMAnalyzer(api_key=None)
    
    def test_encode_image_to_base64(self, temp_image_path):
        """Test image encoding to base64."""
        from src.vlm_analyzer import VLMAnalyzer
        
        analyzer = VLMAnalyzer(api_key="test_key")
        encoded = analyzer._encode_image_to_base64(temp_image_path)
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Should be valid base64
        import base64
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0
    
    def test_get_image_mime_type(self):
        """Test MIME type detection."""
        from src.vlm_analyzer import VLMAnalyzer
        
        analyzer = VLMAnalyzer(api_key="test_key")
        
        assert analyzer._get_image_mime_type("test.jpg") == "image/jpeg"
        assert analyzer._get_image_mime_type("test.jpeg") == "image/jpeg"
        assert analyzer._get_image_mime_type("test.png") == "image/png"
        assert analyzer._get_image_mime_type("test.gif") == "image/gif"
        assert analyzer._get_image_mime_type("test.webp") == "image/webp"
    
    def test_is_available(self):
        """Test API availability check."""
        from src.vlm_analyzer import VLMAnalyzer
        
        analyzer = VLMAnalyzer(api_key="test_key")
        
        assert analyzer.is_available() == True
        
        analyzer_no_key = VLMAnalyzer.__new__(VLMAnalyzer)
        analyzer_no_key.api_key = ""
        
        assert analyzer_no_key.is_available() == False
    
    @patch('requests.post')
    def test_make_request_success(self, mock_post):
        """Test successful API request."""
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer._make_request([{"role": "user", "content": "test"}])
        
        assert "choices" in result
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_make_request_timeout(self, mock_post):
        """Test API request timeout handling."""
        import requests
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_post.side_effect = requests.exceptions.Timeout()
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer._make_request([{"role": "user", "content": "test"}])
        
        assert "error" in result
        assert "timed out" in result["error"].lower()
    
    @patch('requests.post')
    def test_analyze_road_layer_success(self, mock_post, temp_image_path):
        """Test successful road layer analysis."""
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Layer Number: 5\nLayer Name: Surface Course\nConfidence: 85%\nMaterial: Premix asphalt"
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer.analyze_road_layer(temp_image_path)
        
        assert result["success"] == True
        assert "raw_response" in result
    
    @patch('requests.post')
    def test_analyze_road_layer_error(self, mock_post, temp_image_path):
        """Test road layer analysis with API error."""
        import requests
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer.analyze_road_layer(temp_image_path)
        
        assert result["success"] == False
        assert "error" in result
    
    def test_parse_analysis_response_layer_detection(self):
        """Test parsing VLM response for layer detection."""
        from src.vlm_analyzer import VLMAnalyzer
        
        analyzer = VLMAnalyzer(api_key="test_key")
        
        # Test layer 5 detection
        response = "This appears to be Layer 5: Surface Course with confidence 90%"
        result = analyzer._parse_analysis_response(response)
        
        assert result["layer_number"] == 5
        assert result["layer_name"] == "Surface Course"
        assert result["confidence"] > 0
    
    def test_parse_analysis_response_confidence_extraction(self):
        """Test confidence extraction from response."""
        from src.vlm_analyzer import VLMAnalyzer
        
        analyzer = VLMAnalyzer(api_key="test_key")
        
        # Test percentage format
        response = "Subgrade layer detected. Confidence: 75%"
        result = analyzer._parse_analysis_response(response)
        
        assert result["confidence"] == 0.75
    
    @patch('requests.post')
    def test_get_detailed_analysis(self, mock_post, temp_image_path):
        """Test detailed analysis request."""
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Detailed analysis..."}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer.get_detailed_analysis(temp_image_path)
        
        assert result["success"] == True
    
    @patch('requests.post')
    def test_compare_layers(self, mock_post, temp_image_path):
        """Test comparing multiple images."""
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Layer 3 - Base Course"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer.compare_layers([temp_image_path, temp_image_path])
        
        assert "individual_results" in result
        assert "summary" in result
        assert len(result["individual_results"]) == 2
    
    @patch('requests.post')
    def test_test_connection_success(self, mock_post):
        """Test API connection test."""
        from src.vlm_analyzer import VLMAnalyzer
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        analyzer = VLMAnalyzer(api_key="test_key")
        result = analyzer.test_connection()
        
        assert result["success"] == True
