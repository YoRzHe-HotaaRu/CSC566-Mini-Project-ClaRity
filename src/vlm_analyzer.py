"""
VLM Analyzer Module for Road Surface Layer Analyzer
Integrates GLM-4.6V Vision Language Model via ZenMux API.

CSC566 Image Processing Mini Project
"""

import base64
import json
import requests
from pathlib import Path
from typing import Dict, Optional, Union

from .config import (
    ZENMUX_API_KEY,
    ZENMUX_BASE_URL,
    VLM_MODEL,
    VLM_CONFIG,
    ROAD_ANALYSIS_PROMPT,
    ROAD_LAYERS
)


class VLMAnalyzer:
    """
    Vision Language Model analyzer using GLM-4.6V via ZenMux API.
    Provides AI-powered analysis of road construction layers.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize VLM Analyzer.
        
        Args:
            api_key: ZenMux API key (defaults to config)
            base_url: API base URL (defaults to config)
            model: Model name (defaults to config)
        """
        self.api_key = api_key or ZENMUX_API_KEY
        self.base_url = base_url or ZENMUX_BASE_URL
        self.model = model or VLM_MODEL
        self.endpoint = f"{self.base_url}/chat/completions"
        
        if not self.api_key:
            raise ValueError("ZenMux API key not provided. Set ZENMUX_API_KEY in .env")
    
    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _get_image_mime_type(self, image_path: Union[str, Path]) -> str:
        """
        Get MIME type for image based on extension.
        
        Args:
            image_path: Path to image file
            
        Returns:
            MIME type string
        """
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        return mime_types.get(ext, "image/jpeg")
    
    def _make_request(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict:
        """
        Make API request to ZenMux.
        
        Args:
            messages: List of message objects
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            API response dictionary
        """
        temperature = temperature or VLM_CONFIG["temperature"]
        max_tokens = max_tokens or VLM_CONFIG["max_tokens"]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=VLM_CONFIG["timeout"]
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def analyze_road_layer(
        self,
        image_path: Union[str, Path],
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        Analyze image to identify road construction layer.
        
        Args:
            image_path: Path to image file
            custom_prompt: Optional custom analysis prompt
            
        Returns:
            Analysis result dictionary
        """
        # Encode image
        base64_image = self._encode_image_to_base64(image_path)
        mime_type = self._get_image_mime_type(image_path)
        data_url = f"data:{mime_type};base64,{base64_image}"
        
        # Prepare prompt
        prompt = custom_prompt or ROAD_ANALYSIS_PROMPT
        
        # Create message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]
        
        # Make request
        response = self._make_request(messages)
        
        if "error" in response:
            return {
                "success": False,
                "error": response["error"],
                "layer_number": None,
                "layer_name": None,
                "confidence": 0
            }
        
        # Parse response
        try:
            content = response["choices"][0]["message"]["content"]
            result = self._parse_analysis_response(content)
            result["success"] = True
            result["raw_response"] = content
            return result
        
        except (KeyError, IndexError) as e:
            return {
                "success": False,
                "error": f"Failed to parse response: {e}",
                "layer_number": None,
                "layer_name": None,
                "confidence": 0
            }
    
    def _parse_analysis_response(self, content: str) -> Dict:
        """
        Parse VLM response to extract structured data.
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed result dictionary
        """
        result = {
            "layer_number": None,
            "layer_name": None,
            "confidence": 0.0,
            "material": None,
            "texture_description": None,
            "additional_notes": None
        }
        
        content_lower = content.lower()
        
        # Try to extract layer number
        for layer_num in range(1, 6):
            layer_info = ROAD_LAYERS[layer_num]
            layer_name = layer_info["name"].lower()
            
            if f"layer {layer_num}" in content_lower or layer_name in content_lower:
                result["layer_number"] = layer_num
                result["layer_name"] = layer_info["name"]
                result["full_name"] = layer_info["full_name"]
                result["material"] = layer_info["material"]
                break
        
        # Try to extract confidence
        import re
        confidence_patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*confidence",
            r"confidence[:\s]+(\d+(?:\.\d+)?)"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content_lower)
            if match:
                conf_value = float(match.group(1))
                # Normalize to 0-1 if percentage
                result["confidence"] = conf_value / 100 if conf_value > 1 else conf_value
                break
        
        # If no confidence found, estimate based on certainty words
        if result["confidence"] == 0:
            if any(word in content_lower for word in ["definitely", "clearly", "certainly"]):
                result["confidence"] = 0.9
            elif any(word in content_lower for word in ["likely", "probably", "appears"]):
                result["confidence"] = 0.7
            elif any(word in content_lower for word in ["possibly", "might", "could"]):
                result["confidence"] = 0.5
            else:
                result["confidence"] = 0.6
        
        # Extract texture description
        texture_keywords = ["texture", "surface", "pattern", "roughness"]
        for keyword in texture_keywords:
            if keyword in content_lower:
                # Try to find sentence with keyword
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        result["texture_description"] = sentence.strip()
                        break
                break
        
        return result
    
    def get_detailed_analysis(
        self,
        image_path: Union[str, Path]
    ) -> Dict:
        """
        Get comprehensive analysis of road layer image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Detailed analysis result
        """
        detailed_prompt = """
        Analyze this aerial satellite image of a road construction site in detail.
        
        Please provide:
        
        1. LAYER IDENTIFICATION
           - Which of the 5 road construction layers is shown?
           - Confidence level (0-100%)
        
        2. MATERIAL ANALYSIS
           - What materials can you observe?
           - Estimate of material composition
        
        3. SURFACE CONDITION
           - Surface roughness assessment
           - Any visible defects or issues?
        
        4. TEXTURE CHARACTERISTICS
           - Describe the texture pattern
           - Uniformity assessment
           - Color variations
        
        5. CONSTRUCTION STAGE
           - What stage of road construction is this?
           - Any recommendations for next steps?
        
        Road Layer Reference:
        1. Subgrade (in-site soil/backfill) - Earth/soil
        2. Subbase Course (crushed aggregate) - Coarse stones
        3. Base Course (crushed aggregate) - Finer aggregate
        4. Binder Course (premix asphalt) - Dark asphalt with visible aggregate
        5. Surface Course (premix asphalt) - Smooth asphalt finish
        """
        
        return self.analyze_road_layer(image_path, custom_prompt=detailed_prompt)
    
    def compare_layers(
        self,
        image_paths: list
    ) -> Dict:
        """
        Compare multiple road layer images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Comparison analysis
        """
        results = []
        
        for path in image_paths:
            result = self.analyze_road_layer(path)
            result["image_path"] = str(path)
            results.append(result)
        
        return {
            "individual_results": results,
            "summary": self._generate_comparison_summary(results)
        }
    
    def _generate_comparison_summary(self, results: list) -> Dict:
        """
        Generate summary of comparison results.
        
        Args:
            results: List of individual analysis results
            
        Returns:
            Summary dictionary
        """
        layers_found = []
        for r in results:
            if r.get("layer_number"):
                layers_found.append(r["layer_number"])
        
        return {
            "total_images": len(results),
            "successful_analyses": sum(1 for r in results if r.get("success")),
            "layers_identified": list(set(layers_found)),
            "average_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1)
        }
    
    def is_available(self) -> bool:
        """
        Check if VLM API is available and configured.
        
        Returns:
            True if API is available
        """
        return bool(self.api_key)
    
    def test_connection(self) -> Dict:
        """
        Test API connection with a simple request.
        
        Returns:
            Test result dictionary
        """
        messages = [
            {
                "role": "user",
                "content": "Hello, please respond with 'OK' if you can receive this message."
            }
        ]
        
        response = self._make_request(messages, max_tokens=10)
        
        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        return {"success": True, "message": "API connection successful"}
